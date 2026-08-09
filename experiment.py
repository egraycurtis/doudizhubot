"""Finite, resumable, metrics-first experiment runner.

The coordinator deliberately separates a normal producer stop from an abort:
on normal completion every batch accepted by Redis is trained and checkpointed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time

import redis

from compete import evaluate_families
from model_registry import get_model_config, get_model_dir, initialize_model_family
from self_play import self_play
from train import train


ROOT = Path(__file__).resolve().parent
RUN_IDENTITY_FIELDS = ("model_name", "source_model", "baseline_model", "use_payout_head", "model_schema_version")


def _cpu_evaluate(result_queue, baseline_model, challenger_model, deals, seed, challenger_uses_payout):
    """Spawned evaluator: reserve GPU exclusively for the learner process."""
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    result_queue.put(evaluate_families(baseline_model, challenger_model, deals=deals, seed=seed, challenger_uses_payout=challenger_uses_payout))


def _atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_metric(path: Path, metric: dict) -> None:
    metric["wall_time_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metric, sort_keys=True) + "\n")


def _stream_seed(*parts: object) -> int:
    """Stable non-overlapping seed namespace without numeric offset assumptions."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _run_identity(args) -> dict:
    identity = {field: getattr(args, field) for field in RUN_IDENTITY_FIELDS if hasattr(args, field)}
    identity["model_schema_version"] = get_model_config(args.model_name).schema_version
    return identity


def _prepare_run_directory(args, run_dir: Path) -> tuple[Path, int]:
    """Create a run or atomically append an explicit deterministic resume session."""
    config_path, sessions_path = run_dir / "config.json", run_dir / "sessions.json"
    if args.reset and args.resume:
        raise ValueError("--reset and --resume cannot be used together")
    if args.reset and run_dir.exists():
        archive = run_dir.with_name(f"{run_dir.name}.archived")
        suffix = 1
        while archive.exists():
            archive = run_dir.with_name(f"{run_dir.name}.archived-{suffix}")
            suffix += 1
        run_dir.rename(archive)
    if args.resume:
        if not config_path.exists() or not sessions_path.exists():
            raise ValueError(f"--resume requires an existing initialized run: {run_dir}")
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        expected = _run_identity(args)
        actual = {key: stored.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"resume configuration is incompatible: expected {expected}, found {actual}")
        sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
        epoch = len(sessions["sessions"])
    else:
        if config_path.exists() or get_model_dir(args.model_name).exists():
            raise ValueError("run or experimental model family already exists; choose a unique --run-name, use --resume, or use --reset")
        run_dir.mkdir(parents=True, exist_ok=True)
        epoch, sessions = 0, {"sessions": []}
        config = vars(args).copy()
        config.update(_run_identity(args))
        _atomic_write_json(config_path, config)
    session = {"epoch": epoch, "seed": _stream_seed(args.seed, args.run_name, args.model_name, epoch)}
    sessions["sessions"].append(session)
    _atomic_write_json(sessions_path, sessions)
    return run_dir / "metrics.jsonl", session["seed"]


def validate_args(args) -> None:
    positive = ("workers", "game_batch_size", "max_queue_items", "min_queue_items", "max_payloads", "reconstruction_workers", "eval_deals")
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_queue_items < args.min_queue_items:
        raise ValueError("--max-queue-items must be at least --min-queue-items")
    if args.duration < 0 or args.max_batches < 0 or args.eval_every < 0:
        raise ValueError("duration, max-batches, and eval-every cannot be negative")
    if args.eval_timeout <= 0 or args.shutdown_timeout <= 0:
        raise ValueError("eval-timeout and shutdown-timeout must be positive")
    if not 0 <= args.explore_rate <= 1:
        raise ValueError("--explore-rate must be between 0 and 1")
    if args.duration <= 0 and args.max_batches <= 0:
        raise ValueError("set --duration or --max-batches to a positive value")


def _drain_metrics(stats, totals, client, queue_key, metrics_path, elapsed, first_timeout=0) -> int:
    count = 0
    while True:
        try:
            metric = stats.get(timeout=first_timeout) if count == 0 and first_timeout else stats.get_nowait()
        except queue.Empty:
            return count
        count += 1
        if metric["kind"] == "generation":
            totals["games"] += metric["games"]
            totals["candidate_rows"] += metric["candidate_rows"]
            totals["generation_seconds"] += metric["batch_seconds"]
        elif metric["kind"] == "learner":
            totals["examples"] += metric["examples"]
            totals["replay_seconds"] += metric["replay_seconds"]
            totals["fit_seconds"] += metric["fit_seconds"]
            totals["checkpoint_seconds"] += metric["checkpoint_seconds"]
            totals["durable_checkpoints"] += int(metric.get("durable_checkpoint", False))
        _write_metric(metrics_path, {"event": metric["kind"], "elapsed_seconds": elapsed, "queue_items": client.llen(queue_key), **metric})


def _settle_metrics(stats, totals, client, queue_key, metrics_path, elapsed, quiet_seconds=0.2) -> None:
    """Allow multiprocessing Queue feeder threads to publish their final metric."""
    quiet_deadline = time.monotonic() + quiet_seconds
    while time.monotonic() < quiet_deadline:
        count = _drain_metrics(stats, totals, client, queue_key, metrics_path, elapsed, first_timeout=0.05)
        if count:
            quiet_deadline = time.monotonic() + quiet_seconds


def _wait_with_metrics(processes, stats, totals, client, queue_key, metrics_path, started, state, timeout=None):
    """Join children without letting a stats Queue feeder starve or deadlock them."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while any(process.is_alive() for process in processes):
        _drain_metrics(stats, totals, client, queue_key, metrics_path, time.monotonic() - started)
        _raise_if_failed(processes[0], processes[1:], state)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"timed out while {state}")
        for process in processes:
            process.join(timeout=0.05)
    _settle_metrics(stats, totals, client, queue_key, metrics_path, time.monotonic() - started)
    _raise_if_failed(processes[0], processes[1:], state)


def _abort(processes, stats, totals, client, queue_key, metrics_path, started):
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 15
    while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
        _drain_metrics(stats, totals, client, queue_key, metrics_path, time.monotonic() - started)
        for process in processes:
            process.join(timeout=0.05)
    _settle_metrics(stats, totals, client, queue_key, metrics_path, time.monotonic() - started)


def _wait_for_producers(learner, actors, stats, totals, client, queue_key, metrics_path, started, timeout):
    """Let actors finish a current batch while the learner continues consuming."""
    deadline = time.monotonic() + timeout
    while any(actor.is_alive() for actor in actors):
        _drain_metrics(stats, totals, client, queue_key, metrics_path, time.monotonic() - started)
        _raise_if_failed(learner, actors, "stopping_producers")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out while stopping producers")
        for actor in actors:
            actor.join(timeout=0.05)
    _settle_metrics(stats, totals, client, queue_key, metrics_path, time.monotonic() - started)
    _raise_if_failed(learner, actors, "stopping_producers")


def _raise_if_failed(learner, actors, state):
    """A clean learner exit is valid only after producers have been declared done."""
    if learner.exitcode is not None:
        if learner.exitcode != 0:
            raise RuntimeError(f"{learner.name} exited unexpectedly with code {learner.exitcode}")
        if state not in {"draining_learner", "completed", "aborting"}:
            raise RuntimeError(f"{learner.name} exited unexpectedly with code 0 during {state}")
    for actor in actors:
        if actor.exitcode is None:
            continue
        if actor.exitcode != 0:
            raise RuntimeError(f"{actor.name} exited unexpectedly with code {actor.exitcode}")
        if state == "running" and not state == "finite_running":
            raise RuntimeError(f"{actor.name} exited unexpectedly with code 0 during {state}")


def _evaluate_without_gpu(context, args, stats, totals, client, queue_key, metrics_path, started, learner=None, actors=()):
    """Read the result while evaluator runs, while retaining training health checks."""
    result_queue = context.Queue(1)
    evaluator = context.Process(target=_cpu_evaluate, args=(result_queue, args.baseline_model, args.model_name, args.eval_deals, args.seed, args.use_payout_head), name="cpu-evaluator")
    evaluator.start()
    result, deadline = None, time.monotonic() + args.eval_timeout
    try:
        while True:
            elapsed = time.monotonic() - started
            _drain_metrics(stats, totals, client, queue_key, metrics_path, elapsed)
            if learner is not None:
                _raise_if_failed(learner, actors, "running")
            if result is None:
                try:
                    result = result_queue.get(timeout=0.05)
                except queue.Empty:
                    pass
            evaluator.join(timeout=0.01)
            if not evaluator.is_alive():
                if evaluator.exitcode != 0:
                    raise RuntimeError(f"cpu-evaluator exited unexpectedly with code {evaluator.exitcode}")
                if result is None:
                    try:
                        result = result_queue.get(timeout=1)
                    except queue.Empty as error:
                        raise RuntimeError("cpu-evaluator exited without a result") from error
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError("cpu-evaluator timed out")
    except BaseException:
        if evaluator.is_alive():
            evaluator.terminate()
        evaluator.join(timeout=2)
        raise
    finally:
        result_queue.close()
        result_queue.join_thread()


def _complete_normally(learner, actors, producer_stop_event, producers_done, stats, totals, client, queue_key, metrics_path, started, shutdown_timeout):
    """Normal state transition: stop producers, drain learner, then verify Redis."""
    producer_stop_event.set()
    _wait_for_producers(learner, actors, stats, totals, client, queue_key, metrics_path, started, shutdown_timeout)
    # No actor can push after this event. The learner drains even a
    # below-threshold tail, saves dirty weights, and then exits.
    producers_done.set()
    _wait_with_metrics([learner], stats, totals, client, queue_key, metrics_path, started, "draining_learner", timeout=shutdown_timeout)
    _raise_if_failed(learner, actors, "completed")
    if client.llen(queue_key) != 0:
        raise RuntimeError("normal completion left untrained Redis payloads")


def run_experiment(args):
    validate_args(args)
    client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    client.ping()
    run_dir = ROOT / "experiments" / args.run_name
    metrics_path, session_seed = _prepare_run_directory(args, run_dir)
    queue_key = f"training:{args.run_name}"
    if not args.resume:
        client.delete(queue_key)
    version = initialize_model_family(args.model_name, args.source_model, force_reset=args.reset)
    _write_metric(metrics_path, {"event": "session_started", "initial_version": version, "session_seed": session_seed})
    context = mp.get_context("spawn")
    abort_event, producer_stop_event, producers_done, stats = context.Event(), context.Event(), context.Event(), context.Queue()
    learner = context.Process(target=train, kwargs={"stop_event": abort_event, "stats_queue": stats, "queue_key": queue_key, "min_queue_items": args.min_queue_items, "max_payloads": args.max_payloads, "reconstruction_workers": args.reconstruction_workers, "redis_host": args.redis_host, "redis_port": args.redis_port, "producers_done_event": producers_done}, name="learner")
    actors = [context.Process(target=self_play, args=(worker, args.model_name), kwargs={"stop_event": abort_event, "producer_stop_event": producer_stop_event, "stats_queue": stats, "queue_key": queue_key, "max_queue_items": args.max_queue_items, "game_batch_size": args.game_batch_size, "explore_rate": args.explore_rate, "max_batches": args.max_batches or None, "cpu_only": True, "use_payout_head": args.use_payout_head, "seed": _stream_seed(session_seed, "actor", worker), "redis_host": args.redis_host, "redis_port": args.redis_port}, name=f"actor-{worker}") for worker in range(args.workers)]
    processes = [learner, *actors]
    for process in processes:
        process.start()
    started, next_eval = time.monotonic(), time.monotonic() + args.eval_every
    totals = {"games": 0, "candidate_rows": 0, "generation_seconds": 0.0, "replay_seconds": 0.0, "fit_seconds": 0.0, "checkpoint_seconds": 0.0, "examples": 0, "durable_checkpoints": 0}
    normal_completion = False
    try:
        state = "running"
        while True:
            elapsed = time.monotonic() - started
            _drain_metrics(stats, totals, client, queue_key, metrics_path, elapsed)
            _raise_if_failed(learner, actors, "finite_running" if args.max_batches else state)
            duration_complete = bool(args.duration and elapsed >= args.duration)
            finite_complete = bool(args.max_batches and all(not actor.is_alive() for actor in actors))
            if duration_complete or finite_complete:
                state = "stopping_producers"
                _complete_normally(learner, actors, producer_stop_event, producers_done, stats, totals, client, queue_key, metrics_path, started, args.shutdown_timeout)
                normal_completion = True
                break
            if args.eval_every and time.monotonic() >= next_eval:
                evaluation = _evaluate_without_gpu(context, args, stats, totals, client, queue_key, metrics_path, started, learner, actors)
                _write_metric(metrics_path, {"event": "evaluation", "elapsed_seconds": time.monotonic() - started, **evaluation})
                next_eval = time.monotonic() + args.eval_every
            time.sleep(0.02)
    except BaseException:
        abort_event.set()
        _abort(processes, stats, totals, client, queue_key, metrics_path, started)
        stats.close()
        stats.join_thread()
        raise
    if not normal_completion:
        raise RuntimeError("experiment did not complete normally")
    _settle_metrics(stats, totals, client, queue_key, metrics_path, time.monotonic() - started)
    if totals["examples"] == 0:
        raise RuntimeError("experiment completed without a learner update")
    if totals["durable_checkpoints"] == 0:
        raise RuntimeError("experiment completed without a durable learner checkpoint")
    evaluation = _evaluate_without_gpu(context, args, stats, totals, client, queue_key, metrics_path, started)
    elapsed = time.monotonic() - started
    summary = {"event": "final", "elapsed_seconds": elapsed, "games_per_second": totals["games"] / max(elapsed, 0.001), "candidates_per_second": totals["candidate_rows"] / max(totals["generation_seconds"], 0.001), "totals": totals, "queue_items_remaining": client.llen(queue_key), "evaluation": evaluation}
    _write_metric(metrics_path, summary)
    stats.close()
    stats.join_thread()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--model-name", choices=("transformer_v2", "transformer_payout_v1", "smoke_transformer_v2", "smoke_transformer_payout_v1"), default="transformer_v2")
    parser.add_argument("--source-model", default="transformer")
    parser.add_argument("--baseline-model", default="transformer")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 2))
    parser.add_argument("--game-batch-size", type=int, default=50)
    parser.add_argument("--max-queue-items", type=int, default=64)
    parser.add_argument("--min-queue-items", type=int, default=4)
    parser.add_argument("--max-payloads", type=int, default=16)
    parser.add_argument("--reconstruction-workers", type=int, default=1)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--max-batches", type=int, default=0, help="per actor; useful for smoke runs")
    parser.add_argument("--eval-every", type=float, default=0)
    parser.add_argument("--eval-deals", type=int, default=20)
    parser.add_argument("--eval-timeout", type=float, default=900)
    parser.add_argument("--shutdown-timeout", type=float, default=600)
    parser.add_argument("--explore-rate", type=float, default=0.2)
    parser.add_argument("--use-payout-head", action="store_true", help="separate challenger-policy experiment; default remains production-compatible EV")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true", help="archives, never deletes, an existing experimental family and run")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
