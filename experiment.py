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
import signal
import threading
import uuid
import time

import redis

from compete import evaluate_families
from model_registry import archive_destination, get_model_config, get_model_dir, get_metadata_path, initialize_model_family, validate_experiment_family
from self_play import self_play
from train import train


ROOT = Path(__file__).resolve().parent
RUN_IDENTITY_FIELDS = ("model_name", "source_model", "baseline_model", "seed", "model_schema_version")
LOCK_LEASE_SECONDS = 90


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RedisRunLock:
    """Token-owned Redis lease that prevents two learners sharing one family."""

    _REFRESH_SCRIPT = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
      return redis.call('pexpire', KEYS[1], ARGV[2])
    end
    return 0
    """
    _RELEASE_SCRIPT = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
      return redis.call('del', KEYS[1])
    end
    return 0
    """

    def __init__(self, client, run_name: str, model_name: str, lease_seconds: int = LOCK_LEASE_SECONDS):
        self.client, self.key = client, f"training-lock:{run_name}:{model_name}"
        self.token, self.lease_seconds = uuid.uuid4().hex, lease_seconds
        self._stop, self._thread, self.lost = threading.Event(), None, False

    def acquire(self):
        if not self.client.set(self.key, self.token, nx=True, ex=self.lease_seconds):
            raise RuntimeError(f"another coordinator owns {self.key}; wait for its lease or use explicit recovery after it exits")
        self._thread = threading.Thread(target=self._heartbeat, name="training-lock-heartbeat", daemon=True)
        self._thread.start()

    def _heartbeat(self):
        while not self._stop.wait(max(1, self.lease_seconds // 3)):
            try:
                if not self.refresh():
                    self.lost = True
                    return
            except BaseException:
                self.lost = True
                return

    def refresh(self) -> bool:
        return bool(self.client.eval(self._REFRESH_SCRIPT, 1, self.key, self.token, str(self.lease_seconds * 1000)))

    def assert_held(self):
        if self.lost or not self.refresh():
            raise RuntimeError("training ownership lock was lost; refusing concurrent model writes")

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        try:
            self.client.eval(self._RELEASE_SCRIPT, 1, self.key, self.token)
        except BaseException:
            # Never mask the original training failure during best-effort unlock.
            pass


def _load_sessions(sessions_path: Path) -> dict:
    return json.loads(sessions_path.read_text(encoding="utf-8"))


def _update_session(sessions_path: Path, epoch: int, **changes) -> dict:
    """Durably update one session while the coordinator ownership lock is held."""
    sessions = _load_sessions(sessions_path)
    for session in sessions.get("sessions", []):
        if session.get("epoch") == epoch:
            session.update(changes)
            _atomic_write_json(sessions_path, sessions)
            return session
    raise RuntimeError(f"session epoch {epoch} is missing")


def _close_queue(queue_object) -> None:
    if queue_object is None:
        return
    try:
        queue_object.close()
    finally:
        try:
            queue_object.join_thread()
        except (AttributeError, RuntimeError):
            pass


def _start_processes(requested_processes):
    """Start children transactionally; a later start failure cannot orphan peers."""
    started = []
    try:
        for process in requested_processes:
            process.start()
            started.append(process)
    except BaseException:
        for process in started:
            if process.is_alive():
                process.terminate()
        for process in started:
            process.join(timeout=2)
        raise
    return started


def _phase(args) -> str:
    return "payout_policy" if getattr(args, "use_payout_head", False) else "warmup"


def _reset_artifacts(run_dir: Path, model_dir: Path) -> dict[Path, Path]:
    """Archive both sides of reset, restoring the first if the second move fails."""
    moves = [(path, archive_destination(path)) for path in (run_dir, model_dir) if path.exists()]
    moved = {}
    try:
        for source, destination in moves:
            source.rename(destination)
            moved[source] = destination
    except BaseException:
        for source, destination in reversed(list(moved.items())):
            if destination.exists() and not source.exists():
                destination.rename(source)
        raise
    return moved


def _rollback_reset(moved: dict[Path, Path], fresh_run: Path, fresh_model: Path) -> None:
    """Keep partial new output as an archive and restore the old transaction."""
    for fresh in (fresh_run, fresh_model):
        if fresh.exists():
            fresh.rename(archive_destination(fresh.with_name(f"{fresh.name}.failed-init")))
    for source, destination in reversed(list(moved.items())):
        if destination.exists() and not source.exists():
            destination.rename(source)


def install_evaluator_signal_policy():
    """Evaluator children also leave Ctrl+C ownership with the coordinator."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _cpu_evaluate(result_queue, baseline_model, challenger_model, deals, seed, challenger_uses_payout):
    """Spawned evaluator: reserve GPU exclusively for the learner process."""
    import tensorflow as tf

    install_evaluator_signal_policy()
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
    if args.resume:
        if not config_path.exists() or not sessions_path.exists():
            raise ValueError(f"--resume requires an existing initialized run: {run_dir}")
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        expected = _run_identity(args)
        actual = {key: stored.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"resume configuration is incompatible: expected {expected}, found {actual}")
        sessions = _load_sessions(sessions_path)
        if not sessions.get("sessions"):
            raise ValueError("--resume requires at least one prior session record")
        prior = sessions["sessions"][-1]
        prior_state = prior.get("state", "incomplete")
        prior_phase = prior.get("phase", "warmup")
        phase = _phase(args)
        if prior_phase == "payout_policy" and phase != "payout_policy":
            raise ValueError("cannot resume a payout-policy session in warmup mode")
        if prior_state != "completed" and not getattr(args, "recover", False):
            raise ValueError(f"last session is {prior_state}; pass --recover to acknowledge and resume recoverable state")
        if phase == "payout_policy" and not args.model_name.endswith("payout_v1"):
            raise ValueError("payout-policy phase requires a payout challenger family")
        if phase == "payout_policy" and (prior_phase != "warmup" or prior_state != "completed" or prior.get("examples", 0) <= 0 or prior.get("durable_checkpoints", 0) <= 0 or prior.get("final_version") is None):
            raise ValueError("payout-policy phase requires a completed warmup with learner updates and a durable checkpoint")
        epoch = len(sessions["sessions"])
    else:
        if _phase(args) == "payout_policy":
            raise ValueError("payout-policy phase must resume a completed payout warmup run")
        if config_path.exists() or get_model_dir(args.model_name).exists():
            raise ValueError("run or experimental model family already exists; choose a unique --run-name, use --resume, or use --reset")
        run_dir.mkdir(parents=True, exist_ok=True)
        epoch, sessions = 0, {"sessions": []}
        config = vars(args).copy()
        config.update(_run_identity(args))
        _atomic_write_json(config_path, config)
    session = {
        "epoch": epoch,
        "seed": _stream_seed(args.seed, args.run_name, args.model_name, epoch),
        "phase": _phase(args),
        "policy_mode": "payout_head" if _phase(args) == "payout_policy" else "ev_fallback",
        "state": "initializing",
        "created_at": _utcnow(),
    }
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
    if getattr(args, "recover", False) and not getattr(args, "resume", False):
        raise ValueError("--recover requires --resume")


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
        elif metric["kind"] == "producer_complete":
            completions = totals.setdefault("actor_completions", {})
            actor = str(metric.get("actor"))
            if actor in completions:
                raise RuntimeError(f"duplicate completion report from actor {actor}")
            completions[actor] = metric
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
    if stats is not None:
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


def _validate_actor_completions(totals, actors, game_batch_size: int, expected_batches: int | None = None) -> None:
    """Require one coherent completion report from every actor before draining."""
    completed = totals.get("actor_completions", {})
    expected_actors = {str(index) for index in range(len(actors))}
    if set(completed) != expected_actors:
        raise RuntimeError("missing, duplicate, or malformed actor completion reports")
    for actor, report in completed.items():
        if not isinstance(report, dict) or report.get("actor") != int(actor):
            raise RuntimeError("malformed actor completion report")
        batches, games, reason = report.get("batches"), report.get("games"), report.get("reason")
        if not isinstance(batches, int) or batches < 0 or games != batches * game_batch_size:
            raise RuntimeError("actor completion counts are inconsistent")
        if reason not in {"max_batches", "producer_stop", "abort"}:
            raise RuntimeError("actor completion has an invalid reason")
        if expected_batches is not None and (reason != "max_batches" or batches != expected_batches):
            raise RuntimeError("finite actors did not report their configured batch completion")


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


def _evaluate_without_gpu(context, args, stats, totals, client, queue_key, metrics_path, started, learner=None, actors=(), lifecycle_state="running"):
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
                _raise_if_failed(learner, actors, lifecycle_state)
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


def _complete_normally(learner, actors, producer_stop_event, producers_done, stats, totals, client, queue_key, metrics_path, started, shutdown_timeout, expected_batches=None, game_batch_size=1):
    """Normal state transition: stop producers, drain learner, then verify Redis."""
    producer_stop_event.set()
    _wait_for_producers(learner, actors, stats, totals, client, queue_key, metrics_path, started, shutdown_timeout)
    # No actor can push after this event. The learner drains even a
    # below-threshold tail, saves dirty weights, and then exits.
    producers_done.set()
    _wait_with_metrics([learner], stats, totals, client, queue_key, metrics_path, started, "draining_learner", timeout=shutdown_timeout)
    _raise_if_failed(learner, actors, "completed")
    _validate_actor_completions(totals, actors, game_batch_size, expected_batches)
    if client.llen(queue_key) != 0:
        raise RuntimeError("normal completion left untrained Redis payloads")


def run_experiment(args):
    validate_args(args)
    if args.reset and args.resume:
        raise ValueError("--reset and --resume cannot be used together")
    client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    client.ping()
    run_dir = ROOT / "experiments" / args.run_name
    model_dir = get_model_dir(args.model_name)
    lock = RedisRunLock(client, args.run_name, args.model_name)
    lock.acquire()
    reset_moves = {}
    metrics_path = run_dir / "metrics.jsonl"
    stats = None
    processes = []
    abort_event = None
    producer_stop_event = producers_done = learner = None
    actors = []
    sessions_path = None
    session_epoch = None
    started = time.monotonic()
    queue_key = f"training:{args.run_name}"
    totals = {"games": 0, "candidate_rows": 0, "generation_seconds": 0.0, "replay_seconds": 0.0, "fit_seconds": 0.0, "checkpoint_seconds": 0.0, "examples": 0, "durable_checkpoints": 0, "actor_completions": {}}
    final_status, interrupted, final_metadata = "failed", False, None
    try:
        if args.reset:
            # Validate before mutating either artifact, then archive both as one transaction.
            get_model_config(args.model_name)
            reset_moves = _reset_artifacts(run_dir, model_dir)
        try:
            metrics_path, session_seed = _prepare_run_directory(args, run_dir)
            sessions_path = run_dir / "sessions.json"
            session_epoch = _load_sessions(sessions_path)["sessions"][-1]["epoch"]
            if args.resume:
                metadata = validate_experiment_family(args.model_name, args.source_model)
                version = metadata["version"]
            else:
                version = initialize_model_family(args.model_name, args.source_model, force_reset=False)
                metadata = validate_experiment_family(args.model_name, args.source_model)
        except BaseException:
            if reset_moves:
                _rollback_reset(reset_moves, run_dir, model_dir)
            raise
        if not args.resume:
            client.delete(queue_key)
        _update_session(sessions_path, session_epoch, initial_version=version, initial_checkpoint_sha256=metadata["sha256"], source_model=metadata["source_model"], source_version=metadata.get("source_version"), source_sha256=metadata.get("source_sha256"))
        _write_metric(metrics_path, {"event": "session_started", "initial_version": version, "session_seed": session_seed, "phase": _phase(args), "policy_mode": "payout_head" if args.use_payout_head else "ev_fallback", "source_model": metadata["source_model"], "checkpoint_sha256": metadata["sha256"]})
        context = mp.get_context("spawn")
        abort_event, producer_stop_event, producers_done, stats = context.Event(), context.Event(), context.Event(), context.Queue()
        learner = context.Process(target=train, kwargs={"stop_event": abort_event, "stats_queue": stats, "queue_key": queue_key, "min_queue_items": args.min_queue_items, "max_payloads": args.max_payloads, "reconstruction_workers": args.reconstruction_workers, "redis_host": args.redis_host, "redis_port": args.redis_port, "producers_done_event": producers_done, "ignore_sigint": True}, name="learner")
        actors = [context.Process(target=self_play, args=(worker, args.model_name), kwargs={"stop_event": abort_event, "producer_stop_event": producer_stop_event, "stats_queue": stats, "queue_key": queue_key, "max_queue_items": args.max_queue_items, "game_batch_size": args.game_batch_size, "explore_rate": args.explore_rate, "max_batches": args.max_batches or None, "cpu_only": True, "use_payout_head": args.use_payout_head, "seed": _stream_seed(session_seed, "actor", worker), "redis_host": args.redis_host, "redis_port": args.redis_port, "ignore_sigint": True}, name=f"actor-{worker}") for worker in range(args.workers)]
        requested_processes = [learner, *actors]
        try:
            processes = _start_processes(requested_processes)
        except BaseException:
            abort_event.set()
            raise
        _update_session(sessions_path, session_epoch, state="running", started_at=_utcnow())
        started, next_eval = time.monotonic(), time.monotonic() + args.eval_every
        state = "running"
        while True:
            lock.assert_held()
            elapsed = time.monotonic() - started
            _drain_metrics(stats, totals, client, queue_key, metrics_path, elapsed)
            _raise_if_failed(learner, actors, "finite_running" if args.max_batches else state)
            duration_complete = bool(args.duration and elapsed >= args.duration)
            finite_complete = bool(args.max_batches and all(not actor.is_alive() for actor in actors))
            if duration_complete or finite_complete:
                state = "stopping_producers"
                _complete_normally(learner, actors, producer_stop_event, producers_done, stats, totals, client, queue_key, metrics_path, started, args.shutdown_timeout, args.max_batches if finite_complete else None, args.game_batch_size)
                break
            if args.eval_every and time.monotonic() >= next_eval:
                evaluation = _evaluate_without_gpu(context, args, stats, totals, client, queue_key, metrics_path, started, learner, actors, "finite_running" if args.max_batches else "running")
                _write_metric(metrics_path, {"event": "evaluation", "elapsed_seconds": time.monotonic() - started, **evaluation})
                next_eval = time.monotonic() + args.eval_every
            time.sleep(0.02)
        _settle_metrics(stats, totals, client, queue_key, metrics_path, time.monotonic() - started)
        if totals["examples"] == 0:
            raise RuntimeError("experiment completed without a learner update")
        if totals["durable_checkpoints"] == 0:
            raise RuntimeError("experiment completed without a durable learner checkpoint")
        lock.assert_held()
        final_metadata = validate_experiment_family(args.model_name, args.source_model)
        evaluation = _evaluate_without_gpu(context, args, stats, totals, client, queue_key, metrics_path, started)
        lock.assert_held()
        elapsed = time.monotonic() - started
        summary = {"event": "final", "elapsed_seconds": elapsed, "games_per_second": totals["games"] / max(elapsed, 0.001), "candidates_per_second": totals["candidate_rows"] / max(totals["generation_seconds"], 0.001), "totals": totals, "queue_items_remaining": client.llen(queue_key), "evaluation": evaluation, "session_completed": True, "session_interrupted": False}
        _write_metric(metrics_path, summary)
        final_status = "completed"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    except KeyboardInterrupt:
        # First Ctrl+C behaves like a finite run: actors finish their current
        # batches, the learner drains Redis and saves one final checkpoint.
        if learner is None or producer_stop_event is None or producers_done is None or stats is None:
            # No worker lifecycle exists yet.  The finally block still records
            # an interrupted initializer without inventing a drain.
            interrupted, final_status = True, "interrupted"
            raise
        try:
            interrupted = True
            _complete_normally(learner, actors, producer_stop_event, producers_done, stats, totals, client, queue_key, metrics_path, started, args.shutdown_timeout, game_batch_size=args.game_batch_size)
            _write_metric(metrics_path, {"event": "session_interrupted", "redis_drained": client.llen(queue_key) == 0})
            final_metadata = validate_experiment_family(args.model_name, args.source_model)
            lock.assert_held()
            final_status = "interrupted"
            return {"event": "interrupted", "totals": totals, "queue_items_remaining": client.llen(queue_key), "session_completed": False, "session_interrupted": True}
        except KeyboardInterrupt:
            if abort_event is not None:
                abort_event.set()
            _abort(processes, stats, totals, client, queue_key, metrics_path, started)
            interrupted, final_status = True, "interrupted"
            raise
        except BaseException:
            if abort_event is not None:
                abort_event.set()
            _abort(processes, stats, totals, client, queue_key, metrics_path, started)
            raise
    except BaseException:
        if abort_event is not None:
            abort_event.set()
        _abort(processes, stats, totals, client, queue_key, metrics_path, started)
        raise
    finally:
        if sessions_path is not None and session_epoch is not None:
            try:
                lock.assert_held()
                final_hashes = final_metadata.get("sha256") if final_metadata else None
                final_version = final_metadata.get("version") if final_metadata else None
                _update_session(sessions_path, session_epoch, state=final_status, ended_at=_utcnow(), final_version=final_version, final_checkpoint_sha256=final_hashes, examples=totals["examples"], durable_checkpoints=totals["durable_checkpoints"], actor_completions=totals.get("actor_completions", {}), redis_remaining=client.llen(queue_key), redis_drained=client.llen(queue_key) == 0, interrupted=interrupted)
            except BaseException:
                # The original lifecycle error is more actionable; leave the
                # initializing/running record for explicit --recover instead.
                pass
        _close_queue(stats)
        lock.release()


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
    parser.add_argument("--recover", action="store_true", help="explicitly resume after an interrupted, failed, or incomplete session")
    parser.add_argument("--reset", action="store_true", help="archives, never deletes, an existing experimental family and run")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
