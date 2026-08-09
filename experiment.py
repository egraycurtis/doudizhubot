"""Finite, resumable, metrics-first experiment runner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time

import redis

from compete import evaluate_families
from model_registry import initialize_model_family
from self_play import self_play
from train import train


ROOT = Path(__file__).resolve().parent


def _write_metric(path: Path, metric: dict) -> None:
    metric["wall_time_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metric, sort_keys=True) + "\n")


def _stop(processes):
    for process in processes:
        process.join(timeout=10)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2)


def run_experiment(args):
    if args.duration <= 0 and args.max_batches <= 0:
        raise ValueError("set --duration or --max-batches to a positive value")
    client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    client.ping()
    run_dir = ROOT / "experiments" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    queue_key = f"training:{args.run_name}"
    if not args.resume:
        client.delete(queue_key)
    version = initialize_model_family(args.model_name, args.source_model, force_reset=args.reset)
    config = vars(args).copy()
    config["initial_version"] = version
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    context = mp.get_context("spawn")
    stop_event, stats = context.Event(), context.Queue()
    processes = [context.Process(target=train, kwargs={"stop_event": stop_event, "stats_queue": stats, "queue_key": queue_key, "min_queue_items": args.min_queue_items, "max_payloads": args.max_payloads, "reconstruction_workers": args.reconstruction_workers}, name="learner")]
    for worker in range(args.workers):
        processes.append(context.Process(target=self_play, args=(worker, args.model_name), kwargs={"stop_event": stop_event, "stats_queue": stats, "queue_key": queue_key, "max_queue_items": args.max_queue_items, "game_batch_size": args.game_batch_size, "explore_rate": args.explore_rate, "max_batches": args.max_batches or None, "cpu_only": True, "use_payout_head": args.use_payout_head}, name=f"actor-{worker}"))
    for process in processes:
        process.start()
    started, next_eval = time.monotonic(), time.monotonic() + args.eval_every
    totals = {"games": 0, "candidate_rows": 0, "generation_seconds": 0.0, "replay_seconds": 0.0, "fit_seconds": 0.0, "checkpoint_seconds": 0.0, "examples": 0}
    try:
        while True:
            elapsed = time.monotonic() - started
            if args.duration and elapsed >= args.duration:
                break
            if args.max_batches and all(not process.is_alive() for process in processes[1:]):
                break
            try:
                metric = stats.get(timeout=0.5)
                if metric["kind"] == "generation":
                    totals["games"] += metric["games"]
                    totals["candidate_rows"] += metric["candidate_rows"]
                    totals["generation_seconds"] += metric["batch_seconds"]
                elif metric["kind"] == "learner":
                    totals["examples"] += metric["examples"]
                    totals["replay_seconds"] += metric["replay_seconds"]
                    totals["fit_seconds"] += metric["fit_seconds"]
                    totals["checkpoint_seconds"] += metric["checkpoint_seconds"]
                _write_metric(metrics_path, {"event": metric["kind"], "elapsed_seconds": elapsed, "queue_items": client.llen(queue_key), **metric})
            except queue.Empty:
                pass
            if args.eval_every and time.monotonic() >= next_eval:
                evaluation = evaluate_families(args.baseline_model, args.model_name, deals=args.eval_deals, seed=args.seed, challenger_uses_payout=args.use_payout_head)
                _write_metric(metrics_path, {"event": "evaluation", "elapsed_seconds": elapsed, **evaluation})
                next_eval += args.eval_every
    finally:
        stop_event.set()
        _stop(processes)
    evaluation = evaluate_families(args.baseline_model, args.model_name, deals=args.eval_deals, seed=args.seed, challenger_uses_payout=args.use_payout_head)
    elapsed = time.monotonic() - started
    summary = {"event": "final", "elapsed_seconds": elapsed, "games_per_second": totals["games"] / max(elapsed, 0.001), "candidates_per_second": totals["candidate_rows"] / max(totals["generation_seconds"], 0.001), "totals": totals, "queue_items_remaining": client.llen(queue_key), "evaluation": evaluation}
    _write_metric(metrics_path, summary)
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
    parser.add_argument("--explore-rate", type=float, default=0.2)
    parser.add_argument("--use-payout-head", action="store_true", help="separate challenger-policy experiment; default remains production-compatible EV")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true", help="archives, never deletes, an existing experimental family")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
