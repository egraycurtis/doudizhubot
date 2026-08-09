from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest.mock import patch

import experiment


class FakeClient:
    def __init__(self, items=0):
        self.items = items

    def llen(self, _key):
        return self.items


class FakeProcess:
    def __init__(self, name, alive=True, exitcode=None, stop_on_join=True):
        self.name = name
        self._alive = alive
        self.exitcode = exitcode
        self.stop_on_join = stop_on_join
        self.started = self.terminated = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        if self.stop_on_join:
            self._alive = False
            if self.exitcode is None:
                self.exitcode = 0

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.exitcode = -15


class OrderedEvent:
    def __init__(self, calls, name):
        self.calls, self.name, self.set_value = calls, name, False

    def set(self):
        self.calls.append(self.name)
        self.set_value = True

    def is_set(self):
        return self.set_value


class FakeResultQueue:
    def __init__(self, result=None, on_get=None):
        self.result, self.on_get = result, on_get
        self.closed = self.joined = False

    def get(self, timeout=None):
        if self.result is None:
            raise queue.Empty
        result, self.result = self.result, None
        if self.on_get:
            self.on_get()
        return result

    def close(self):
        self.closed = True

    def join_thread(self):
        self.joined = True


class FakeContext:
    def __init__(self, evaluator, result_queue):
        self.evaluator, self.result_queue = evaluator, result_queue

    def Queue(self, _size):
        return self.result_queue

    def Process(self, **_kwargs):
        return self.evaluator


def totals():
    return {"games": 0, "candidate_rows": 0, "generation_seconds": 0.0, "replay_seconds": 0.0, "fit_seconds": 0.0, "checkpoint_seconds": 0.0, "examples": 0, "durable_checkpoints": 0}


class ExperimentLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.metrics = Path(self.directory.name) / "metrics.jsonl"
        self.client = FakeClient()

    def tearDown(self):
        self.directory.cleanup()

    def test_learner_failure_in_final_drain_is_not_missed(self):
        learner = FakeProcess("learner", alive=False, exitcode=1)
        with self.assertRaisesRegex(RuntimeError, "code 1"):
            experiment._wait_with_metrics([learner], queue.Queue(), totals(), self.client, "q", self.metrics, time.monotonic(), "draining_learner", timeout=1)

    def test_unexpected_clean_learner_exit_fails_while_running(self):
        with self.assertRaisesRegex(RuntimeError, "code 0"):
            experiment._raise_if_failed(FakeProcess("learner", alive=False, exitcode=0), [], "running")

    def test_actor_failure_fails_immediately(self):
        with self.assertRaisesRegex(RuntimeError, "actor-0"):
            experiment._raise_if_failed(FakeProcess("learner"), [FakeProcess("actor-0", alive=False, exitcode=2)], "running")

    def test_pending_final_learner_metric_is_accounted(self):
        stats = queue.Queue()
        stats.put({"kind": "learner", "examples": 5, "replay_seconds": 0, "fit_seconds": 0, "checkpoint_seconds": 1, "durable_checkpoint": True})
        collected = totals()
        experiment._settle_metrics(stats, collected, self.client, "q", self.metrics, 0, quiet_seconds=0.01)
        self.assertEqual(collected["examples"], 5)
        self.assertEqual(collected["durable_checkpoints"], 1)

    def test_normal_shutdown_orders_producer_stop_before_learner_drain(self):
        calls, stats = [], queue.Queue()
        learner = FakeProcess("learner", alive=True, stop_on_join=True)
        actor = FakeProcess("actor-0", alive=False, exitcode=0)
        with patch("experiment._wait_for_producers", side_effect=lambda *args: calls.append("actors_joined")), patch("experiment._wait_with_metrics", side_effect=lambda *args, **kwargs: calls.append("learner_drained")):
            experiment._complete_normally(learner, [actor], OrderedEvent(calls, "producer_stop"), OrderedEvent(calls, "producers_done"), stats, totals(), self.client, "q", self.metrics, time.monotonic(), 1)
        self.assertEqual(calls, ["producer_stop", "actors_joined", "producers_done", "learner_drained"])

    def test_below_threshold_tail_is_drained_by_learner_protocol(self):
        class RedisTail:
            def __init__(self): self.payloads = [b"tail"]
            def llen(self, _): return len(self.payloads)
            def pipeline(self): return self
            def lrange(self, *_): return self
            def ltrim(self, *_): return self
            def execute(self):
                result, self.payloads = self.payloads, []
                return result, True
        from train import _pop_training_payloads
        self.assertEqual(_pop_training_payloads(RedisTail(), "q", min_items=4, drain=True), [b"tail"])

    def test_resume_sessions_advance_without_changing_fresh_reproducibility(self):
        args = Namespace(run_name="repro", model_name="transformer_v2", source_model="transformer", baseline_model="transformer", use_payout_head=False, seed=9, reset=False, resume=False)
        with patch("experiment.get_model_dir", return_value=Path(self.directory.name) / "models" / "transformer_v2"):
            _, first = experiment._prepare_run_directory(args, Path(self.directory.name) / "run")
            args.resume = True
            _, second = experiment._prepare_run_directory(args, Path(self.directory.name) / "run")
        self.assertNotEqual(first, second)
        self.assertEqual(first, experiment._stream_seed(9, "repro", "transformer_v2", 0))
        self.assertEqual(second, experiment._stream_seed(9, "repro", "transformer_v2", 1))
        self.assertEqual(experiment._stream_seed(first, "actor", 0), experiment._stream_seed(first, "actor", 0))
        self.assertNotEqual(experiment._stream_seed(first, "actor", 0), experiment._stream_seed(first, "actor", 1))

    def test_resume_rejects_incompatible_family_and_argument_guards(self):
        args = Namespace(run_name="guard", model_name="transformer_v2", source_model="transformer", baseline_model="transformer", use_payout_head=False, seed=9, reset=False, resume=False)
        with patch("experiment.get_model_dir", return_value=Path(self.directory.name) / "models" / "transformer_v2"):
            experiment._prepare_run_directory(args, Path(self.directory.name) / "guard")
            args.resume, args.baseline_model = True, "other"
            with self.assertRaisesRegex(ValueError, "incompatible"):
                experiment._prepare_run_directory(args, Path(self.directory.name) / "guard")
        validation_args = Namespace(workers=0, game_batch_size=1, max_queue_items=1, min_queue_items=1, max_payloads=1, reconstruction_workers=1, eval_deals=1, duration=1, max_batches=0, eval_every=0, eval_timeout=1, shutdown_timeout=1, explore_rate=0.2)
        with self.assertRaisesRegex(ValueError, "workers"):
            experiment.validate_args(validation_args)

    def test_evaluation_reads_large_result_before_child_join(self):
        evaluator = FakeProcess("cpu-evaluator", alive=True, stop_on_join=False)
        result = {"results": ["x" * 100_000]}
        def finish_evaluator():
            evaluator._alive = False
            evaluator.exitcode = 0
        result_queue = FakeResultQueue(result, on_get=finish_evaluator)
        args = Namespace(baseline_model="transformer", model_name="transformer_v2", eval_deals=1, seed=1, use_payout_head=False, eval_timeout=1)
        actual = experiment._evaluate_without_gpu(FakeContext(evaluator, result_queue), args, queue.Queue(), totals(), self.client, "q", self.metrics, time.monotonic(), FakeProcess("learner"), [])
        self.assertEqual(actual, result)
        self.assertTrue(result_queue.closed)

    def test_training_failure_stops_evaluator(self):
        evaluator = FakeProcess("cpu-evaluator", alive=True, stop_on_join=False)
        args = Namespace(baseline_model="transformer", model_name="transformer_v2", eval_deals=1, seed=1, use_payout_head=False, eval_timeout=1)
        with self.assertRaisesRegex(RuntimeError, "learner"):
            experiment._evaluate_without_gpu(FakeContext(evaluator, FakeResultQueue()), args, queue.Queue(), totals(), self.client, "q", self.metrics, time.monotonic(), FakeProcess("learner", alive=False, exitcode=1), [])
        self.assertTrue(evaluator.terminated)


if __name__ == "__main__":
    unittest.main()
