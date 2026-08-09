from __future__ import annotations

import unittest
import tempfile
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from model_registry import load_models
from payout_challenger import create_payout_challenger
from portability_smoke import BASE_INPUT_SHAPES, PROBE_BATCH_SIZE, assert_base_input_contract, fixed_inputs


class PayoutChallengerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = load_models("transformer", compile_model=False)[0]

    def setUp(self):
        self.challenger = create_payout_challenger(self.source)

    def test_transfer_preserves_win_prediction(self):
        inputs = fixed_inputs()
        batch_size = assert_base_input_contract(inputs)
        self.assertEqual([tensor.shape[1:] for tensor in inputs], list(BASE_INPUT_SHAPES))
        baseline = np.asarray(self.source(inputs, training=False))
        output = self.challenger(inputs + [np.zeros((batch_size, 4), dtype=np.float32)], training=False)
        self.assertEqual(set(output), {"win_probability", "expected_payout"})
        self.assertTrue(np.array_equal(baseline, np.asarray(output["win_probability"])))

    def test_smoke_training_step_and_reload(self):
        inputs = fixed_inputs()
        batch_size = assert_base_input_contract(inputs)
        inputs.append(np.zeros((batch_size, 4), dtype=np.float32))
        self.challenger.train_on_batch(inputs, {"win_probability": np.full((batch_size,), 0.5, dtype=np.float32), "expected_payout": np.zeros((batch_size,), dtype=np.float32)})
        self.assertEqual(self.challenger.output_names, ["expected_payout", "win_probability"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "challenger.keras"
            self.challenger.save(checkpoint)
            reloaded = tf.keras.models.load_model(checkpoint)
            self.assertEqual(set(reloaded.output_names), {"expected_payout", "win_probability"})

    def test_probe_contract_is_batch_derived_and_preserves_random_state(self):
        random.seed(123)
        before = random.getstate()
        inputs = fixed_inputs()
        self.assertEqual(assert_base_input_contract(inputs), PROBE_BATCH_SIZE)
        self.assertEqual(random.getstate(), before)
        subset = fixed_inputs(2)
        self.assertEqual(assert_base_input_contract(subset), 2)
        self.assertTrue(all(tensor.shape[0] == 2 for tensor in subset))


if __name__ == "__main__":
    unittest.main()
