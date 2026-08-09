from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from model_registry import load_models
from payout_challenger import create_payout_challenger
from portability_smoke import fixed_inputs


class PayoutChallengerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = load_models("transformer", compile_model=False)[0]
        cls.challenger = create_payout_challenger(cls.source)

    def test_transfer_preserves_win_prediction(self):
        inputs = fixed_inputs()
        baseline = np.asarray(self.source(inputs, training=False))
        output = self.challenger(inputs + [np.zeros((2, 4), dtype=np.float32)], training=False)
        self.assertEqual(set(output), {"win_probability", "expected_payout"})
        self.assertTrue(np.array_equal(baseline, np.asarray(output["win_probability"])))

    def test_smoke_training_step_and_reload(self):
        inputs = fixed_inputs()
        inputs.append(np.zeros((2, 4), dtype=np.float32))
        self.challenger.train_on_batch(inputs, {"win_probability": np.full((2,), 0.5, dtype=np.float32), "expected_payout": np.zeros((2,), dtype=np.float32)})
        self.assertEqual(self.challenger.output_names, ["expected_payout", "win_probability"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "challenger.keras"
            self.challenger.save(checkpoint)
            reloaded = tf.keras.models.load_model(checkpoint)
            self.assertEqual(set(reloaded.output_names), {"expected_payout", "win_probability"})


if __name__ == "__main__":
    unittest.main()
