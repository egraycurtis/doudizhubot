from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import model_registry


class ModelRegistryValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.family = Path(self.directory.name) / "transformer_v2"
        self.family.mkdir()
        self._write_valid_family()

    def tearDown(self):
        self.directory.cleanup()

    def _write_valid_family(self, version=7):
        paths, hashes = [], []
        for position in range(3):
            path = self.family / f"transformer_v2{position}_v{version:06d}.keras"
            path.write_bytes(f"role-{position}".encode())
            paths.append(path.name)
            hashes.append(model_registry._file_sha256(path))
        metadata = {
            "version": version,
            "model_name": "transformer_v2",
            "schema_version": 1,
            "source_model": "transformer",
            "checkpoints": paths,
            "sha256": hashes,
            "config": model_registry.asdict(model_registry.get_model_config("transformer_v2")),
        }
        (self.family / "metadata.json").write_text(json.dumps(metadata))

    def _validate(self):
        with patch("model_registry.get_model_dir", return_value=self.family):
            return model_registry.validate_experiment_family("transformer_v2", "transformer")

    def test_valid_family_and_pinned_version(self):
        metadata = self._validate()
        self.assertEqual(metadata["version"], 7)
        with patch("model_registry.get_model_dir", return_value=self.family):
            self.assertEqual(model_registry.get_checkpoint_path("transformer_v2", 2, 7).name, "transformer_v22_v000007.keras")

    def test_missing_directory_or_metadata_is_rejected(self):
        with patch("model_registry.get_model_dir", return_value=Path(self.directory.name) / "missing"):
            with self.assertRaisesRegex(ValueError, "missing"):
                model_registry.validate_experiment_family("transformer_v2")
        (self.family / "metadata.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self._validate()

    def test_missing_role_filename_hash_schema_and_source_mismatches_are_rejected(self):
        metadata_path = self.family / "metadata.json"
        cases = [
            ("missing role", lambda record: (self.family / record["checkpoints"][1]).unlink()),
            ("filename", lambda record: record["checkpoints"].__setitem__(0, "wrong.keras")),
            ("hash", lambda record: record["sha256"].__setitem__(2, "wrong")),
            ("schema", lambda record: record.__setitem__("schema_version", 999)),
            ("config", lambda record: record.__setitem__("config", {})),
            ("source", lambda record: record.__setitem__("source_model", "other")),
        ]
        for _name, mutate in cases:
            self._write_valid_family()
            record = json.loads(metadata_path.read_text())
            mutate(record)
            metadata_path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                self._validate()

    def test_payout_family_is_not_a_compatible_base_transfer_source(self):
        with self.assertRaisesRegex(ValueError, "stake-aware"):
            model_registry._validate_transfer_compatibility("transformer_v2", "transformer_payout_v1", 0)


if __name__ == "__main__":
    unittest.main()
