#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3.10}"
REQUIRE_GPU=0
if [[ "${1:-}" == "--require-gpu" ]]; then REQUIRE_GPU=1; fi
command -v "$PYTHON" >/dev/null
"$PYTHON" --version
"$PYTHON" - <<'PY'
import tensorflow as tf
import sys
print("TensorFlow:", tf.__version__)
print("Python:", sys.version)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
for gpu in gpus:
    print("GPU name:", gpu.name)
PY
"$PYTHON" - <<'PY'
from model_registry import production_model_hashes
for name, digest in production_model_hashes().items():
    print(f"{name} SHA-256: {digest}")
PY
redis-cli ping
if [[ "$REQUIRE_GPU" == 1 ]]; then
  "$PYTHON" - <<'PY'
import tensorflow as tf
raise SystemExit(0 if tf.config.list_physical_devices("GPU") else "TensorFlow did not detect a GPU")
PY
fi
