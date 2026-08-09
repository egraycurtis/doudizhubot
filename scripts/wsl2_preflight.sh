#!/usr/bin/env bash
set -euo pipefail

python3 --version
python3 - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
redis-cli ping
