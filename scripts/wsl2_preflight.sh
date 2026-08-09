#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3.10}"
REQUIRE_GPU=0
if [[ "${1:-}" == "--require-gpu" ]]; then REQUIRE_GPU=1; fi
if [[ "${1:-}" != "" && "${1:-}" != "--require-gpu" ]]; then
  echo "usage: $0 [--require-gpu]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
command -v "$PYTHON" >/dev/null || { echo "Python command '$PYTHON' was not found. Create the documented Python 3.10 venv first." >&2; exit 1; }
"$PYTHON" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python 3.10 is required for the pinned training environment; found {sys.version.split()[0]}")
import tensorflow as tf
if not tf.__version__.startswith("2.15."):
    raise SystemExit(f"TensorFlow 2.15.x is required for the saved Keras 2.15 models; found {tf.__version__}")
print("Python:", sys.version.split()[0])
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPUs:", [gpu.name for gpu in gpus])
PY

if [[ "$REQUIRE_GPU" == 1 ]]; then
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable in WSL2. Update the Windows NVIDIA driver and confirm WSL GPU support." >&2; exit 1; }
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  "$PYTHON" - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise SystemExit("TensorFlow did not detect a GPU. Install requirements-training.txt in Ubuntu 22.04 and update the Windows NVIDIA driver; do not install a Linux display driver in WSL.")
print("TensorFlow GPU device(s):", [gpu.name for gpu in gpus])
PY
fi

"$PYTHON" portability_smoke.py --preflight
redis-cli ping
