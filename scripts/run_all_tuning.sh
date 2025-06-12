#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Path to your Python tuning script
SCRIPT="$PROJECT_ROOT/src/tmw/evaluation.py"

# Number of Optuna trials per dataset
N_TRIALS=200

# Root directory for logs (will get subfolders per dataset)
LOG_ROOT="$PROJECT_ROOT/logs"

# Loop over each dataset folder in data/processed
for DATASET_DIR in "$PROJECT_ROOT/data/processed"/*/; do
  # Remove trailing slash and get dataset name
  DATASET_DIR=${DATASET_DIR%/}
  DATASET_NAME=$(basename "$DATASET_DIR")
  
  echo "=========================================="
  echo "Running TMW tuning & evaluation on: $DATASET_NAME"
  echo "------------------------------------------"
  
  python "$SCRIPT" \
    --dataset_dir "$DATASET_DIR" \
    --n_trials $N_TRIALS \
    --log_dir "$LOG_ROOT"
  
  echo "→ Completed $DATASET_NAME"
  echo "=========================================="
  echo
  
done

echo "All experiments finished."
