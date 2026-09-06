#!/usr/bin/env bash
# Corpus -> detector -> calibration -> evidence. ~4 minutes on one L40S.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${AIRLOCK_DATA_DIR:-.data_cache}"
mkdir -p "$DATA"
BASE=https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual/resolve/main/data
for split in test train; do
  f="$DATA/English_${split}.parquet"
  [ -f "$f" ] || curl -sSL -o "$f" "$BASE/English_${split}-00000-of-00001.parquet"
done
python3 scripts/prepare_data.py
python3 -m airlock.train --epochs 2 --batch 32 --out models/detector
python3 scripts/fit_gate.py
python3 -m airlock.evaluate --thresholds 0.1,0.2,0.3,0.5
python3 scripts/eval_gate.py --budgets 0.045,0.06,0.09,0.12,0.16,0.2,0.3,0.5,1.0
python3 scripts/demo_run.py
