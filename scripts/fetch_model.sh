#!/usr/bin/env bash
# Fetch the exact weights and calibration every number in evidence/ was measured
# with. No credential needed. Rebuild instead with scripts/train_all.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://github.com/jianwang-ntu/airlock-privacy-gate/releases/download/v0.1.0/airlock-detector-v0.1.0.tar.gz
EXPECTED=7a0ac1490da4c4d878f898dbe9888ab056938e62ca1c536c5922e5cfb716af1d
mkdir -p models
curl -sSL -o /tmp/airlock-detector.tar.gz "$URL"
GOT=$(sha256sum /tmp/airlock-detector.tar.gz | cut -d' ' -f1)
[ "$GOT" = "$EXPECTED" ] || { echo "sha256 mismatch: $GOT != $EXPECTED"; exit 1; }
tar xzf /tmp/airlock-detector.tar.gz -C models
echo "models/detector and models/gate_calibration.json are in place"
