#!/usr/bin/env bash
# Prove the thing a judge would actually do: clone this repository with no
# credential, fetch the published weights, run the controls, run the CLI.
# Anything that only works on the machine it was built on fails here.
set -uo pipefail
DIR=$(mktemp -d /tmp/airlock_verify_XXXXXX)
echo "# clean-clone verification $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# workdir $DIR (fresh, did not exist before this run)"
cd "$DIR"
env -u GITHUB_TOKEN -u GH_TOKEN git -c credential.helper= -c http.extraheader= \
    clone -q https://github.com/jianwang-ntu/airlock-privacy-gate.git app || exit 1
cd app
echo "# HEAD $(git rev-parse HEAD)"
env -u GITHUB_TOKEN -u GH_TOKEN bash scripts/fetch_model.sh || exit 1
echo "## tests/run_checks.py"
python3 tests/run_checks.py 2>&1 | grep -viE "loading weights|it/s\]" | tail -3
echo "## tests/check_readme_numbers.py"
python3 tests/check_readme_numbers.py 2>&1 | tail -2
echo "## airlock.cli"
python3 -m airlock.cli --text "From: Meera Subramanian <meera.s@example.co.uk>. Please transfer GBP 12,400 to IBAN DE89370400440532013000; card 4111 1111 1111 1111 must not be charged. Call +44 7700 900123." 2>&1 | grep -viE "loading weights|it/s\]"
echo "# workdir left at $DIR"
