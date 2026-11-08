#!/usr/bin/env bash
set -euo pipefail
python -m taglab demo
python -m taglab normalization
python -m taglab infer --symbols '[-2, 0, 2]' --family chain
