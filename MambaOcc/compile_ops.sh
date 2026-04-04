#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/ops_dcnv3"
python setup.py install
