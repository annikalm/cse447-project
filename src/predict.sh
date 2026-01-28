#!/usr/bin/env bash
set -euo pipefail
PYTHONUNBUFFERED=1 python src/myprogram.py test --work_dir work --test_data "$1" --test_output "$2"
