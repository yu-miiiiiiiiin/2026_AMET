#!/usr/bin/env bash
# Source this file to start ver13_1 autonomous driving: wait for a green
# light (forced start after GREEN_TIMEOUT seconds if none is seen), then
# drive on the trained camera+LiDAR model with the ver13 curve/cone speed
# policy.
# Example: source /home/physicar/physicar_ws/run.sh

(
    set -euo pipefail
    cd "$(dirname "${BASH_SOURCE[0]}")"

    python3 drive_ver13_1.py \
        --green-timeout "${GREEN_TIMEOUT:-20}" \
        "$@"
)
