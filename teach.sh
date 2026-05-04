#!/bin/bash
# Launch the teach-pose helper (torque off, read servo positions by hand).
# Output streams to this terminal. Stop main.py first or it will contend for USB.
# Usage: ./teach.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /root/miniforge3/etc/profile.d/conda.sh
conda activate iotc-tria-xarm

exec stdbuf -oL -eL python -u teach_pose.py "$@"
