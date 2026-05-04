#!/bin/bash
# Launch the ball-color HSV calibrator. Output streams to this terminal.
# Run from the board's local HDMI terminal so the OpenCV preview window can render.
# Stop main.py first — this script grabs the USB arm bus to release wrist torque.
# Usage: ./calibrate.sh [--camera N] [--width W] [--height H] [--output FILE]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /root/miniforge3/etc/profile.d/conda.sh
conda activate iotc-tria-xarm

exec stdbuf -oL -eL python -u ball_calibrate.py "$@"
