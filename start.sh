#!/bin/bash
# Launch the IoTConnect XArm demo. Output streams to this terminal — Ctrl-C to stop.
# Pass --mode {asl,ball,pickplace} and other main.py args after the script name.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /root/miniforge3/etc/profile.d/conda.sh
conda activate iotc-tria-xarm

# stdbuf + python -u so lines flush immediately rather than getting buffered
# into multi-second chunks when stdout isn't a terminal-detected fd.
exec stdbuf -oL -eL python -u main.py "$@"
