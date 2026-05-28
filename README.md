# Tria VisionAI Kit 6490 + IOTCONNECT — XArm Vision Demos

Three independent vision-driven robotic-arm demos that all run on the same
**Tria VisionAI Kit 6490** (Qualcomm QCS6490 SoC) board and stream telemetry +
accept remote commands through **IOTCONNECT**. Each demo highlights a
different rung of perception sophistication, on the same hardware:

```
   1. ASL gesture control       MediaPipe + PointNet    CPU
   2. HSV pick-and-place        Classical CV (OpenCV)   CPU
   3. YOLO + depth pick-and-place  Custom YOLOv8n + MiDaS-V2   Hexagon NPU
```

The progression is the demo: same arm, same camera, same cloud platform —
showing classical CV vs. NPU-accelerated deep learning side by side.

---

## The three demos

### 1. ASL Gesture Control  (`--mode asl`)

**Value prop.** Natural human–robot interaction. The operator stands in front
of the camera and signs single-handed ASL letters; the arm responds in
real-time. Demonstrates **on-board hand tracking (MediaPipe) + landmark
classification (PointNet)** with no operator setup beyond launching it.

**Best for**: showing the kit running a useful, end-user-facing AI experience
with no labelled data and no per-site calibration.

**Launch**:
```
./start.sh --mode asl
```

**Quick reference**: [`RUNBOOK_ASL.txt`](RUNBOOK_ASL.txt) — gesture mappings,
common issues.

---

### 2. HSV Pick-and-Place  (`--mode ball` or `--mode pickplace`)

**Value prop.** Fully autonomous pick-and-drop using **purely classical
computer vision** — HSV color thresholding, contour analysis, and a
proportional visual-servo controller. **No machine learning, no NPU.** Runs
fast on CPU, is fully deterministic, easy to debug, and works on any board
with OpenCV.

**Best for**: a baseline showing that classical CV is still a viable
production technique for known-color, controlled-lighting tasks. Also makes
the NPU comparison meaningful in demo #3.

**Trade-off**: brittle to lighting changes, requires color recalibration per
site, can't tell a colored ball from a similarly-colored backdrop.

**Launch**:
```
./start.sh --mode pickplace        # find box, grab ball, drop in box
./start.sh --mode ball              # ball-follow only (no drop phase)
```

**Quick reference**: [`RUNBOOK_HSV.txt`](RUNBOOK_HSV.txt) — calibration
workflow (`ball_calibrate.py`, `teach_pose.py`), tuning constants, and
troubleshooting.

---

### 3. YOLO + Depth Pick-and-Place  (NPU)

**Value prop.** The classical CV demo, but with **two neural networks
running concurrently on the Hexagon NPU**:

- **YOLOv8n** (custom-trained on the actual ball at this site) replaces the
  HSV color filter. Robust to lighting, partial occlusion, and similar-color
  backgrounds.
- **MiDaS-V2 monocular depth** replaces the pixel-radius "is the ball
  close" proxy. The grab gate fires on real (relative) distance — independent
  of ball size or camera focus.

Combined NPU load: **~7 ms YOLO + ~5 ms depth per frame** through the QNN
TFLite delegate, leaving the CPU free for the control loop. Live latency,
fps, and depth values stream to IOTCONNECT as telemetry.

**Best for**: showing the QCS6490's Hexagon NPU doing real, useful AI work
that the CPU-only demo can't do — with side-by-side comparison via demo #2.

**Trade-off**: requires a captured + labelled dataset and a one-time INT8
TFLite conversion (we use Qualcomm AI Hub semantics via the bundled MiDaS).
Also requires a per-site **D_grab teach** (the grab-distance reference is
scene-specific).

**Launch (recommended — opinionated defaults)**:
```
./start.sh --mode yolo-pickplace --web-port 8080
```
Then watch in any browser at `http://<board-ip>:8080/`. Over SSH the demo
auto-detects there's no display and runs headless; locally with a monitor
attached you can omit `--web-port` and a desktop window opens.

**Launch (full CLI control — pick your own model, conf, camera, depth on/off)**:
```
./start_yolo.sh --model model/ball_best.tflite --depth --conf 0.7 \
                --camera 3 --headless --web-port 8080
```

The unified `./start.sh --mode yolo-pickplace` auto-picks
`model/ball_best.tflite` if present (else falls back to the stock board
YOLO-X), enables depth if MiDaS is installed, and arms the depth-gated grab
if `grab_depth.json` exists. Use `./start_yolo.sh` when you need to override
any of those.

**Quick reference**: [`RUNBOOK_YOLO.txt`](RUNBOOK_YOLO.txt) — pre-flight,
re-teach-per-location workflow, perf telemetry.

**Reproduction guide**: [`CUSTOM_NPU_DETECTOR.md`](CUSTOM_NPU_DETECTOR.md) —
how to retrain the YOLO model end-to-end (capture, label, Colab training,
INT8 export, NPU deployment).

---

## Hardware

- **[Tria VisionAI-Kit 6490](https://www.newark.com/avnet/sm2-sk-qcs6490-ep6-kit001/dev-kit-64bit-arm-cortex-a55-a78/dp/51AM9843)** — Qualcomm QCS6490 SoC compute platform.
- **[Hiwonder xArm 1S](https://www.amazon.com/LewanSoul-Programmable-Feedback-Parameter-Programming/dp/B0CHY63V9P)** — 6-DOF arm + gripper, USB.
- **USB camera**:
  - For demos #2/#3: mounted on the wrist roll servo (eye-in-hand).
  - For demo #1: any USB camera facing the operator.
  - The reference build uses a Logitech Brio 100; any UVC camera works.
- One yellow practice ball (Nerf Rival Ammo balls work well — small,
  matte, compress slightly inside the gripper).
- A target drop box (any small container in a clear color, e.g. blue).
- USB-C power supply + cable (in kit).
- Ethernet (or Wi-Fi) for IOTCONNECT.
- HDMI monitor, USB keyboard, USB mouse — optional but useful for first
  setup.

---

## First-time setup

### 1. Board bring-up

1. Power up the board (12 VDC via USB-C #1), hold S1 for 2–3 seconds.
2. Find its IP (DHCP) — connect a monitor, or check your router's lease table.
3. SSH in: `ssh root@<board-ip>` (password `oelinux123`).

### 2. Clone + Python env

```bash
cd ~
git clone https://github.com/avnet-iotconnect/iotc-tria-vision-ai-kit-robotic-arm.git
cd iotc-tria-vision-ai-kit-robotic-arm

# miniforge (one-time)
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

conda create -y -n iotc-tria-xarm python=3.11
conda activate iotc-tria-xarm
conda install opencv -c conda-forge
pip3 install -r requirements.txt

# ASL model weights
source model/get_model.sh

# Optional: depth-gated grab dependency (only for demo #3)
pip3 install -r requirements-yolo.txt
```

### 3. IOTCONNECT onboarding

Follow the
[device onboarding guide](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/blob/main/common/general-guides/UI-ONBOARD.md)
to register this board in IOTCONNECT and drop the resulting
`iotcDeviceConfig.json` + `device-cert.pem` + `device-pkey.pem` into the
project root. Once placed, all three demos publish telemetry and accept
remote commands automatically.

Per-payload telemetry includes a top-level `state` field identifying the
active mode (`IDLE`, `SCANNING`, `TRACKING`, `GRABBING`, `HOLDING`,
`BALL_PHASE`, etc.) plus mode-specific augmentations (`ballTrack` in HSV
mode; `yolo_ms`/`depth_ms`/`npu_fps`/`depth_at_ball` in YOLO mode).

### 4. Sanity-check the hardware

```bash
lsusb | grep 0483:5750     # should show the xArm
v4l2-ctl --list-devices    # should list the wrist camera
ifconfig eth1              # should show an IPv4 address
```

---

## Remote commands via IOTCONNECT

Available in all three modes, regardless of which is running:

| Command           | Effect                                            |
|-------------------|---------------------------------------------------|
| `home`            | Return arm to a safe centered pose                |
| `open_gripper`    | Open the gripper                                  |
| `close_gripper`   | Close the gripper                                 |
| `advance`         | Move the arm forward                              |
| `backup`          | Move the arm backward                             |
| `left` / `right`  | Move the arm laterally                            |
| `up` / `down`     | Move the arm vertically                           |

Mode-switch commands also work — IOTCONNECT can swap which demo is running
without restarting the process. See the demo-specific runbooks for the
mode-switch command names.

---

## Picking a demo

| If you want to show... | Run |
|---|---|
| Natural human interaction, no setup | Demo #1 — ASL |
| A clean, reliable autonomous pick on a controlled surface | Demo #2 — HSV pickplace |
| The NPU doing real work, comparing modern ML against classical CV | Demo #3 — YOLO + depth |
| End-to-end "evolution" story (classical → ML → ML+depth) | Run #2, then #3 |

---

## Troubleshooting (cross-cutting)

| Symptom | First thing to check |
|---|---|
| `lsusb` doesn't show xArm | Power on the arm BEFORE plugging USB; re-plug after powering on |
| Camera not detected | `v4l2-ctl --list-devices` — note the Brio's current `/dev/videoN` |
| IOTCONNECT offline | Ethernet up? `ping 8.8.8.8`? Certs in project root? |
| Demo crashes on Qt platform plugin | Run with `--headless --web-port 8080` over SSH |
| Stuck process won't quit | `pkill -9 -f main.py` or `pkill -9 -f yolo_pickplace.py` |

Mode-specific troubleshooting lives in each `RUNBOOK_*.txt`.

---

## References

### Tria VisionAI Kit 6490
- [Tria VisionAI Kit 6490 Setup Guide](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/tree/main/tria-vision-ai-kit-6490)
- [Tria VisionAI-Kit 6490 Product Page](https://www.newark.com/avnet/sm2-sk-qcs6490-ep6-kit001/dev-kit-64bit-arm-cortex-a55-a78/dp/51AM9843)
- [Tria Startup Guide](https://avnet.com/wcm/connect/137a97f1-eb6e-48ba-89a4-40b024558593/Vision+AI-KIT+6490+Startup+Guide+v1.3.pdf)

### IOTCONNECT
- [IOTCONNECT Python SDK](https://github.com/avnet-iotconnect/avnet-iotconnect-python-sdk)
- [IOTCONNECT device onboarding](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/blob/main/common/general-guides/UI-ONBOARD.md)
- [IOTCONNECT platform](https://www.iotconnect.io/)

### Robotics & AI
- [xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
- [ASL MediaPipe + PointNet](https://github.com/AlbertaBeef/asl_mediapipe_pointnet)
- [Qualcomm AI Hub](https://aihub.qualcomm.com) — source of the bundled
  YOLO-X, MiDaS-V2, and other Hexagon-optimized models on this board.
