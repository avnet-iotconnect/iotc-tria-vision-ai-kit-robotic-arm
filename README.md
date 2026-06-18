# Tria VisionAI Kit 6490 + IOTCONNECT — XArm Vision Demos

Three independent vision-driven robotic-arm demos that all run on the same
**Tria VisionAI Kit 6490** (Qualcomm QCS6490 SoC) board and stream telemetry +
accept remote commands through **IOTCONNECT**. Each demo highlights a
different rung of perception sophistication on the same hardware:

```
   1. ASL gesture control          MediaPipe + PointNet          CPU
   2. HSV pick-and-place           Classical CV (OpenCV)         CPU
   3. YOLO + depth pick-and-place  Custom YOLOv8n + MiDaS-V2    Hexagon NPU
```

The progression is the demo: same arm, same camera, same cloud platform —
showing classical CV vs. NPU-accelerated deep learning side by side.

---

## Hardware

- **[Tria VisionAI-Kit 6490](https://www.newark.com/avnet/sm2-sk-qcs6490-ep6-kit001/dev-kit-64bit-arm-cortex-a55-a78/dp/51AM9843)** — Qualcomm QCS6490 SoC compute platform.
- **[Hiwonder xArm 1S](https://www.amazon.com/LewanSoul-Programmable-Feedback-Parameter-Programming/dp/B0CHY63V9P)** — 6-DOF arm + gripper, USB.
- **USB camera**:
  - For demos #2/#3: mounted on the wrist roll servo (eye-in-hand).
  - For demo #1: any USB camera facing the operator.
  - The reference build uses a Logitech Brio 100; any UVC camera works.
- One yellow practice ball (Nerf Rival Ammo balls work well — small, matte,
  compress slightly inside the gripper).
- A target drop box (any small container in a clear color, e.g. blue).
- USB-C power supply + cable (in kit).
- Ethernet (or Wi-Fi) for IOTCONNECT.
- HDMI monitor, USB keyboard, USB mouse — optional but useful for first setup.

---

## First-Time Setup

### 1. Board Bring-Up

1. Power up the board (12 VDC via USB-C #1), hold S1 for 2–3 seconds.
2. Find its IP (DHCP) — connect a monitor, or check your router's lease table.
3. SSH in: `ssh root@<board-ip>` (password `oelinux123`).

### 2. Clone + Python Environment

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

# Optional: NPU inference dependency (only for demo #3)
pip3 install -r requirements-yolo.txt
```

### 3. ASL Model Weights (Demo #1 Only)

The PointNet classifier weights are not bundled in the repo. Download them once:

```bash
source model/get_model.sh
ls -lh model/point_net_1.pth   # should be ~38–42 MB
```

Demo #1 will refuse to start with a clear error if this file is absent.

### 4. IOTCONNECT Onboarding

Follow the
[device onboarding guide](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/blob/main/common/general-guides/UI-ONBOARD.md)
to register this board in IOTCONNECT and drop the resulting
`iotcDeviceConfig.json` + `device-cert.pem` + `device-pkey.pem` into the
project root. Once placed, all three demos publish telemetry and accept remote
commands automatically.

> **Template note:** This repo ships two device templates:
> - `robarmwebrtc-template.json` — use this if you want **live WebRTC video streaming** from the wrist camera to the IOTCONNECT portal. During device creation select **WebRTC** as the Stream Resource.
> - Standard template — use the default IOTCONNECT onboarding flow if you don't need WebRTC streaming.
>
> Devices created with `robarmwebrtc-template.json` can enable streaming with `--webrtc`. Devices created without it work normally without the flag. The stream-resource choice cannot be changed after device creation — if needed, create a new device.

---

## Demo 1 — ASL Gesture Control (`--mode asl`)

**Value prop.** Natural human–robot interaction. The operator stands in front of
the camera and signs single-handed ASL letters; the arm responds in real-time.
Demonstrates **on-board hand tracking (MediaPipe) + landmark classification
(PointNet)** with no per-site calibration once the model weights are downloaded.

**Best for:** showing the kit running a useful, end-user-facing AI experience
with no labelled data and no per-site calibration.

### Camera Setup

**Placement.** Use any operator-facing USB UVC camera — not the wrist-mount used
by demos #2 and #3. The wrist-mounted Brio can double as the operator-facing
camera if you re-mount it; a second USB webcam at a typical monitor-top position
also works.

- Distance: 1.5–3 feet from the signing hand.
- Height: shoulder-to-chest level gives the best hand-to-background contrast.
  Avoid angles that put bright ceiling lights directly behind the hand.
- Frame fill: the hand should occupy roughly ¼–⅓ of the frame. MediaPipe Hands
  performs best when the hand is large and centered, not a small blob in a corner.

**Lighting.** Even, diffuse front-lighting works best. Avoid strong backlighting
(operator in front of a bright window) — the hand goes silhouette and MediaPipe
loses landmarks. A neutral, low-clutter background helps recognition.

**Camera index.** The app auto-detects the Brio 100 by kernel device name. For
a different camera, find its index with `v4l2-ctl --list-devices` and pass it
explicitly:
```bash
./start.sh --mode asl --camera N
```

### Running the Demo

```bash
./start.sh --mode asl                                        # local monitor
python3 main.py --mode asl --headless --web-port 8080        # over SSH
```

Watch the live stream at `http://<board-ip>:8080/` if running `--web-port`.

**Startup sequence:**
1. App connects to the xArm and moves to the home/center pose.
2. IoTConnect connects (if certs are in the project root).
3. PointNet loads from `model/point_net_1.pth` (~2–3 seconds).
4. Camera loop starts. The OSD is blank until a hand appears.

**On-screen overlay:**
- `LEFT=<letter>` (top-left, green) — left hand detected, letter classified
- `RIGHT=<letter>` (top-right, red) — right hand detected, letter classified
- `[<action label>]` below the letter — the arm command that will fire
- Hand skeleton wireframe drawn over the detected landmarks

**Gesture mechanic.** A command fires each time the recognized sign **changes**
— not on every frame. This prevents a held gesture from flooding the arm with
repeat commands. To repeat the same action: briefly drop the hand out of frame
(or form a different sign), then re-sign.

**Model coverage.** The PointNet model classifies 24 static ASL letters (A–Y,
excluding J and Z which require motion). Every frame is classified into one of
these 24 letters — only the mapped letters in the table below trigger arm
commands. Unmapped letters appear in the overlay without an action label.

### Gesture Mapping

| Hand | Sign | Arm action |
|------|------|------------|
| Left | A | Advance (move forward) |
| Left | B | Back up (move backward) |
| Left | L | Left (shoulder pan) |
| Left | R | Right (shoulder pan) |
| Left | U | Up (raise) |
| Left | Y | Down (lower) |
| Left | H | Home (all servos to center) |
| Right | A | Close gripper |
| Right | B | Open gripper |

All other recognized letters display in the overlay but produce no arm action.

> **Direction note.** The physical direction of Advance/Backup/Left/Right depends
> on how the arm is mounted. If the arm moves the wrong way, adjust
> `LEFT_GESTURE_TO_ACTION` in `modes/asl.py` (swap `'advance'`/`'backup'` or
> `'left'`/`'right'`).

### Telemetry — ASL

Sent on every gesture-triggered arm action and every 5 seconds while running.

| Field | Description |
|-------|-------------|
| `state` | `"ASL-Gesture"` (constant) |
| `gripper`, `wrist_roll`, `wrist_flex`, `elbow_flex`, `shoulder_lift`, `shoulder_pan` | Servo positions (0–1000 units) |
| `sysInfo_cpu` | CPU usage + temperature |
| `sysInfo_memory` | RAM usage + temperature |
| `sysInfo_storage` | Disk usage |
| `sysInfo_gpu` | Adreno GPU % (near zero — workload is on CPU) |

If fields are missing from the IOTCONNECT dashboard, verify they are declared on
the device template — the broker silently drops undeclared attributes.

---

## Demo 2 — HSV Pick-and-Place (`--mode ball` / `--mode pickplace`)

**Value prop.** Fully autonomous pick-and-drop using **purely classical computer
vision** — HSV color thresholding, contour analysis, and a proportional
visual-servo controller. No machine learning, no NPU. Runs fast on CPU, fully
deterministic, easy to debug, and works on any board with OpenCV.

Two flavors:
- `--mode ball` — ball-follow + grab + return home (no drop phase)
- `--mode pickplace` — find the drop box, grab the ball, drop it in the box

**Best for:** a clean baseline showing that classical CV is still viable for
known-color, controlled-lighting tasks — and making the NPU comparison in
demo #3 meaningful.

**Trade-off:** brittle to lighting changes; requires per-site color
recalibration; can't distinguish a colored ball from a similarly-colored
backdrop.

### One-Time Calibration (Per Site / Per Ball / Per Lighting)

Three pieces of calibration data are required. Capture them in order — each
step writes a JSON file the next may depend on.

> **Safety note.** All calibration scripts release all six servo torques so you
> can free-pose the arm to aim the wrist camera. **Always support the arm before
> pressing Enter**, especially on a wall- or ceiling-mounted arm.

#### 2a. Ball HSV Thresholds — `ball_calibrate.py`

```bash
./calibrate.sh
# or:
python3 ball_calibrate.py [--camera N]
```

1. Hold the arm, Enter → torque drops.
2. Click the ball in the live preview. Each click samples a 7×7 HSV patch and
   widens the mask range. The overlay shows what the detector will see.
3. `h` = re-engage torque at current pose; `w` = release again to re-aim.
4. `s` = save → writes `ball_color.json`
   `r` = reset samples  `q`/ESC = quit without saving.

#### 2b. Scan Poses — `teach_pose.py`

Captures the arm poses cycled through while the camera sweeps for the ball.

```bash
./teach.sh
# or:
python3 teach_pose.py
```

1. Hold the arm, Enter → torque drops.
2. Pose the camera at one position: center, left-edge, or right-edge of the
   area where the ball will be placed.
3. `s` = snapshot. The script prints a `SCAN_POSE = [...]` block to paste into
   `modes/ball_follow.py`.
4. `h` = re-enable torque (do this before letting go!).
5. Repeat for each pose; `q` to quit.

#### 2c. Drop Box HSV Thresholds — `ball_calibrate.py` (Pickplace Only)

```bash
python3 ball_calibrate.py --output box_color.json [--camera N]
```

Same UI as ball-color calibration. Click on the drop box in the live preview.
The pickplace mode reads `box_color.json` to find and approach the drop target.

#### 2d. Camera-Gripper Offset — `calibrate_cam_offset.py` (Optional)

Skip unless the gripper consistently closes next to the ball rather than on it.

```bash
./calibrate_offset.sh
# or:
python3 calibrate_cam_offset.py [--camera N]
```

1. Hold the arm, Enter → torque drops. Pose the gripper directly over the ball
   at the height it would normally grab from.
2. OSD shows live `(bx, by)` and the resulting offset.
3. `s` = snapshot. Averages ~30 frames and prints `CAM_GRIPPER_OFFSET_X/_Y`
   to paste into `modes/ball_follow.py`.
4. `h` = re-enable torque, `q` = quit.

### Running the Demo

```bash
# Ball-follow only (no drop phase):
./start.sh --mode ball
python3 main.py --mode ball --headless --web-port 8080

# Pick-and-place (find box, grab, drop):
./start.sh --mode pickplace
python3 main.py --mode pickplace --headless --web-port 8080
```

Watch the live stream at `http://<board-ip>:8080/` if running `--web-port`.

The arm homes, then enters SCANNING. Drop the ball anywhere in the scan
envelope. The arm centers on it, descends until the apparent radius matches
the grab-distance target, closes the gripper, and either returns home (ball
mode) or carries the ball to the taught box position and releases (pickplace
mode). Open the gripper manually or via the `open_gripper` cloud command to
re-arm the cycle.

### State Machine

```
IDLE      → SCANNING                               (immediately at launch)
SCANNING  → TRACKING                               (ball detected)
TRACKING  → TRACKING  (centering pan/tilt + descent)
          → GRABBING                               (centered AND radius OK)
          → SCANNING                               (ball lost > grace window)
GRABBING  → HOLDING                               (gripper closed)
HOLDING   → IDLE                                  (gripper opened by user or IOTCONNECT)
```

`NO_BALL_GRACE_FRAMES` (~5 s at 6 fps) lets the arm hold its pose during brief
detection drop-outs from clipping or HSV flicker, instead of bouncing back to
SCAN every time the ball flickers off for a frame.

### Key Tuning Constants (`modes/ball_follow.py`)

| Constant | Default | Notes |
|----------|---------|-------|
| `PAN_GAIN`, `TILT_GAIN` | — | Servo units commanded per pixel of error. `TILT_GAIN` is higher because wrist_flex fights gravity at extended poses. |
| `PAN_DIR`, `TILT_DIR`, `TILT_ELBOW_DIR` | `+1` | Sign flips. Set to `-1` if the arm moves *away* from the ball instead of toward it. Mount-dependent. |
| `MIN_TRIM_STEP` | 8 | Floor on non-zero P-controller commands. Bus servos silently ignore commands below ~5 units (static friction). |
| `MIN_TRIM_STEP_PAN` | 18 | Higher floor for shoulder_pan — it carries the whole forearm + wrist + camera, so static friction is ~2× the wrist's. |
| `APPROACH_STEP` | 15 | Per-frame shoulder_lift step during descent. Below ~12 the lift servo can't break friction at extended poses (shoulder_lift > 600). |
| `MAX_STEP` | 25 | Hard cap on any single-frame servo delta. |
| `MOVE_DURATION_MS` | 220 | Duration of each per-frame move. Too short = small commands ignored; too long = loop rate drops. |
| `CENTER_DEADBAND_PX` | 60 | Pixel error inside which centering trims stop. Must be ≥ `MIN_TRIM_STEP_PAN × pixels/unit` (~7 px/unit) or the controller oscillates. |
| `APPROACH_DEADBAND_PX` | — | Looser threshold — once inside this, the arm can descend while still fine-centering. |
| `TARGET_RADIUS_PX`, `RADIUS_TOLERANCE` | — | Apparent ball radius (px) meaning "close enough to grab". Tune by watching the OSD `r=` value at the moment you want the grab to fire. |
| `CAM_GRIPPER_OFFSET_X/Y` | 0/0 | Pixel offset between camera optical axis and gripper jaws. Only set after running `calibrate_cam_offset.py`. |
| `LIFT_MAX`, `ELBOW_MAX` | — | Safety ceilings so a never-satisfied radius check can't drive the gripper into the table. |
| `NO_BALL_GRACE_FRAMES` | — | Consecutive lost-ball frames before falling back to SCANNING. Increase if the ball flickers at the frame edge. |
| `SCAN_POSES` | — | Poses cycled while searching. Captured with `teach_pose.py`. |

> All gain/step constants scale implicitly with camera frame rate. If you
> change camera resolution or drop the preview, expect to re-tune.

### Telemetry — HSV

Each payload (sent every ~2 s) carries:

| Field | Description |
|-------|-------------|
| `state` | `IDLE` / `SCANNING` / `TRACKING` / `GRABBING` / `HOLDING` |
| `ballTrack.ball_x`, `ball_y`, `ball_r` | Last detected pixel position + radius (0 if not seen) |
| `ballTrack.pan_err`, `tilt_err` | Current centering error in pixels |
| `ballTrack.radius_err` | Apparent-radius error from `TARGET_RADIUS_PX` |
| `ballTrack.velocity_x`, `velocity_y` | Recent ball motion (px/frame) |
| `ballTrack.d_pan`, `d_tilt`, `d_lift`, `d_elbow` | Last commanded servo deltas |
| `ballTrack.is_prediction` | 1 if current bbox is extrapolated, not detected |
| `ballTrack.no_ball_frames` | Consecutive frames without a detection |
| `ballTrack.pred_frames_left` | Extrapolation budget remaining |

---

## Demo 3 — YOLO + Depth Pick-and-Place (`--mode yolo-pickplace`)

**Value prop.** The HSV demo, but with **two neural networks running concurrently
on the Hexagon NPU**:

- **YOLOv8n** (custom-trained on the actual ball) replaces the HSV color filter.
  Robust to lighting, partial occlusion, and similar-color backgrounds.
- **MiDaS-V2 monocular depth** replaces the pixel-radius distance proxy. The grab
  gate fires on real relative distance — independent of ball size or camera focus.

Combined NPU load: **~7 ms YOLO + ~5 ms depth per frame** through the QNN TFLite
delegate, leaving the CPU free for the control loop. Live latency, fps, and depth
values stream to IOTCONNECT as telemetry.

**Best for:** showing the QCS6490 Hexagon NPU doing real, useful AI work that the
CPU-only demo can't — with side-by-side comparison via demo #2.

**Trade-off:** requires a per-site **D_grab teach** (MiDaS depth values are
scene-specific). Also requires a captured + labelled dataset and one-time INT8
TFLite conversion for a custom YOLO model (see
[`CUSTOM_NPU_DETECTOR.md`](CUSTOM_NPU_DETECTOR.md)).

### Per-Site Re-Teaching

Same ball + same arm + new room = a few things need re-teaching. MiDaS depth
values, HSV thresholds for the drop box, and arm-relative scene geometry are all
site-dependent.

| File | Required? | Command |
|------|-----------|---------|
| `grab_depth.json` | **Required at every new site** | `./teach_grab.sh --model model/ball_best.tflite --camera N --web-port 8080` |
| `drop_pose.json` | Only if drop target moved | Send `release_torque`, hand-pose arm, then `hold_pose` + `teach_drop_pose` from cloud |
| `scan_poses.json` | Only if ball staging area moved | `python3 teach_pose.py` |
| `box_color.json` | Only if drop box or lighting changed | `python3 ball_calibrate.py --output box_color.json --camera N` |
| `camera_settings.json` | Only if lighting is very different | Re-run the exposure/WB lock script |

#### Re-Teaching D_grab (Required at Every New Site)

MiDaS depth values are scene-specific — the same physical grab pose can read
100+ units differently across lighting conditions and camera enumerations.

1. Stop any running demo:
   ```bash
   pkill -f yolo_pickplace.py
   ```

2. Run the teach helper:
   ```bash
   ./teach_grab.sh --model model/ball_best.tflite --camera N --web-port 8080
   ```

3. Open `http://<board-ip>:8080/` in a browser to see the wrist camera.

4. **Hold the arm by hand**, press Enter → torque releases. The arm will fall
   under gravity if you let go.

5. Hand-pose the gripper exactly where you want it at the grab moment. The ball
   should sit just below the gripper jaws and be fully visible to the camera.

6. **While still holding the arm**, press `s` + Enter. Captures 20 frames and
   saves the median D as `grab_depth.json`. (The script refuses to save if you
   press `h` first — a locked-arm capture has zero stdev and is useless.)

7. `h` + Enter to lock the pose, then `q` + Enter to quit cleanly.

Sanity check:
```bash
cat grab_depth.json
```
`D_stdev` should be roughly 10–30, **not 0**. If it's 0, redo the snapshot
before pressing `h`.

### Running the Demo

**Option A — unified launcher (recommended, opinionated defaults):**
```bash
./start.sh --mode yolo-pickplace --web-port 8080 2>&1 | tee /tmp/yolo.log
./start.sh --mode yolo-ball --web-port 8080          # ball-follow only
```

**Option B — full CLI control (override model, conf, camera, depth):**
```bash
./start_yolo.sh --model model/ball_best.tflite --depth --conf 0.7 \
                --camera N --headless --web-port 8080 \
                2>&1 | tee /tmp/yolo.log
```

Open `http://<board-ip>:8080/` to watch the live annotated stream. Over SSH
the demo auto-detects there's no display and runs headless automatically.

**Defaults applied by the unified launcher:**

| Setting | Value |
|---------|-------|
| Model | `model/ball_best.tflite` (falls back to `/etc/models/yolox_quantized.tflite`) |
| Confidence | 0.7 for custom model, 0.25 for stock YOLO-X |
| Depth | Enabled if `/etc/models/midas_quantized.tflite` is present |
| Depth gate | Armed if `grab_depth.json` exists; else legacy radius gate |

**Expected startup log:**
```
[yolo] model=ball_best.tflite format=yolov8 nc=1 in=640x640 float32
       NPU=YES (Hexagon HTP via QNN)
[depth] model=midas_quantized.tflite in=256x256 uint8
        NPU=YES (Hexagon HTP via QNN)
[yolo-ball] depth gate ARMED: D_grab=... +/-...
[yolo-pickplace] entering BALL_PHASE...
```

**Live overlay during a pick:**
```
r=152 dx=+4 dy=-122                              ← YOLO bbox + pan/tilt error
center=OK  settle 2/3  (D >= 829, target 860)    ← gate state
D=860 (raw 884)  higher=closer                   ← smoothed and raw depth
NPU: yolo=7.2ms  depth=4.8ms  14 fps             ← live inference perf
```

The grab fires when `center=OK` AND settle reaches 3/3 (`DEPTH_SETTLE_FRAMES`
consecutive in-window frames). Closer-than-taught still counts as OK — the gate
is a floor, not a window.

### Control-Loop Tuning (`modes/yolo_pickplace.py`)

| Constant | Default | Notes |
|----------|---------|-------|
| `PAN_COOLDOWN_MS` | 250 | Minimum gap between shoulder_pan corrections. Pan can't issue commands smaller than ~18 units without stalling; without a cooldown the loop retargets before the servo finishes and the ball oscillates. Lower → faster tracking; higher → more stable settling. |
| `CENTERED_LATCH_FRAMES` | 0 | Latch `centered_ok` for N frames after first becoming true. Latch ≥ 4 fires too early in testing — keep at 0. |
| `DEPTH_SETTLE_FRAMES` | 3 | Consecutive in-window depth_ok frames before firing the grab. At ~12–15 Hz this is ~200 ms of sustained in-window depth. |
| `D_EWMA_ALPHA` | 0.25 | EWMA smoothing factor for MiDaS depth. Higher → more responsive; lower → heavier smoothing. 0.25 ≈ 4-frame effective average. |
| `DEPTH_TOL_FLOOR` | 20 | Minimum effective gate tolerance — protects against pathologically tight teach captures. |
| `DEPTH_TOL_MULT` | 1.5 | Tolerance multiplier: effective tolerance = `max(FLOOR, MULT × D_stdev_from_teach)`. |

### Telemetry — YOLO

Each `ballTrack` payload (~2 s) carries all HSV-mode fields plus:

| Field | Description |
|-------|-------------|
| `state` | `BALL_PHASE` / `BOX_PHASE` / `SCANNING` / `TRACKING` / `GRABBING` / `HOLDING` |
| `yolo_ms` | Live YOLO inference time on Hexagon HTP |
| `depth_ms` | Live MiDaS inference time on Hexagon HTP |
| `npu_fps` | Effective end-to-end frame rate |
| `depth_at_ball` | Raw D at the ball (single frame) |
| `depth_at_ball_smoothed` | EWMA-smoothed D (what the control law uses) |
| `depth_settle` | Consecutive in-window depth_ok frames (0–3) |
| `D_grab` | Currently-loaded teach value |
| `D_tolerance` | Effective gate width |

Note: `sysInfo_gpu` will stay near zero — the workload is on the Hexagon NPU,
not the Adreno graphics GPU. NPU activity is visible through `yolo_ms` and
`depth_ms`.

---

## Remote Commands (All Modes)

These commands work via IOTCONNECT regardless of which demo is running:

| Command | Effect |
|---------|--------|
| `home` | Return arm to safe centered pose |
| `open_gripper` | Open the gripper |
| `close_gripper` | Close the gripper |
| `advance` / `backup` | Move arm forward / backward |
| `left` / `right` | Move arm laterally |
| `up` / `down` | Move arm vertically |
| `set_mode mode=<name>` | Switch active demo without restarting (`asl`, `ball`, `pickplace`, `yolo-ball`, `yolo-pickplace`) |
| `set_mode mode=idle` | Stop the running demo; arm + cloud stay connected |
| `calibrate target=<name>` | Launch a browser-UI calibrator (`ball`, `box`, `offset`, `grab_depth`) — open `http://<board-ip>:8000/` |
| `release_torque` | Drop all servo torques for hand-posing |
| `hold_pose` | Re-engage torque at the current pose |
| `teach_scan_pose name=<n>` | Snapshot current arm pose into a named scan slot |
| `camera_setting <name> <value>` | Update a camera setting live (exposure, saturation, etc.) |

---

## Custom NPU Detector — End-to-End Reproduction

[`CUSTOM_NPU_DETECTOR.md`](CUSTOM_NPU_DETECTOR.md) covers the full pipeline for
retraining the YOLO model from scratch on your own ball: image capture on the
board, HSV auto-labelling, labelImg review, train/val split, Colab YOLOv8n
training, INT8 TFLite export, and NPU deployment. Start there if the bundled
model doesn't recognize your specific ball.

---

## Live WebRTC Video Streaming (Optional)

Any demo can stream the wrist-camera feed live to the IOTCONNECT portal using
AWS Kinesis Video Streams (KVS) WebRTC. This is entirely opt-in and does not
affect vision or arm control when unused.

### Prerequisites

Your device **must** be created in IOTCONNECT using the `robarmwebrtc`
template (`robarmwebrtc-template.json` in this repo). During device creation,
when prompted to select a **Stream Resource**, choose **WebRTC**. This choice
cannot be changed after device creation — if your device was created with a
different template, create a new one.

### Enabling WebRTC Streaming

Add the `--webrtc` flag when launching any demo:

```bash
python3 main.py --mode ball --webrtc
# or headless over SSH:
python3 main.py --mode asl --webrtc --headless
```

When connected, the stream appears in the IOTCONNECT portal under your device's
live view. The wrist-camera feed updates at ~15 fps independent of the main
vision loop so arm moves don't stall the stream.

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
- [Qualcomm AI Hub](https://aihub.qualcomm.com) — source of the bundled YOLO-X,
  MiDaS-V2, and other Hexagon-optimized models on this board.
