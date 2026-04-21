# TRIA Vision AI Kit 6490 + /IOTCONNECT XArm Vision Demos

This project showcases the **TRIA Vision AI Kit 6490** running **/IOTCONNECT** integration with the Hiwonder XArm 1S robotic arm. It ships two interchangeable vision modes selectable at launch with `--mode`:

- **`asl`** (default) — American Sign Language gesture control. Operator drives the arm by signing letters in front of a webcam.
- **`ball`** — Autonomous eye-in-hand visual servoing. The wrist-mounted camera detects a colored ball, and the arm pans/tilts/advances on its own to center, approach, and grab it.

Both modes run on the TRIA board, stream live telemetry to /IOTCONNECT, and accept remote commands from the cloud.

## Key Features

- **TRIA Vision AI Kit 6490**: Qualcomm QCS6490-powered edge AI platform for real-time gesture recognition
- **/IOTCONNECT Cloud Integration**: Real-time telemetry transmission and remote command execution
- **ASL Gesture Control**: AI-powered American Sign Language recognition using MediaPipe + PointNet
- **Autonomous Ball Pick-Up**: Eye-in-hand HSV detection + proportional visual-servo controller that scans, tracks, approaches, and grabs a colored ball with no operator input
- **Robotic Arm Control**: Hiwonder XArm 1S with 6-DOF movement and gripper control
- **Edge-to-Cloud Architecture**: Local AI inference on TRIA board with cloud connectivity via /IOTCONNECT

## Setup

This project is designed to run on the **TRIA Vision AI Kit 6490** with **/IOTCONNECT** cloud integration, creating a powerful edge-to-cloud AI robotics solution.

### Why TRIA Vision AI Kit 6490 + /IOTCONNECT?

- **TRIA Vision AI Kit 6490**: Energy-efficient Qualcomm QCS6490 SOC with multi-camera support, perfect for real-time AI inference
- **/IOTCONNECT Integration**: Seamless cloud connectivity for telemetry, remote monitoring, and command execution
- **Edge AI**: Run neural network inference locally on TRIA board while streaming results to the cloud
- **Industrial IoT**: Enterprise-grade IoT platform for robotics and automation applications

### Hardware Requirements

- **[TRIA Vision AI-KIT 6490](https://www.newark.com/avnet/sm2-sk-qcs6490-ep6-kit001/dev-kit-64bit-arm-cortex-a55-a78/dp/51AM9843)** - Main compute platform with Qualcomm QCS6490 SOC
- **[HiWonder xArm1S](https://www.amazon.com/LewanSoul-Programmable-Feedback-Parameter-Programming/dp/B0CHY63V9P?th=1)** - Robotic Arm connected via USB
- USB-C Cable for flashing and USB-ADB debug (included with kit)
- USB-C 12VDC Power Supply and Cable (included with kit)
- Ethernet Cable (not included)
- USB camera for hand tracking (ASL mode) and eye-in-hand visual servoing (ball mode). For the ball mode the camera is mounted on the **wrist roll** servo with zip ties so it pitches with the gripper. The build pictured here uses a USB camera module pulled out of a Logitech webcam shell to keep the wrist payload small. **Recommended:** a bare USB-camera PCB module mounted directly behind the gripper jaws — that gives the cleanest line of sight to whatever the gripper is about to grab and removes the parallax that makes the camera-gripper offset calibration necessary.
- HDMI Monitor with Active
- USB Mouse and Keyboard


### Board Setup

1. **Hardware Connections**:
   - Connect 12VDC USB-C power supply to the USB-C connector labeled #1
   - Connect ethernet cable to the board's ethernet port
   - Connect USB mouse/keyboard to USB-A ports
   - Connect second USB-C cable for USB-ADB communication
   - Connect Logitech Camera for hand tracking

2. **Power On**: Hold S1 button for 2-3 seconds until red LED turns off

3. **SSH Connection**:
   - Login as `root` with password `oelinux123`

4. **Clone and Setup Project**:
   ```bash
   git clone https://github.com/avnet-iotconnect/iotc-tria-vision-ai-kit-robotic-arm.git
   cd iotc-tria-vision-ai-kit-robotic-arm
   
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash      Miniforge3-$(uname)-$(uname -m).sh

   conda      create -y -n iotc-tria-xarm python=3.11
   conda      activate iotc-tria-xarm
   conda      install opencv -c conda-forge

   pip3 install -r requirements.txt
   
   source model/get_model.sh

   python3 main.py
   ```

### /IOTCONNECT Device Onboarding

Follow [this guide](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/blob/main/common/general-guides/UI-ONBOARD.md) to onboard your TRIA Vision AI Kit 6490 to /IOTCONNECT.


### Supported Gestures

**Left Hand (Arm Movement)**:
- A: Advance, B: Back-up, L: Left, R: Right, U: Up, Y: Down, H: Home

**Right Hand (Gripper Control)**:
- A: Close Gripper, B: Open Gripper

### Remote Command Control via /IOTCONNECT

Control your XArm robot remotely through /IOTCONNECT cloud commands:
- **Movement Commands**: `advance`, `backup`, `left`, `right`, `up`, `down` for arm positioning
- **Gripper Control**: `open_gripper`, `close_gripper` for object manipulation
- **System Commands**: `home` for safe return to center position
- **Command Acknowledgment**: Real-time feedback and execution confirmation

### /IOTCONNECT Device Configuration
- **Device Config JSON**: Contains your /IOTCONNECT platform credentials and device information
- **Device Certificates**: X.509 certificates for secure cloud connectivity
- **Platform Settings**: AWS IoT, Azure IoT, or other supported IoT platforms

### Remote Command Execution

The system supports real-time command execution through /IOTCONNECT:

**Supported Commands:**
- `home` - Return robot to center position
- `open_gripper` - Open the gripper mechanism
- `close_gripper` - Close the gripper mechanism
- `advance` - Move arm forward
- `backup` - Move arm backward
- `left` - Move arm left
- `right` - Move arm right
- `up` - Move arm up
- `down` - Move arm down

**Command Processing:**
- Commands are queued and executed asynchronously
- Each command receives acknowledgment with execution status
- Commands can be sent while gesture control is active
- Real-time telemetry confirms command execution


## Ball-Follow Mode (Autonomous Visual Servoing)

The `ball` mode turns the XArm into an autonomous pick-and-place demo. The wrist-mounted camera looks for a single colored ball, the controller centers it in the frame, advances until the ball fills the expected radius, then closes the gripper, lifts, and returns to home. No operator input is required after launch.

**Recommended target object:** Nerf Rival Ammo Balls. Their small diameter lets the wrist camera capture the entire ball within the frame even at close approach distance (so the radius-based "close enough" gate stays reliable), and the soft foam compresses slightly inside the gripper jaws — giving a tolerant grab that doesn't require sub-millimeter centering.

### How It Works

The controller aims for the **geometric center** of the camera frame and trusts the gripper to be close enough to that aim point. (An optional camera-gripper offset exists in [modes/ball_follow.py](modes/ball_follow.py) — `CAM_GRIPPER_OFFSET_X / _Y` — for hardware where the wrist camera is mounted noticeably off-axis from the gripper fingers. Both default to `0` for the current build; only set them if you observe the gripper consistently closing next to the ball rather than on it.)

The mode runs as a state machine driven by per-frame HSV detection:

| State        | What it does                                                                                       | Exit                                                              |
|--------------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `IDLE`       | Initial state at launch. Falls through immediately.                                                | → `SCANNING`                                                      |
| `SCANNING`   | Cycles through `SCAN_POSES` (center / left / right) so the camera sweeps the workspace.            | Ball detected → `TRACKING`                                        |
| `TRACKING`   | P-controller drives `shoulder_pan` + `wrist_flex` (with elbow assist) to center the ball pixel.    | Ball centerd AND radius below target → advance; centerd + radius OK → `GRABBING` |
| `GRABBING`   | Closes the gripper, watches the actual position to detect a stall against the ball, then lifts.    | → `HOLDING`                                                       |
| `HOLDING`    | Returns to home (keeping the gripper closed) and waits for the operator to manually open it.       | Gripper opened by user → `IDLE`                                   |

A `NO_BALL_GRACE_FRAMES` window (~5 s at 6 fps) lets the arm hold its current pose during brief detection drop-outs from clipping or HSV flicker, instead of bouncing back into a scan move every time the ball flickers off for a frame.

### Calibration Workflow

The ball mode needs two required pieces of calibration data, plus an optional third. Capture them in this order — each step writes a JSON file (or constants you paste into [modes/ball_follow.py](modes/ball_follow.py)) that the next may depend on.

Both calibration scripts release **all six servo torques** so you can free-pose the entire arm — useful for aiming the wrist camera before sampling. **Always physically support the arm before pressing Enter**; on a wall- or ceiling-mounted arm the whole assembly will swing under gravity the instant torque drops.

#### 1. Ball color — [ball_calibrate.py](ball_calibrate.py)

Captures the HSV thresholds used for ball segmentation.

```bash
./calibrate.sh                # or: python ball_calibrate.py
```

Workflow:
1. **Support the arm**, press Enter to release all torque so you can aim the wrist camera at where the ball will sit.
2. Click the ball in the live preview. Each click samples a 7×7 HSV patch and widens the mask range. The live overlay shows what the ball-follow detector will see.
3. Press `h` to re-engage torque at the current pose, `w` to release again if you need to re-aim.
4. Press `s` to save → writes `ball_color.json`. Use `r` to reset samples, `q`/`ESC` to quit without saving. The script re-engages full torque at the current pose on exit.

#### 2. Scan poses — [teach_pose.py](teach_pose.py)

Captures the arm poses cycled through during `SCANNING`. Torque is dropped so you can pose the arm by hand.

```bash
./teach.sh                    # or: python teach_pose.py
```

Workflow:
1. **Support the arm with your hand** — torque is about to drop and a wall/ceiling-mounted arm will swing under gravity.
2. Press Enter to release torque.
3. Pose the camera at one of the scan positions (center, left edge, right edge).
4. Press `s` + Enter to snapshot. The script prints a `SCAN_POSE = [...]` block ready to paste into [modes/ball_follow.py](modes/ball_follow.py).
5. Press `h` + Enter to re-enable torque before letting go of the arm.
6. Repeat for each pose, then `q` + Enter to quit.

#### 3. Camera-gripper offset (OPTIONAL) — [calibrate_cam_offset.py](calibrate_cam_offset.py)

Skip this step unless you observe the gripper consistently closing next to the ball rather than on it. The default build aims at the geometric image center (`CAM_GRIPPER_OFFSET_X = CAM_GRIPPER_OFFSET_Y = 0`) and that is correct for the current camera mount. Only run this if you change the camera mount, swap the gripper, or notice a systematic miss.

```bash
./calibrate_offset.sh         # or: python calibrate_cam_offset.py
```

Workflow:
1. Place the ball on the table.
2. **Support the arm**, press Enter to release all torque, then physically pose the gripper directly over the ball at the height it would normally grab from.
3. The live OSD shows the detected ball's `bx, by` and the resulting `OFFSET_X / OFFSET_Y`.
4. Hold steady and press `s` + Enter — the script averages the last ~30 frames and prints `CAM_GRIPPER_OFFSET_X` / `CAM_GRIPPER_OFFSET_Y`. Paste them into [modes/ball_follow.py](modes/ball_follow.py).
5. Press `h` + Enter to re-enable torque, `r` to release again, `q` to quit (re-engages torque first as a safety).

### Running the Demo

```bash
./start.sh --mode ball                # via launcher
python main.py --mode ball            # directly
python main.py --mode ball --headless # no preview window (SSH-friendly)
```

The arm homes, then the ball-follow mode takes over. Drop the ball anywhere within the scan envelope — the arm will find it, approach, and grab. Open the gripper by hand (or via the `open_gripper` /IOTCONNECT command) to return to `IDLE` and re-arm the cycle.

### Key Tuning Constants — [modes/ball_follow.py](modes/ball_follow.py)

All knobs live at the top of [modes/ball_follow.py](modes/ball_follow.py). The most important ones:

| Constant                                       | What it controls                                                                          |
|------------------------------------------------|-------------------------------------------------------------------------------------------|
| `PAN_GAIN`, `TILT_GAIN`                        | Servo units commanded per pixel of error. `TILT_GAIN` is higher because `wrist_flex` fights gravity at extended poses. |
| `PAN_DIR`, `TILT_DIR`, `TILT_ELBOW_DIR`        | Sign flips. Determined by live test — flip from `+1` to `-1` if the arm moves away from the ball instead of toward it. |
| `MIN_TRIM_STEP`                                | Floor on any non-zero wrist-flex P-controller command. Hiwonder bus servos silently ignore commands below ~5 units due to static friction; the floor prevents the controller stalling just outside the deadband. |
| `MIN_TRIM_STEP_PAN`                            | Same idea for `shoulder_pan` — set higher (default 18) because the pan axis carries the entire forearm + wrist + camera, so its static-friction floor is roughly 2× the wrist's. |
| `APPROACH_STEP`                                | Per-frame `shoulder_lift` step during descent toward the ball. Default 15. Below ~12 the lift servo can't break friction at extended poses (`shoulder_lift` > 600) and the descent silently stalls. |
| `MAX_STEP`                                     | Hard cap on any single-frame servo delta — keeps a large pixel error from snapping the arm. |
| `MOVE_DURATION_MS`                             | How long each per-frame move takes. Too short and small commands get ignored; too long and the loop rate drops. |
| `CENTER_DEADBAND_PX`                           | Pixel error inside which the controller stops trimming. Must be ≥ `MIN_TRIM_STEP_PAN × pixels-per-servo-unit` (~7 px/unit) or a single floored pan command will fling the ball clear past the deadband and the controller will oscillate. |
| `APPROACH_DEADBAND_PX`                         | Looser threshold — once inside this, the arm is allowed to descend toward the ball even while still fine-centering. |
| `TARGET_RADIUS_PX`, `RADIUS_TOLERANCE`         | Apparent ball radius (in pixels) that means "close enough to grab". Tune for your ball + grab height. |
| `CAM_GRIPPER_OFFSET_X / _Y`                    | Optional aim-point shift in pixels from the geometric image center, measured by [calibrate_cam_offset.py](calibrate_cam_offset.py). Default `0`/`0` — leave at zero unless you observe a systematic miss. |
| `LIFT_MAX`, `ELBOW_MAX`                        | Hard safety ceilings during approach so a never-satisfied radius check can't drive the gripper into the table. |
| `NO_BALL_GRACE_FRAMES`                         | How many consecutive lost-ball frames before falling back to `SCANNING`. Increase if the ball flickers in and out at the frame edge. |
| `SCAN_POSES`                                   | The poses cycled through while searching. Captured with [teach_pose.py](teach_pose.py). |

> **Warning:** `PAN_GAIN`, `TILT_GAIN`, `APPROACH_STEP`, and `MAX_STEP` all scale implicitly with the camera/loop frame rate — if you change camera resolution, drop the preview, or otherwise change fps, expect to re-tune them.

### /IOTCONNECT Telemetry

Every telemetry payload carries a top-level **`state`** field identifying the active mode:

- Ball mode publishes its state-machine value — `IDLE`, `SCANNING`, `TRACKING`, `PREDICTING`, `GRABBING`, or `HOLDING`.
- ASL mode publishes the fixed label `ASL-Gesture` so the dashboard can tell which demo is running even when no gesture is currently being acted on.

Ball mode additionally augments each payload with a `ballTrack` block:

- `state` — same state-machine value as the top-level field (also mirrored here for convenience).
- `ball_x`, `ball_y`, `ball_r` — last detected ball pixel position and radius (or `0` if not seen).
- `pan_err`, `tilt_err`, `radius_err` — current pixel/radius error from the aim point.
- `velocity_x`, `velocity_y` — recent ball motion in px/frame.
- `d_pan`, `d_tilt`, `d_lift`, `d_elbow` — last commanded servo deltas.
- `no_ball_frames`, `pred_frames_left`, `is_prediction` — detection-loss / extrapolation bookkeeping.

Publishing cadence is ~2 s (see `TELEMETRY_INTERVAL_S` in [modes/ball_follow.py](modes/ball_follow.py) and [modes/asl.py](modes/asl.py)). If `state` or `ballTrack.*` doesn't appear on your /IOTCONNECT dashboard, verify the fields are declared on the device's template — the broker drops undeclared attributes silently.

## Troubleshooting

- **XArm Connection Issues**: Ensure XArm 1S is connected to TRIA board's USB ports and powered on
- **HIDAPI Issues**: The xarm library uses hidapi for USB communication - ensure proper USB device permissions
- **Camera Not Detected**: Verify camera is connected to TRIA board and accessible
- **Model Loading Errors**: Ensure model files are downloaded and accessible in the `model/` directory
- **/IOTCONNECT Connection**: Check ethernet connectivity and device onboarding status
- **Permission Issues**: Run applications with appropriate permissions for USB/serial access

### Ball-Follow Mode

- **Gripper closes on empty space next to the ball**: the camera is far enough off-axis from the gripper that aiming at the geometric image center is no longer good enough. Run [calibrate_cam_offset.py](calibrate_cam_offset.py) and paste the resulting `CAM_GRIPPER_OFFSET_X / _Y` into [modes/ball_follow.py](modes/ball_follow.py). Re-run after any change to the camera mount, gripper, or wrist plate.
- **Arm finds the ball but "just sits there" shaking slightly**: classic bus-servo static-friction stall. The P-controller is commanding a delta below ~5 servo units that the motor physically ignores. Make sure `MIN_TRIM_STEP` is at least `8` and `MIN_TRIM_STEP_PAN` is at least `18` in [modes/ball_follow.py](modes/ball_follow.py), and don't try to fix this by lowering `CENTER_DEADBAND_PX` — that just moves the dead-spot inward.
- **Arm centers the ball but never descends toward it**: same friction-floor failure on the lift axis. Raise `APPROACH_STEP` (default 15) — at extended poses (`shoulder_lift` > 600) the lift servo needs ≥12-unit commands to actually move, even though shorter commands get accepted by the bus.
- **Arm bounces between tracking and scan-move every time the ball flickers**: the detector is losing the ball briefly (clipping at frame edges, HSV margin too tight). Raise `NO_BALL_GRACE_FRAMES` so the controller holds pose through short detection drop-outs, or re-run [ball_calibrate.py](ball_calibrate.py) under the actual lighting.
- **Arm drives _away_ from the ball instead of toward it**: a sign is wrong. Flip `PAN_DIR`, `TILT_DIR`, or `TILT_ELBOW_DIR` in [modes/ball_follow.py](modes/ball_follow.py) (whichever axis is wrong). These are mount-dependent — re-mounting the arm on a different bracket can flip them.
- **Arm reaches the right pose but the gripper closes before reaching the ball**: `TARGET_RADIUS_PX` is too low for your ball / grab distance. Watch the live OSD's reported radius at the moment you'd want it to fire and set `TARGET_RADIUS_PX` to that value.
- **`ball_color.json` not found**: run [ball_calibrate.py](ball_calibrate.py) first — both ball-follow and the offset calibrator depend on it.

### Device Detection

Check XArm detection:
```bash
lsusb | grep 0483:5750  # Should show XArm device
```

Check camera detection:
```bash
ls /dev/video*  # Should show available camera devices
```


## Important Notes for TRIA + /IOTCONNECT Operation

### TRIA Vision AI Kit 6490 Best Practices
- Always enable the XArm robot before sending movement commands from the TRIA board
- Use `arm.query("HOME")` to safely return to home position before shutdown
- Ensure stable ethernet connection for reliable /IOTCONNECT cloud communication
- Monitor TRIA board temperature during extended AI inference sessions

### /IOTCONNECT Integration Notes
- Device telemetry streams continuously when /IOTCONNECT connection is active
- Remote commands are queued and executed asynchronously on the TRIA board
- Local telemetry logging provides backup when cloud connectivity is interrupted
- Use /IOTCONNECT dashboard to monitor TRIA board performance and gesture recognition accuracy

### ASL Gesture Recognition on TRIA
- Camera and XArm USB connections must be maintained during operation
- Gesture recognition runs locally on TRIA board for low-latency robotic control
- Confidence scores and hand landmarks are transmitted to /IOTCONNECT for analysis
- Model inference optimized for TRIA's Qualcomm QCS6490 AI capabilities
- Keep safety zone clear before moving.

## References & Documentation

### TRIA Vision AI Kit 6490
- [TRIA Vision AI Kit 6490 Setup Guide](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/tree/main/tria-vision-ai-kit-6490) - Complete setup and configuration guide
- [TRIA Vision AI-KIT 6490 Product Page](https://www.newark.com/avnet/sm2-sk-qcs6490-ep6-kit001/dev-kit-64bit-arm-cortex-a55-a78/dp/51AM9843) - Hardware specifications and purchase information
- [TRIA Startup Guide](https://avnet.com/wcm/connect/137a97f1-eb6e-48ba-89a4-40b024558593/Vision+AI-KIT+6490+Startup+Guide+v1.3.pdf?MOD=AJPERES&attachment=true&id=1761931434976) - Hardware setup and cable connections

### /IOTCONNECT Platform
- [/IOTCONNECT SDK](https://github.com/avnet-iotconnect/avnet-iotconnect-python-sdk) - /IOTCONNECT Python SDK for cloud connectivity
- [/IOTCONNECT Device Onboarding](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/blob/main/common/general-guides/UI-ONBOARD.md) - Step-by-step device registration guide
- [/IOTCONNECT Overview](https://www.iotconnect.io/) - Enterprise IoT platform information

### Robotics & AI Components
- [xArm Python Library](https://github.com/xArm-Developer/xArm-Python-SDK) - Official Python SDK for Hiwonder XArm robotic arms
- [ASL MediaPipe PointNet](https://github.com/AlbertaBeef/asl_mediapipe_pointnet) - ASL gesture recognition using MediaPipe and PointNet neural network