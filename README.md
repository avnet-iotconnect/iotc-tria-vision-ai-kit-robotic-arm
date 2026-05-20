# TRIA Vision AI Kit 6490 + /IOTCONNECT XArm Vision Demos

This project showcases the **TRIA Vision AI Kit 6490** running
**/IOTCONNECT** integration with the Hiwonder XArm 1S robotic arm.
It ships two interchangeable vision modes selectable at launch with
`--mode`:

- **`asl`** (default) — American Sign Language gesture control.
  Operator drives the arm by signing letters in front of a webcam.
- **`ball`** — Autonomous eye-in-hand visual servoing. The
  wrist-mounted camera detects a colored ball, and the arm
  pans/tilts/advances on its own to center, approach, and grab it.

Both modes run on the TRIA board, stream live telemetry to
/IOTCONNECT, and accept remote commands from the cloud. The demo
also supports **live WebRTC video streaming** via AWS Kinesis Video
Streams (KVS), letting you watch the arm's wrist camera feed in real
time directly from the /IOTCONNECT portal. Streaming is opt-in and
enabled with the `--webrtc` flag so it never interferes with the
core vision and arm-control functionality.

> [!TIP]
> It is strongly recommended that users complete the [/IOTCONNECT basic quickstart guide for the TRIA VISION AI-KIT 6490](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/tree/main/tria-vision-ai-kit-6490) 
> so they can familiarize themselves with the hardware and the /IOTCONNECT UI before proceeding on to this project.

## Hardware Requirements

### Included with **[TRIA Vision AI-KIT 6490](https://www.newark.com/avnet/sm2-sk-qcs6490-ep6-kit001/dev-kit-64bit-arm-cortex-a55-a78/dp/51AM9843)**
- USB-C Cable for flashing and USB-ADB debug (included with kit)
- USB-C 12VDC Power Supply and Cable (included with kit)

### Purchased Separately
- **[HiWonder xArm1S](https://www.amazon.com/LewanSoul-Programmable-Feedback-Parameter-Programming/dp/B0CHY63V9P?th=1)**
- Ethernet Cable 
- USB camera for hand tracking (ASL mode) and eye-in-hand visual
  servoing (ball mode). For the ball mode the camera is mounted on
  the **wrist roll** servo with zip ties so it pitches with the
  gripper

> [!TIP]
> If possible, it is recommended to use a bare USB-camera PCB module mounted directly
> behind the gripper jaws, giving the cleanest line of sight to
> whatever the gripper is about to grab and removing the parallax that
> makes the camera-gripper offset calibration necessary.

- USB Mouse and Keyboard
- HDMI Monitor with **active** mini-DP to HDMI adapter

> [!IMPORTANT]
> The mini-DP to HDMI adapter must be **active** so to avoid purchasing the wrong product it is recommended to use the adapter used and tested by Avnet's engineer available [here](https://www.amazon.com/Cable-Matters-DisplayPort-Supporting-Technology/dp/B00PJ3LSIG/ref=sr_1_1?crid=XR4HA3U2IVD0&dib=eyJ2IjoiMSJ9.7o239haE8CcYdAqsOPF7Se6OXe8Sz47i-Az7Mq9_PvLySbMg4xxB8QbnT7rNODDTxSh882r-DD24OPLilxONY3rqmtq2d-y9-PdgAE7xHVKKFR7sSypCPC5w6yW8QYkxKJag31Qy-DlnbIz1F9XIBGWrG6Ric9NSsSSTfHpZG58gk_bvzo6qGpsQa11HI9C3rp4MSgjK6X5zBcp_98AzK_elv_1tTuomClMsDK_tZuw.c21P4pWnM5M33qDFmO5u0CjFbJWeyxQZ93-Fv3nExKw&dib_tag=se&keywords=cable+matters+mini+dp+to+hdmi&qid=1762415607&sprefix=%2Caps%2C84&sr=8-1).


## Initial Setup

### Hiwonder XArm 1S Robotic Arm Hardware Setup

> [!NOTE]
> It is assumed that users have purchased the pre-assembled version of the robotic arm. If not, users should follow 
> Hiwonder's assembly instructions first and then return here.

1. Attach the included suction-cup feet to the base of the arm using the provided hardware as shown below

<img src="./media/suction-cup-hardware.png">

<img src="./media/suction-cup-assembled.png">

2. Using a strong adhesive (our engineer used epoxy), attach your USB camera to the top of the "hand" rotational servo 
motor.

> [!TIP]
> Prop up the arm with a small box (see image below) and use masking tape to hold the camera in position while the adhesive 
> sets. You can also use a "helping hand" soldering arm to hold the USB cable of the camera for added stability.

<img src="./media/supporting-box.png">

> [!IMPORTANT]
> Ensure not to allow any adhesive to drip down into any joints or moving parts.

3. After the adhesive has cured, the final mounting of the camera should look like this:

<img src="./media/final-camera-mounted.png">

### TRIA Vision AI Kit 6490 Hardware Setup
   - Connect 12VDC USB-C power supply to the USB-C "DC PWR" connector
   - Connect ethernet cable to the board's ethernet port
   - Connect USB mouse/keyboard to USB-A ports
   - Connect second USB-C cable for USB-ADB communication
   - Connect Logitech Camera for hand tracking

> [!NOTE]
> These steps only need to be performed once. After completing them,
> see [Starting the Demo](#starting-the-demo) for the commands to run
> after every reboot.

3. **Power On**: Hold S1 button for 2-3 seconds until red LED turns
   off

4. **SSH into the board**:
   ```bash
   ssh root@<board-ip>
   ```
   Login with password `oelinux123`.

5. **Clone the repository**:
   ```bash
   git clone https://github.com/avnet-iotconnect/iotc-tria-vision-ai-kit-robotic-arm.git
   cd iotc-tria-vision-ai-kit-robotic-arm
   ```

6. **Install Miniforge (conda)**:

   Download and run the installer:
   ```bash
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```

   Respond to each prompt as follows:
   - **License agreement**: Press **ENTER** to scroll through, then type `yes` and **ENTER** to accept.
   - **Installation location**: Press **ENTER** to confirm the default path.
   - **Shell initialization**: Type `yes` and **ENTER** when asked `Proceed with initialization?`

   Reload your shell so `conda` becomes available:
   ```bash
   source ~/.bashrc
   ```

   The board's login shell does not source `.bashrc` automatically on reboot. Run this once to fix that permanently:
   ```bash
   echo '. ~/.bashrc' >> ~/.profile
   ```

7. **Create the conda environment and install dependencies**:
   ```bash
   conda create -y -n iotc-tria-xarm python=3.11 pip
   conda activate iotc-tria-xarm
   conda install opencv -c conda-forge
   python -m pip install -r requirements.txt
   ```

8. **Download the AI model**:
   ```bash
   sh model/get_model.sh
   ```

After completing setup, continue with [/IOTCONNECT Device Onboarding](#iotconnect-device-onboarding) below before running the demo.


## Starting the Demo

After the board has been set up and rebooted, run these commands each
time to start the demo:

1. **SSH into the board**:
   ```bash
   ssh root@<board-ip>
   ```
   Login with password `oelinux123`.

2. **Activate the environment and launch**:
   ```bash
   source ~/.bashrc
   conda activate iotc-tria-xarm
   cd ~/iotc-tria-vision-ai-kit-robotic-arm
   python3 main.py --webrtc
   ```

   > [!NOTE]
   > The `source ~/.bashrc` line is needed because the board's login
   > shell does not load conda automatically on reboot. It is safe to
   > run every time.

   > [!NOTE]
   > If running over SSH with no monitor connected, add `--headless` to
   > suppress the camera preview window:
   > ```bash
   > python3 main.py --webrtc --headless
   > ```

   > [!NOTE]
   > `--webrtc` enables live video streaming to the /IOTCONNECT portal.
   > To run ASL gesture-control mode instead of the default ball-follow
   > mode, add `--mode asl`.

## /IOTCONNECT Device Onboarding

> [!IMPORTANT]
> If you intend to run this demo with the `--webrtc` flag to enable
> live video streaming, the device **must** be created in /IOTCONNECT
> using the `robarmwebrtc` template provided in this repository
> ([robarmwebrtc-template.json](robarmwebrtc-template.json)). During the device
> creation process you will be prompted to select a **Stream
> Resource** — choose **WebRTC**. The /IOTCONNECT backend provisions
> a KVS WebRTC signaling channel at device creation time and this
> choice cannot be changed afterwards. If your device was already
> created with a different template or with the wrong stream resource,
> you must create a new device using the `robarmwebrtc` template and
> select WebRTC at that point. Devices created without the
> `robarmwebrtc` template can still run the demo without the
> `--webrtc` flag and will not be affected.

Follow [this guide](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/blob/main/common/general-guides/UI-ONBOARD.md)
to onboard your TRIA Vision AI Kit 6490 to /IOTCONNECT.

> [!CAUTION]
> In **Step 14** of the onboarding guide, you must run the quickstart script from inside the project directory so that the device certificate and config files are placed where this demo expects them. Make sure you are in the project directory first, then use this command instead of the one shown in the guide:
> ```bash
> wget https://raw.githubusercontent.com/avnet-iotconnect/iotc-python-lite-sdk-demos/refs/heads/main/common/scripts/quickstart.sh && bash ./quickstart.sh
> ```

> [!NOTE]
> The starter `app.py` downloaded by the quickstart script can be disregarded — this demo uses `main.py` instead.


## Supported Gestures

**Left Hand (Arm Movement)**:
- A: Advance, B: Back-up, L: Left, R: Right, U: Up, Y: Down, H: Home

**Right Hand (Gripper Control)**:
- A: Close Gripper, B: Open Gripper

## Remote Command Control via /IOTCONNECT

Control your XArm robot remotely through /IOTCONNECT cloud commands:
- **Movement Commands**: `advance`, `backup`, `left`, `right`, `up`,
  `down` for arm positioning
- **Gripper Control**: `open_gripper`, `close_gripper` for object
  manipulation
- **System Commands**: `home` for safe return to center position
- **Command Acknowledgment**: Real-time feedback and execution
  confirmation

## Remote Command Execution

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

> [!NOTE]
> This demo is intended to showcase the potential capabilities of the TRIA VISION AI-KIT in conjunction with robotics and 
> /IOTCONNECT. The consistency of success with the Ball Follow demo is reliant on a multitude of factors relating to specific 
> camera angle and positioning on the arm, lighting, calibration, ball position, and surface/room colors. If a user's 
> setup is unable to reliably pick up the target balls, they are encouraged to tweak the code and calibration sequences 
> until they are successful.

The `ball` mode turns the XArm into an autonomous pick-and-place
demo. The wrist-mounted camera looks for a single colored ball, the
controller centers it in the frame, advances until the ball fills the
expected radius, then closes the gripper, lifts, and returns to home.
No operator input is required after launch.

### How It Works

The controller aims for the **geometric center** of the camera frame
and trusts the gripper to be close enough to that aim point. (An
optional camera-gripper offset exists in
[modes/ball_follow.py](modes/ball_follow.py) —
`CAM_GRIPPER_OFFSET_X / _Y` — for hardware where the wrist camera is
mounted noticeably off-axis from the gripper fingers. Both default to
`0` for the current build; only set them if you observe the gripper
consistently closing next to the ball rather than on it.)

The mode runs as a state machine driven by per-frame HSV detection:

| State        | What it does                                                                                       | Exit                                                              |
|--------------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `IDLE`       | Initial state at launch. Falls through immediately.                                                | → `SCANNING`                                                      |
| `SCANNING`   | Cycles through `SCAN_POSES` (center / left / right) so the camera sweeps the workspace.            | Ball detected → `TRACKING`                                        |
| `TRACKING`   | P-controller drives `shoulder_pan` + `wrist_flex` (with elbow assist) to center the ball pixel.    | Ball centered AND radius below target → advance; centered + radius OK → `GRABBING` |
| `GRABBING`   | Closes the gripper, watches the actual position to detect a stall against the ball, then lifts.    | → `HOLDING`                                                       |
| `HOLDING`    | Returns to home (keeping the gripper closed) and waits for the operator to manually open it.       | Gripper opened by user → `IDLE`                                   |

A `NO_BALL_GRACE_FRAMES` window (~5 s at 6 fps) lets the arm hold its
current pose during brief detection drop-outs from clipping or HSV
flicker, instead of bouncing back into a scan move every time the
ball flickers off for a frame.

### Calibration Workflow

The ball mode needs three required pieces of calibration data. Capture them in this order — each step writes a JSON
file (or constants you paste into [modes/ball_follow.py](modes/ball_follow.py))
that the next may depend on.

All calibration scripts release **all six servo torques** so you can
free-pose the entire arm — useful for aiming the wrist camera before
sampling. **Always physically support the arm before pressing
Enter** — the arm will fall under gravity the instant torque is
released.

#### 1. Ball Color — [ball_calibrate.py](ball_calibrate.py)

Captures the HSV thresholds used for ball segmentation.

> [!IMPORTANT]
> This script opens an interactive preview window that requires a
> physical display, mouse, and keyboard connected directly to the
> board. It cannot be run over SSH. Connect an HDMI monitor, USB
> mouse, and USB keyboard before running.

```bash
bash ./calibrate.sh
```

Workflow:
1. **Support the arm**, press Enter to release all torque so you can
   aim the wrist camera at where the ball will sit.
2. Click the ball in the live preview. Each click samples a 7×7 HSV
   patch and widens the mask range. The live overlay shows what the
   ball-follow detector will see.
3. Press `h` to re-engage torque at the current pose, `w` to release
   again if you need to re-aim.
4. Press `s` to save → writes `ball_color.json`. Use `r` to reset
   samples, `q`/`ESC` to quit without saving. The script re-engages
   full torque at the current pose on exit.

#### 2. Scan Poses — [teach_pose.py](teach_pose.py)

Captures the arm poses cycled through during `SCANNING`. Torque is
dropped so you can pose the arm by hand.

```bash
bash ./teach.sh
```

Workflow:
1. **Support the arm with your hand** — torque is about to drop and
   the arm will fall under gravity.
2. Press Enter to release torque.
3. Pose the arm at each of the six positions as prompted. You teach
   two rows — **near** (close to the base) and **far** (maximum
   comfortable reach) — at three pan positions each (left, center,
   right). The script automatically interpolates a third **mid**
   row halfway between them. The resulting scan cycles through
   three 180° arcs:

   - **Near arc** (near-left → near-center → near-right): arm
     angled steeply downward, camera covering the table area
     directly below and close to the base.
   - **Mid arc** (mid-left → mid-center → mid-right):
     auto-calculated from the near and far poses — no teaching
     required.
   - **Far arc** (far-left → far-center → far-right): arm extended
     outward at a shallower angle, camera covering the table at
     maximum reach.

   > [!IMPORTANT]
   > The scan poses are **search positions only**, not grab positions.
   > For each pose, orient the wrist camera **downward so it has a
   > clear view of the table surface** in that region of the
   > workspace.
   >
   > Keep the gripper well above the table at every scan pose — once
   > a ball is detected and centered, the grab snap will lower the
   > arm to ball height automatically. If the gripper is already at
   > table level during a scan pose, the arm will crash the moment
   > it moves to that pose.
   >
   > The near/far and left/right extremes should mark the outer
   > boundary of the search area, not the physical grab limit.

4. Press `s` + Enter to snapshot. The script prints the servo values
   and tells you which position to move to next.
5. Press `h` + Enter to re-enable torque while repositioning between
   poses, then `r` + Enter to release torque again before posing.
6. After all six poses are saved the script automatically calculates
   the mid-row poses, updates `SCAN_POSES` in
   [modes/ball_follow.py](modes/ball_follow.py), and prompts you
   to quit.

#### 3. Camera-Gripper Offset — [calibrate_cam_offset.py](calibrate_cam_offset.py)

Because the camera cannot see the ball when the wrist is in grab
position, this calibration uses a two-phase process. It measures three
values and writes them all automatically into
[modes/ball_follow.py](modes/ball_follow.py):
- `CAM_GRIPPER_OFFSET_X / Y` — where the ball appears in the tilted
  camera frame when the gripper is correctly positioned above it
- `WRIST_VIEW_OFFSET` — wrist flex servo units between grab and view
  position; subtracted at grab time so the gripper lands on the ball

```bash
bash ./calibrate_offset.sh
```

Workflow:
1. Place the ball on the table. **Support the arm**, press Enter to
   release all torque.

2. **Phase 1 — Record grab position**: Pose the gripper directly over
   the ball at grab height (gripper gently touching the table surface).
   Press `g` + Enter. The script records all servo positions and
   **locks the shoulder and base rotation servos** so they cannot
   drift while you adjust in Phase 2.

3. **Phase 2 — Record view position**: With the shoulder and base now
   locked, raise the elbow slightly and tilt the wrist down until the
   ball appears in the live camera view. Hold steady and press
   `s` + Enter — the script averages the last ~30 frames and writes
   `CAM_GRIPPER_OFFSET_X/Y`, `WRIST_VIEW_OFFSET`, and
   `ELBOW_VIEW_OFFSET` into [modes/ball_follow.py](modes/ball_follow.py)
   automatically.

4. Press `h` + Enter to re-enable torque, `r` to release again, `q`
   to quit (re-engages torque first as a safety).

### Running the Demo

```bash
python3 main.py --mode ball --webrtc
```

> [!NOTE]
> Add `--headless` if running over SSH with no monitor connected.

The arm homes, then the ball-follow mode takes over. Drop the ball
anywhere within the scan envelope — the arm will find it, approach,
and grab. Open the gripper by hand (or via the `open_gripper`
/IOTCONNECT command) to return to `IDLE` and re-arm the cycle.

### Key Tuning Constants — [modes/ball_follow.py](modes/ball_follow.py)

All knobs live at the top of [modes/ball_follow.py](modes/ball_follow.py).
The most important ones:

| Constant                                       | What it controls                                                                          |
|------------------------------------------------|-------------------------------------------------------------------------------------------|
| `PAN_GAIN`, `TILT_GAIN`                        | Servo units commanded per pixel of error. `TILT_GAIN` is higher because `wrist_flex` fights gravity at extended poses. |
| `PAN_DIR`, `TILT_DIR`, `TILT_ELBOW_DIR`        | Sign flips. Determined by live test — flip from `+1` to `-1` if the arm moves away from the ball instead of toward it. |
| `MIN_TRIM_STEP`                                | Floor on any non-zero wrist-flex P-controller command. Hiwonder bus servos silently ignore commands below ~5 units due to static friction; the floor prevents the controller stalling just outside the deadband. |
| `MIN_TRIM_STEP_PAN`                            | Same idea for `shoulder_pan` — set higher (default 18) because the pan axis carries the entire forearm + wrist + camera, so its static-friction floor is roughly 2× the wrist's. |
| `APPROACH_STEP`                                | Per-frame `shoulder_lift` step during descent toward the ball. **Set to `0` (disabled) in this upright/table-mount build** — the grab snap handles height directly. If re-enabled, values below ~12 cannot break static friction at extended poses and the descent silently stalls. |
| `MAX_STEP`                                     | Hard cap on any single-frame servo delta — keeps a large pixel error from snapping the arm. |
| `MOVE_DURATION_MS`                             | How long each per-frame move takes. Too short and small commands get ignored; too long and the loop rate drops. |
| `CENTER_DEADBAND_PX`                           | Pixel error inside which the controller stops trimming. Must be ≥ `MIN_TRIM_STEP_PAN × pixels-per-servo-unit` (~7 px/unit) or a single floored pan command will fling the ball clear past the deadband and the controller will oscillate. |
| `APPROACH_DEADBAND_PX`                         | Looser threshold — once inside this, the arm is allowed to descend toward the ball even while still fine-centering. |
| `TARGET_RADIUS_PX`, `RADIUS_TOLERANCE`         | Apparent ball radius (in pixels) that means "close enough to grab". Tune for your ball + grab height. |
| `CAM_GRIPPER_OFFSET_X / _Y`                    | Optional aim-point shift in pixels from the geometric image center, measured by [calibrate_cam_offset.py](calibrate_cam_offset.py). Default `0`/`0` — leave at zero unless you observe a systematic miss. |
| `LIFT_MAX`, `ELBOW_MAX`                        | Hard safety ceilings during approach so a never-satisfied radius check can't drive the gripper into the table. |
| `NO_BALL_GRACE_FRAMES`                         | How many consecutive lost-ball frames before falling back to `SCANNING`. Increase if the ball flickers in and out at the frame edge. |
| `SCAN_POSES`                                   | The poses cycled through while searching. Captured with [teach_pose.py](teach_pose.py). |

> **Warning:** `PAN_GAIN`, `TILT_GAIN`, `APPROACH_STEP`, and
> `MAX_STEP` all scale implicitly with the camera/loop frame rate —
> if you change camera resolution, drop the preview, or otherwise
> change fps, expect to re-tune them.

### /IOTCONNECT Telemetry

Every telemetry payload carries a top-level **`state`** field
identifying the active mode:

- Ball mode publishes its state-machine value — `IDLE`, `SCANNING`,
  `TRACKING`, `PREDICTING`, `GRABBING`, or `HOLDING`.
- ASL mode publishes the fixed label `ASL-Gesture` so the dashboard
  can tell which demo is running even when no gesture is currently
  being acted on.

Ball mode additionally augments each payload with a `ballTrack`
block:

- `state` — same state-machine value as the top-level field (also
  mirrored here for convenience).
- `ball_x`, `ball_y`, `ball_r` — last detected ball pixel position
  and radius (or `0` if not seen).
- `pan_err`, `tilt_err`, `radius_err` — current pixel/radius error
  from the aim point.
- `velocity_x`, `velocity_y` — recent ball motion in px/frame.
- `d_pan`, `d_tilt`, `d_lift`, `d_elbow` — last commanded servo
  deltas.
- `no_ball_frames`, `pred_frames_left`, `is_prediction` —
  detection-loss / extrapolation bookkeeping.

Publishing cadence is ~2 s (see `TELEMETRY_INTERVAL_S` in
[modes/ball_follow.py](modes/ball_follow.py) and
[modes/asl.py](modes/asl.py)). If `state` or `ballTrack.*` doesn't
appear on your /IOTCONNECT dashboard, verify the fields are declared
on the device's template — the broker drops undeclared attributes
silently.

## References & Documentation

### TRIA Vision AI Kit 6490
- [TRIA Vision AI Kit 6490 Setup Guide](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/tree/main/tria-vision-ai-kit-6490)
  — Complete setup and configuration guide
- [TRIA Vision AI-KIT 6490 Product Page](https://www.newark.com/avnet/sm2-sk-qcs6490-ep6-kit001/dev-kit-64bit-arm-cortex-a55-a78/dp/51AM9843)
  — Hardware specifications and purchase information
- [TRIA Startup Guide](https://avnet.com/wcm/connect/137a97f1-eb6e-48ba-89a4-40b024558593/Vision+AI-KIT+6490+Startup+Guide+v1.3.pdf?MOD=AJPERES&attachment=true&id=1761931434976)
  — Hardware setup and cable connections

### /IOTCONNECT Platform
- [/IOTCONNECT SDK](https://github.com/avnet-iotconnect/avnet-iotconnect-python-sdk)
  — /IOTCONNECT Python SDK for cloud connectivity
- [/IOTCONNECT Device Onboarding](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/blob/main/common/general-guides/UI-ONBOARD.md)
  — Step-by-step device registration guide
- [/IOTCONNECT Overview](https://www.iotconnect.io/)
  — Enterprise IoT platform information

### Robotics & AI Components
- [xArm Python Library](https://github.com/xArm-Developer/xArm-Python-SDK)
  — Official Python SDK for Hiwonder XArm robotic arms
- [ASL MediaPipe PointNet](https://github.com/AlbertaBeef/asl_mediapipe_pointnet)
  — ASL gesture recognition using MediaPipe and PointNet neural
  network
