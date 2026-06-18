"""Ball-follow mode (eye-in-hand visual servoing).

Reads HSV thresholds from ball_color.json, segments the ball each frame, and
runs a small proportional controller to:
  1) center the ball in the wrist-camera frame (shoulder_pan + wrist_flex)
  2) advance until the ball's apparent radius hits a target (shoulder_lift)
  3) close the gripper, lift, return to home

State machine: IDLE -> TRACKING -> GRABBING -> AFTER_GRAB -> IDLE.
"""

import json
import os
import time
from collections import deque

import cv2
import numpy as np
from xarm import Servo

from .base import Mode

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ball_color.json")

# --- tuning knobs ---
UPDATE_EVERY = 1           # one command per frame; each blocks until done so further throttling pointless
PAN_GAIN = 0.04            # servo units per pixel of horizontal error
TILT_GAIN = 0.10           # servo units per pixel of vertical error — higher than PAN because wrist_flex is fighting gravity at extended poses; raw step grows fast with error so MAX_STEP gets reached and motion is visible
# Sign of correction. +1 if "servo value goes up" corresponds to "camera view
# moves in the +x/+y direction of the pixel frame". Flip either to -1 if the
# arm moves AWAY from the ball. Determined by live test, not theory.
PAN_DIR = -1               # verify with live test: flip to 1 if arm pans away from ball instead of toward it
TILT_DIR = -1
APPROACH_STEP = 0          # DISABLED: increasing shoulder_lift sweeps camera PAST ball (r shrinks as lift rises = arm arcs away)
ELBOW_REACH_RATIO = 0.6    # elbow contribution per lift unit during approach (inactive while APPROACH_STEP=0)
TILT_ELBOW_RATIO = 0.0     # disabled 2026-04-19 — telemetry showed elbow_flex draining to floor (clamp=0) because tilt's negative contribution outweighed approach's positive one over many frames; once elbow saturates the assist is wasted anyway. Wrist alone handles tilt with the wider CENTER_DEADBAND_PX
TILT_ELBOW_DIR = 1         # flip to -1 if elbow moves wrong way for tilt
MAX_STEP = 25              # cap on any single-update servo delta — raised so larger errors actually translate to a visible move; the bus servo has plenty of torque headroom for 25-unit steps
MIN_TRIM_STEP = 3
MIN_TRIM_STEP_PAN = 18     # shoulder_pan carries the whole forearm + wrist + camera, so its static friction is much higher than wrist_flex — 8-unit commands stall in extended poses (shoulder_lift>600). Verified 2026-04-19 with telemetry showing pan stuck at 552-553 while controller commanded +8 every frame.
MOVE_DURATION_MS = 220     # per-step move duration; raised so small servo commands actually move (under 150ms many bus servos ignore sub-5-unit deltas)
CENTER_DEADBAND_PX = 60
APPROACH_DEADBAND_PX = 60
TARGET_RADIUS_PX = 112            # ball consistently appears at r=107-112 at all arm positions (camera stays ~constant distance); grab fires when centered
RADIUS_TOLERANCE = 35             # px: wide enough to fire grab when APPROACH_STEP=0 can't adjust distance
MIN_CONTOUR_AREA = 200     # ignore tiny mask blobs (noise)
MIN_BALL_RADIUS = 20       # reject blobs smaller than this — glints and speck false positives
MAX_BALL_RADIUS = 150      # reject blobs larger than this — ceiling/wall/floor false positives
MIN_FILL_RATIO = 0.60      # fraction of enclosing circle area that must be masked; 0.60 suits orange ball (highlights stay orange-tinted, mask is more complete than white)
# Camera-to-gripper offset in image pixels. Because the camera cannot see the
# ball when the wrist is in grab position, calibrate_cam_offset.py uses a
# two-phase process: Phase 1 records the grab wrist position with the gripper
# physically over the ball; Phase 2 tilts the wrist down until the ball is
# visible and snapshots the ball's pixel position. The pixel offset is where
# the ball appears in the tilted (view) frame when the gripper is correctly
# positioned above it. The controller aims at this pixel during tracking so
# that snapping the wrist back to grab position puts the gripper over the ball.
CAM_GRIPPER_OFFSET_X = 20   # set by calibrate_cam_offset.py — run after each physical setup change
CAM_GRIPPER_OFFSET_Y = 40  # interim: ball consistently at y≈160 vs center 240; run calibrate_cam_offset.py to set precisely
# Wrist and elbow offsets between grab position and view position. Because
# the camera cannot see the ball when the wrist is at grab angle, the arm
# raises the elbow slightly and tilts the wrist down during tracking so the
# camera can see the ball. These offsets are the servo-unit differences
# between those two states (view - grab). _do_grab() subtracts both before
# closing the gripper so the gripper lands on the ball. Set automatically
# by calibrate_cam_offset.py.
WRIST_VIEW_OFFSET = -159    # wrist_flex units: view = grab + this; set by calibrate_cam_offset.py
ELBOW_VIEW_OFFSET = -68    # elbow_flex units: view = grab + this; set by calibrate_cam_offset.py
GRAB_WRIST_TRIM = 15      # extra wrist_flex at grab snap; 0 = exact Phase1 grab wrist; set by calibrate_cam_offset.py 'w' command
# Per-zone grab-lift table: [ref_tracking_lift, grab_lift] pairs sorted by ref ascending.
# ref_tracking_lift — shoulder_lift the arm uses while tracking a ball in that zone (≈ scan row lift).
# grab_lift         — Phase 1 shoulder_lift (gripper at ball equator); set by calibrate_cam_offset.py.
# _do_grab linearly interpolates so the arm snaps to the right height regardless of ball distance.
# Run calibrate_cam_offset.py with ball near the base → updates near zone (ref 673).
# Run again with ball at arm's reach → updates far zone (ref 507). Mid-range interpolates.
GRAB_LIFT_TABLE = [
    [507, 490],   # far  zone — grab_lift raised: 360/430 over-extended (crashed under ball)
    [673, 551],   # near zone — ball close to base
]
# --- search envelope ---
PAN_MIN = 150              # shoulder_pan clamped to [PAN_MIN, PAN_MAX]; scan right limit is pan~129-145
PAN_MAX = 800
# Hard safety limits so an approach that never satisfies radius_ok (bad HSV, wrong-sized ball,
# mis-set TARGET_RADIUS_PX) can't drive the gripper into the table. Tune by teach-mode probing.
LIFT_MAX = 780             # approach ceiling; raised from 700 to allow arm to reach ball (scan poses at 670-673, grab needs ~730+)
LIFT_MIN = 100             # general servo lower bound
ELBOW_MAX = 985            # elbow_flex upper bound (scan poses reach 940-947)
ELBOW_MIN = 100            # general servo lower bound
SCAN_DWELL_S = 1.5         # pause at each pose after movement completes before advancing
SCAN_MOVE_MS = 7500        # duration for moves between scan poses
SCAN_ADVANCE_FRAMES = 2    # when actively scanning with no ball, advance after this many frames
NO_BALL_GRACE_FRAMES = 15  # when transitioning from tracking→scan (just lost a ball), hold this many frames before re-scanning
SCAN_CONFIRM_FRAMES = 1    # frames required before HALT fires; 1 = immediate for orange ball (tight HSV keeps FP rate low); raise only if false HALTs return
# --- prediction + telemetry ---
ENABLE_PREDICTION = False        # master switch for extrapolating ball motion when it's lost
POS_BUFFER_LEN = 5               # how many recent ball positions we keep for velocity est
MIN_VELOCITY_PX_PER_FRAME = 2.0  # velocity below this = don't bother predicting
MAX_PREDICT_FRAMES = 15          # cap predictions so the arm doesn't run away on a bad estimate
TELEMETRY_INTERVAL_S = 2.0       # how often we publish ball telemetry to IoTConnect
GRIPPER_CLOSE_TARGET = 490  # close until ball resists; 490 prevents over-closing on ping pong ball (previous 550 reached ~519 = mechanism limit at wrong grab height)
GRIPPER_STALL_SLACK = 20    # fires when actual < 470; ball contact at correct grab height expected ~400-450 (much less than 519 seen when grabbing empty air)
GRIPPER_RELAX_OFFSET = 3    # hold at stall_point + 3 — minimal squeeze for ping pong
GRIPPER_RELEASE_DELTA = 40  # if current pos < hold_target - this, user opened the gripper
WRIST_ROLL_HOME = 504        # wrist_roll value when gripper is level (from calibrated scan poses)
WRIST_ROLL_COMPENSATION = 0.0    # units of wrist_roll per unit of shoulder_pan offset; set by calibrate_cam_offset.py

# --- arm conventions (from main.execute_arm_action) ---
SERVO_GRIPPER = 1
SERVO_WRIST_ROLL = 2
SERVO_WRIST_FLEX = 3       # smaller = camera tilts up; larger = camera tilts down
SERVO_ELBOW_FLEX = 4
SERVO_SHOULDER_LIFT = 5    # smaller = lift up; larger = reach forward/down
SERVO_SHOULDER_PAN = 6     # smaller = pan left; larger = pan right (verify w/ live test)

# Scan poses — cycled through while no ball is seen. teach_pose.py captures
# 6 poses (near/far × left/center/right) and auto-interpolates a mid row,
# producing 3 arcs of 3 poses each that sweep the full table semicircle.
# THESE ARE PLACEHOLDER VALUES — run teach_pose.py before use.
# wrist_roll should be left at 500 (neutral) when teaching so the wrist_flex
# axis produces pure pitch; off-neutral roll decouples pixel dx/dy from the
# pan/flex servos and causes tracking to oscillate.
SCAN_POSES = [
    # near-left
    [[SERVO_SHOULDER_PAN, 888], [SERVO_SHOULDER_LIFT, 673], [SERVO_ELBOW_FLEX, 939], [SERVO_WRIST_FLEX, 104], [SERVO_WRIST_ROLL, 503], [SERVO_GRIPPER, 365]],
    # near-center
    [[SERVO_SHOULDER_PAN, 504], [SERVO_SHOULDER_LIFT, 672], [SERVO_ELBOW_FLEX, 939], [SERVO_WRIST_FLEX, 105], [SERVO_WRIST_ROLL, 502], [SERVO_GRIPPER, 365]],
    # near-right
    [[SERVO_SHOULDER_PAN, 129], [SERVO_SHOULDER_LIFT, 672], [SERVO_ELBOW_FLEX, 940], [SERVO_WRIST_FLEX, 104], [SERVO_WRIST_ROLL, 502], [SERVO_GRIPPER, 365]],
    # mid-left  (interpolated halfway between near and far)
    [[SERVO_SHOULDER_PAN, 894], [SERVO_SHOULDER_LIFT, 590], [SERVO_ELBOW_FLEX, 886], [SERVO_WRIST_FLEX, 150], [SERVO_WRIST_ROLL, 503], [SERVO_GRIPPER, 365]],
    # mid-center
    [[SERVO_SHOULDER_PAN, 507], [SERVO_SHOULDER_LIFT, 588], [SERVO_ELBOW_FLEX, 886], [SERVO_WRIST_FLEX, 150], [SERVO_WRIST_ROLL, 502], [SERVO_GRIPPER, 365]],
    # mid-right
    [[SERVO_SHOULDER_PAN, 137], [SERVO_SHOULDER_LIFT, 589], [SERVO_ELBOW_FLEX, 886], [SERVO_WRIST_FLEX, 150], [SERVO_WRIST_ROLL, 502], [SERVO_GRIPPER, 365]],
    # far-left
    [[SERVO_SHOULDER_PAN, 900], [SERVO_SHOULDER_LIFT, 507], [SERVO_ELBOW_FLEX, 833], [SERVO_WRIST_FLEX, 196], [SERVO_WRIST_ROLL, 503], [SERVO_GRIPPER, 365]],
    # far-center
    [[SERVO_SHOULDER_PAN, 510], [SERVO_SHOULDER_LIFT, 505], [SERVO_ELBOW_FLEX, 833], [SERVO_WRIST_FLEX, 196], [SERVO_WRIST_ROLL, 503], [SERVO_GRIPPER, 365]],
    # far-right
    [[SERVO_SHOULDER_PAN, 145], [SERVO_SHOULDER_LIFT, 506], [SERVO_ELBOW_FLEX, 833], [SERVO_WRIST_FLEX, 196], [SERVO_WRIST_ROLL, 503], [SERVO_GRIPPER, 365]],
]
SCAN_POSE_LABELS = ['near-left', 'near-center', 'near-right', 'mid-left', 'mid-center', 'mid-right', 'far-left', 'far-center', 'far-right']

# Resting pose: shoulder raised up/back, elbow and wrist curved forward/down.
# Tune these values to match your preferred resting position — they are the
# starting position when ball_follow activates and the return position after
# a successful grab.
HOME_POSE = [
    [SERVO_SHOULDER_PAN,  500],  # centered
    [SERVO_SHOULDER_LIFT, 750],  # raised up/back (scan range 507-673; >673 rotates past down into backward arc)
    [SERVO_ELBOW_FLEX,    900],  # flexed down to keep center of mass over base
    [SERVO_WRIST_FLEX,    300],  # hand parallel to ground (compensates for backward shoulder + elbow flex)
    [SERVO_WRIST_ROLL,    503],  # neutral
    [SERVO_GRIPPER,       365],  # open (match scan gripper)
]
# Same as HOME_POSE but omits the gripper so a held ball is not dropped.
HOME_POSE_KEEP_GRIP = [
    [SERVO_SHOULDER_PAN,  500],
    [SERVO_SHOULDER_LIFT, 750],
    [SERVO_ELBOW_FLEX,    900],
    [SERVO_WRIST_FLEX,    300],
    [SERVO_WRIST_ROLL,    503],
]


# Reusable Servo objects so batched getPosition doesn't allocate per frame.
_ALL_SERVOS = [Servo(i) for i in range(1, 7)]


def _read_all_positions(arm):
    """Per-servo reads of all six positions. Returns {id: pos}.

    We do NOT use the batched ``arm.getPosition([Servo...])`` call: when its
    response parse throws IndexError it leaves leftover bytes in the HID read
    buffer, permanently desynchronising the stream. Every subsequent read
    then mis-parses and the process must be restarted with a USB power-cycle.
    Per-servo reads cost ~6 USB RTTs (~40 ms) but don't poison the stream.
    """
    return {sid: int(arm.getPosition(sid)) for sid in range(1, 7)}


def _clamp(v, lo=0, hi=1000):
    return max(lo, min(hi, int(v)))


def _step_toward(error_px, gain, max_step, min_step=MIN_TRIM_STEP):
    """Convert a pixel error to a clamped servo delta (sign preserved).

    Floors the magnitude at ``min_step`` so trims aren't silently swallowed by
    bus-servo static friction. Caller still gates on a separate deadband check
    before invoking this, so the floor only applies when we've already decided
    a move is warranted.
    """
    raw = error_px * gain
    if raw > max_step:
        return max_step
    if raw < -max_step:
        return -max_step
    if raw > 0 and raw < min_step:
        return min_step
    if raw < 0 and raw > -min_step:
        return -min_step
    return raw


def _lookup_grab_lift(tracking_lift):
    """Linearly interpolate grab shoulder_lift from GRAB_LIFT_TABLE based on tracking lift."""
    table = sorted(GRAB_LIFT_TABLE, key=lambda e: e[0])  # ascending ref_tracking_lift
    if tracking_lift <= table[0][0]:
        return table[0][1]
    if tracking_lift >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        lo, hi = table[i], table[i + 1]
        if lo[0] <= tracking_lift <= hi[0]:
            t = (tracking_lift - lo[0]) / (hi[0] - lo[0])
            return int(round(lo[1] + (hi[1] - lo[1]) * t))
    return table[-1][1]


class BallFollowMode(Mode):
    name = "ball"
    skip_global_home = True   # suppress main.py's all-500 home move; we start from wherever the arm is

    def __init__(self):
        self.lower = None
        self.upper = None
        self.frame_count = 0
        self.state = "IDLE"
        self.last_log = ""
        self.hold_target = None  # gripper position we're holding at; None = not holding
        self.no_ball_frames = 0  # consecutive frames with no ball detected
        self.ball_confirm = 0    # consecutive real-ball frames; HALT waits for SCAN_CONFIRM_FRAMES
        self.scan_active = True  # True = actively sweeping; False = just lost a tracked ball
        self.pos_buffer = deque(maxlen=POS_BUFFER_LEN)  # (frame_idx, bx, by, br)
        self.pred_x = 0.0
        self.pred_y = 0.0
        self.pred_r = 0.0
        self.pred_frames_remaining = 0
        self.last_vel = (0.0, 0.0)
        self.last_ball = (0, 0, 0)   # most recent (bx, by, br), real or predicted
        self.last_errs = (0, 0, 0)   # pan_err, tilt_err, radius_err
        self.last_deltas = (0, 0, 0, 0)  # d_pan, d_tilt, d_lift, d_elbow
        self.last_is_prediction = False
        self.last_telemetry_at = 0.0
        self.scan_idx = 0
        self.last_scan_move_at = 0.0
        # perf instrumentation (printed from process_frame every PERF_EVERY frames)
        self._perf_n = 0
        self._perf_detect = 0.0
        self._perf_arm = 0.0
        self._perf_every = 30

    def setup(self, arm):
        if not os.path.exists(CONFIG_PATH):
            raise RuntimeError(f"ball_color.json not found at {CONFIG_PATH}; run ball_calibrate.py first")
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        self.lower = np.array([cfg["h_min"], cfg["s_min"], cfg["v_min"]], dtype=np.uint8)
        self.upper = np.array([cfg["h_max"], cfg["s_max"], cfg["v_max"]], dtype=np.uint8)
        print(f"[ball] HSV range loaded: lower={self.lower.tolist()} upper={self.upper.tolist()}")
        print(f"[ball] moving to scan pose [{SCAN_POSE_LABELS[0]}]...")
        arm.setPosition(SCAN_POSES[0], duration=2500, wait=True)
        self.scan_idx = 0
        self.scan_active = True
        self.ball_confirm = 0
        # Backdate by SCAN_MOVE_MS only so next_advance_at = now + SCAN_DWELL_S.
        # Arm just arrived at near-left; give it the full dwell (1.5 s) to look
        # before sweeping. Backdating by SCAN_MOVE_MS+SCAN_DWELL_S was advancing
        # instantly, before the arm (coming from HOME) had fully settled at near-left.
        self.last_scan_move_at = time.time() - SCAN_MOVE_MS / 1000.0
        self.state = "IDLE"

    def teardown(self, arm):
        try:
            print("[ball] returning to home pose...")
            arm.setPosition(HOME_POSE, duration=2000, wait=True)
        except Exception as e:
            print(f"[ball] teardown move failed: {e}")

    def process_frame(self, frame, arm):
        from main import send_telemetry  # lazy to avoid circular import

        self.frame_count += 1
        h, w = frame.shape[:2]
        # Aim at the pixel where the ball appears when the gripper is over it,
        # not the geometric image center. See CAM_GRIPPER_OFFSET_X/Y.
        cx_target = w // 2 + CAM_GRIPPER_OFFSET_X
        cy_target = h // 2 + CAM_GRIPPER_OFFSET_Y

        # Single batched USB read for all 6 servos, reused below.
        # If even the per-servo fallback inside _read_all_positions fails,
        # skip this frame — commanding setPosition with bogus "current" values
        # (e.g. all 500s) would make the arm jump.
        try:
            pos = _read_all_positions(arm)
        except Exception as e:
            print(f"[ball] position read failed entirely — skipping frame: {e}")
            return cv2.flip(frame, 1)

        # If we're currently holding something, skip tracking entirely.
        # User can open the gripper (via IoTConnect) to release and resume.
        if self.hold_target is not None:
            gpos = pos[SERVO_GRIPPER]
            if gpos < self.hold_target - GRIPPER_RELEASE_DELTA:
                self._log(f"RELEASED: gripper {gpos} < hold_target {self.hold_target} - {GRIPPER_RELEASE_DELTA}")
                self.hold_target = None
            else:
                annotated = cv2.flip(frame, 1)
                cv2.putText(annotated, f"HOLDING @ {gpos} — tracking paused",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2, cv2.LINE_AA)
                self._log(f"HOLDING: pos={gpos} target={self.hold_target}")
                self.state = "IDLE"
                return annotated

        t_detect_start = time.perf_counter()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        real_ball = self._largest_blob(mask)
        t_detect = time.perf_counter() - t_detect_start
        is_prediction = False
        ball = real_ball
        if real_ball is None:
            self.ball_confirm = 0  # streak broken — reset confirmation counter
            predicted = self._maybe_predict()
            if predicted is not None:
                ball = predicted
                is_prediction = True
        else:
            # Real observation — record it and cancel any in-progress prediction.
            self.pos_buffer.append((self.frame_count, real_ball[0], real_ball[1], real_ball[2]))
            self.pred_frames_remaining = 0
            self.ball_confirm += 1
            # Require SCAN_CONFIRM_FRAMES consecutive real detections before treating
            # this as a genuine ball. Until confirmed, do NOT reset no_ball_frames —
            # if we reset it on every intermittent 1-frame detection, no_ball_frames
            # oscillates 0→1→0→1 and never reaches either advance threshold, freezing
            # the scan indefinitely.
            if self.ball_confirm >= SCAN_CONFIRM_FRAMES:
                self.no_ball_frames = 0
                if self.ball_confirm == SCAN_CONFIRM_FRAMES and self.scan_active:
                    try:
                        halt_targets = [
                            [SERVO_SHOULDER_PAN, pos[SERVO_SHOULDER_PAN]],
                            [SERVO_WRIST_FLEX, pos[SERVO_WRIST_FLEX]],
                            [SERVO_SHOULDER_LIFT, pos[SERVO_SHOULDER_LIFT]],
                            [SERVO_ELBOW_FLEX, pos[SERVO_ELBOW_FLEX]],
                        ]
                        arm.setPosition(halt_targets, duration=80, wait=False)
                        print(f"[ball] HALT scan (confirmed {SCAN_CONFIRM_FRAMES}f): "
                              f"pan={pos[SERVO_SHOULDER_PAN]} flex={pos[SERVO_WRIST_FLEX]} "
                              f"lift={pos[SERVO_SHOULDER_LIFT]} elbow={pos[SERVO_ELBOW_FLEX]}")
                    except Exception as e:
                        print(f"[ball] halt failed: {e}")
                self.scan_active = False  # confirmed tracking; use long grace if ball disappears

        # Flip first so all text we draw below reads the right way around.
        # Ball x-coords must be mirrored too: bx_disp = w - bx.
        annotated = cv2.flip(frame, 1)
        # Mirror the target x for display since the frame is flipped horizontally.
        cv2.drawMarker(annotated, (w - cx_target, cy_target), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)

        if self.state == "GRABBING":
            cv2.putText(annotated, "GRABBING...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            return annotated

        if ball is None:
            # truly lost — nothing real, nothing to predict from
            self.no_ball_frames += 1
            label = SCAN_POSE_LABELS[self.scan_idx]
            moving = (time.time() - self.last_scan_move_at) < (SCAN_MOVE_MS / 1000.0)
            prefix = "→" if moving else ""
            self._log(f"SCAN[{prefix}{label}]: no ball ({self.no_ball_frames})")
            cv2.putText(annotated, f"scan {prefix}{label}  no ball ({self.no_ball_frames})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            self.state = "SCANNING"
            self.last_is_prediction = False
            # After a brief grace period, cycle scan poses to search the table.
            # wait=False so the camera loop keeps reading frames during the move
            # (otherwise video freezes for SCAN_MOVE_MS each transition). We pace
            # transitions ourselves: don't advance until the prior move's expected
            # completion time has passed plus SCAN_DWELL_S of viewing time.
            grace = SCAN_ADVANCE_FRAMES if self.scan_active else NO_BALL_GRACE_FRAMES
            if self.no_ball_frames >= grace:
                now = time.time()
                next_advance_at = self.last_scan_move_at + (SCAN_MOVE_MS / 1000.0) + SCAN_DWELL_S
                if now >= next_advance_at:
                    self.scan_idx = (self.scan_idx + 1) % len(SCAN_POSES)
                    next_label = SCAN_POSE_LABELS[self.scan_idx]
                    print(f"[ball] scanning -> {next_label}")
                    try:
                        arm.setPosition(SCAN_POSES[self.scan_idx], duration=SCAN_MOVE_MS, wait=False)
                    except Exception as e:
                        print(f"[ball] scan move failed: {e}")
                    self.last_scan_move_at = time.time()
                    self.scan_active = True
                    self.no_ball_frames = 0  # reset per-pose so each pose gets a fresh grace window
                elif self.no_ball_frames == grace and not self.scan_active:
                    # A HALT (from confirmed or false ball) froze the arm mid-sweep.
                    # Grace period has elapsed but the sweep timer hasn't expired —
                    # re-issue the move so the arm actually reaches the current target.
                    print(f"[ball] scan resume → {SCAN_POSE_LABELS[self.scan_idx]}")
                    try:
                        arm.setPosition(SCAN_POSES[self.scan_idx], duration=SCAN_MOVE_MS, wait=False)
                    except Exception as e:
                        print(f"[ball] scan resume failed: {e}")
                    self.last_scan_move_at = time.time()
                    self.scan_active = True
                    self.no_ball_frames = 0
            self._maybe_send_telemetry(arm, send_telemetry, positions=pos)
            return annotated

        bx, by, br = ball
        bx_disp = w - bx
        color = (0, 165, 255) if is_prediction else (0, 255, 0)  # orange vs green
        cv2.circle(annotated, (bx_disp, by), int(br), color, 2)
        cv2.circle(annotated, (bx_disp, by), 4, color, -1)
        if is_prediction:
            cv2.putText(annotated, f"PREDICTING ({self.pred_frames_remaining} left)",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 165, 255), 2, cv2.LINE_AA)
        pan_err = bx - cx_target
        tilt_err = by - cy_target
        radius_err = TARGET_RADIUS_PX - br
        centered_ok = abs(pan_err) <= CENTER_DEADBAND_PX and abs(tilt_err) <= CENTER_DEADBAND_PX
        # Looser gate that just requires the ball to be roughly under the gripper:
        # used to allow descent while pan/tilt are still trimming, so we don't
        # sit forever waiting for perfect centering before reaching for the ball.
        approach_centered_ok = abs(pan_err) <= APPROACH_DEADBAND_PX and abs(tilt_err) <= APPROACH_DEADBAND_PX
        radius_ok = abs(radius_err) <= RADIUS_TOLERANCE
        cv2.putText(annotated,
                    f"r={int(br)} dx={pan_err:+d} dy={tilt_err:+d}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated,
                    f"center={'OK' if centered_ok else 'NO'}  radius={'OK' if radius_ok else 'NO'} (target={TARGET_RADIUS_PX}+/-{RADIUS_TOLERANCE})",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if centered_ok and radius_ok else (0, 200, 255),
                    2, cv2.LINE_AA)

        # Are we close + centered enough to grab? Only on real observations —
        # never close the gripper on a predicted position.
        if not is_prediction and centered_ok and radius_ok:
            self.state = "GRABBING"
            self._log("GRAB: centered and at distance — closing gripper")
            self._do_grab(arm, pos)
            self.state = "HOLDING"
            try:
                send_telemetry(arm)
            except Exception as e:
                print(f"[ball] telemetry after grab failed: {e}")
            self.state = "IDLE"
            return annotated

        # Throttle servo commands so we don't spam at full frame rate.
        if self.frame_count % UPDATE_EVERY != 0:
            return annotated

        # Don't issue servo commands while waiting for scan confirmation.
        # A 220ms tracking command overrides the 5000ms scan move and stops
        # the arm from reaching its target — the scan appears to freeze.
        if self.scan_active and self.ball_confirm < SCAN_CONFIRM_FRAMES:
            return annotated

        self.state = "PREDICTING" if is_prediction else "TRACKING"
        d_pan = 0
        d_tilt = 0
        d_lift = 0
        d_elbow = 0

        if abs(pan_err) > CENTER_DEADBAND_PX:
            d_pan = int(round(_step_toward(pan_err, PAN_GAIN, MAX_STEP,
                                            min_step=MIN_TRIM_STEP_PAN))) * PAN_DIR
        if abs(tilt_err) > CENTER_DEADBAND_PX:
            d_tilt = int(round(_step_toward(tilt_err, TILT_GAIN, MAX_STEP))) * TILT_DIR
            # Elbow helps with larger tilts so wrist_flex doesn't max out alone.
            d_elbow += int(round(d_tilt * TILT_ELBOW_RATIO)) * TILT_ELBOW_DIR
        # Allow descent as soon as the ball is roughly under the gripper
        # (looser approach_centered_ok). Pan/tilt corrections continue every
        # frame and trim the ball back toward the gripper aim point even while
        # we descend, so we don't wait forever for perfect centering.
        if approach_centered_ok and br < TARGET_RADIUS_PX - RADIUS_TOLERANCE:
            # Hard ceiling: if we're already at the safety limit, stop approaching.
            # Prevents table slam when radius_ok never triggers.
            if pos[SERVO_SHOULDER_LIFT] >= LIFT_MAX or pos[SERVO_ELBOW_FLEX] >= ELBOW_MAX:
                self._log(f"APPROACH-BLOCKED: lift={pos[SERVO_SHOULDER_LIFT]}>={LIFT_MAX} or elbow={pos[SERVO_ELBOW_FLEX]}>={ELBOW_MAX}; r={int(br)} target={TARGET_RADIUS_PX}")
            else:
                d_lift = APPROACH_STEP   # reach forward
                # Coordinate elbow with shoulder — extends the arm rather than just
                # tipping the whole thing from the shoulder. Ratio is empirical.
                d_elbow += int(round(APPROACH_STEP * ELBOW_REACH_RATIO))
                # Co-trim wrist during descent even when inside CENTER_DEADBAND_PX.
                # Each lift step shifts the ball ~30 px upward in the frame (5 px per
                # lift unit measured 2026-04-19), which kicks tilt_err well past the
                # 60 px deadband and the wrist alone needs 5+ frames to recover —
                # producing the "descend once every several frames, prefers center=NO"
                # pattern. Issuing the wrist nudge in the same setPosition as the lift
                # serializes "descend + recenter" into one command, so the controller
                # makes net progress every frame instead of ping-ponging.
                # Pan is NOT co-trimmed: lift/elbow don't shift ball horizontally
                # much, and a floored MIN_TRIM_STEP_PAN command for a within-deadband
                # error overshoots the deadband (~126 px shift > 60 px deadband),
                # causing the same +18/-18 sign-flip oscillation we hit on 2026-04-19.
                if d_tilt == 0 and abs(tilt_err) > 20:
                    d_tilt = int(round(_step_toward(tilt_err, TILT_GAIN, MAX_STEP))) * TILT_DIR
        elif approach_centered_ok and br > TARGET_RADIUS_PX + RADIUS_TOLERANCE:
            # Ball too close — retreat so we can reach the grab distance.
            # Without this the arm stalls at near scan positions (lift≈670) where
            # the ball already appears larger than TARGET_RADIUS_PX and can never
            # trigger a grab or an approach step.
            if pos[SERVO_SHOULDER_LIFT] <= LIFT_MIN or pos[SERVO_ELBOW_FLEX] <= ELBOW_MIN:
                self._log(f"RETREAT-BLOCKED: lift={pos[SERVO_SHOULDER_LIFT]}<={LIFT_MIN} or elbow={pos[SERVO_ELBOW_FLEX]}<={ELBOW_MIN}; r={int(br)} target={TARGET_RADIUS_PX}")
            else:
                d_lift = -APPROACH_STEP  # pull back from table
                d_elbow += int(round(-APPROACH_STEP * ELBOW_REACH_RATIO))
                if d_tilt == 0 and abs(tilt_err) > 20:
                    d_tilt = int(round(_step_toward(tilt_err, TILT_GAIN, MAX_STEP))) * TILT_DIR

        if d_pan == 0 and d_tilt == 0 and d_lift == 0 and d_elbow == 0:
            return annotated

        targets = []
        pan_now = pos[SERVO_SHOULDER_PAN]
        if d_pan:
            pan_now = _clamp(pos[SERVO_SHOULDER_PAN] + d_pan, lo=PAN_MIN, hi=PAN_MAX)
            targets.append([SERVO_SHOULDER_PAN, pan_now])
        # Wrist-roll compensation: keep gripper tips level when shoulder_pan deviates
        # from center. WRIST_ROLL_COMPENSATION is calibrated by calibrate_cam_offset.py.
        _wrist_roll_tgt = _clamp(int(WRIST_ROLL_HOME - (pan_now - 500) * WRIST_ROLL_COMPENSATION))
        if abs(_wrist_roll_tgt - pos[SERVO_WRIST_ROLL]) >= 5:
            targets.append([SERVO_WRIST_ROLL, _wrist_roll_tgt])
        if d_tilt:
            new_tilt = _clamp(pos[SERVO_WRIST_FLEX] + d_tilt)
            targets.append([SERVO_WRIST_FLEX, new_tilt])
        if d_lift:
            new_lift = _clamp(pos[SERVO_SHOULDER_LIFT] + d_lift, lo=LIFT_MIN, hi=LIFT_MAX)
            targets.append([SERVO_SHOULDER_LIFT, new_lift])
        if d_elbow:
            new_elbow = _clamp(pos[SERVO_ELBOW_FLEX] + d_elbow, lo=ELBOW_MIN, hi=ELBOW_MAX)
            targets.append([SERVO_ELBOW_FLEX, new_elbow])

        t_arm_start = time.perf_counter()
        if targets:
            self._log(f"{self.state}: dpan={d_pan} dtilt={d_tilt} dlift={d_lift} delbow={d_elbow}")
            try:
                # wait=False so the camera loop doesn't freeze for ~MOVE_DURATION_MS
                # while the servo travels. Bus servos accept a new target while a
                # prior move is in flight — they just retarget smoothly. Each new
                # frame issues a fresh small correction based on the latest pixel
                # error, which is what we want.
                arm.setPosition(targets, duration=MOVE_DURATION_MS, wait=False)
            except Exception as e:
                print(f"[ball] setPosition failed: {e}")
        t_arm = time.perf_counter() - t_arm_start
        self._perf_n += 1
        self._perf_detect += t_detect
        self._perf_arm += t_arm
        if self._perf_n >= self._perf_every:
            print(f"[ball-perf] n={self._perf_n}  detect={self._perf_detect/self._perf_n*1000:.1f}ms  "
                  f"arm={self._perf_arm/self._perf_n*1000:.1f}ms")
            self._perf_n = 0
            self._perf_detect = 0.0
            self._perf_arm = 0.0

        # snapshot state for telemetry
        self.last_ball = (int(bx), int(by), int(br))
        self.last_errs = (int(pan_err), int(tilt_err), int(radius_err))
        self.last_deltas = (int(d_pan), int(d_tilt), int(d_lift), int(d_elbow))
        self.last_is_prediction = is_prediction
        self._maybe_send_telemetry(arm, send_telemetry, positions=pos)

        return annotated

    @staticmethod
    def _largest_blob(mask):
        """Pick the largest blob that's also round-ish.

        A ball's contour fills its minimum enclosing circle by ~85% or more.
        Irregular ball-color objects (a logo, a corner of fabric, a shadow
        edge) typically fill <50% of their enclosing circle even if their
        raw area is large. The fill_ratio filter rejects those false-positive
        winners that the area-only check used to accept.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA or area <= best_area:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < MIN_BALL_RADIUS or r > MAX_BALL_RADIUS:
                continue
            fill_ratio = area / (np.pi * r * r)
            if fill_ratio < MIN_FILL_RATIO:
                continue
            best = (int(x), int(y), float(r))
            best_area = area
        return best

    def _do_grab(self, arm, pos):
        # Apply wrist-roll compensation before snapping to grab position so the
        # gripper tips are level when the jaws close, regardless of pan angle.
        _roll_tgt = _clamp(int(WRIST_ROLL_HOME - (pos[SERVO_SHOULDER_PAN] - 500) * WRIST_ROLL_COMPENSATION))
        if abs(_roll_tgt - pos[SERVO_WRIST_ROLL]) >= 5:
            arm.setPosition([[SERVO_WRIST_ROLL, _roll_tgt]], duration=300, wait=True)
        # Snap wrist, elbow, and shoulder back to grab position.
        # grab_lift is interpolated from GRAB_LIFT_TABLE based on current tracking lift
        # so the arm reaches the correct height whether the ball is near or far.
        try:
            tracking_lift = pos[SERVO_SHOULDER_LIFT]
            grab_lift  = _lookup_grab_lift(tracking_lift)
            current_wrist = int(arm.getPosition(SERVO_WRIST_FLEX))
            current_elbow = int(arm.getPosition(SERVO_ELBOW_FLEX))
            grab_wrist = _clamp(current_wrist - WRIST_VIEW_OFFSET + GRAB_WRIST_TRIM)
            grab_elbow = _clamp(current_elbow - ELBOW_VIEW_OFFSET)
            print(f"[ball] grab snap (tracking_lift={tracking_lift}): "
                  f"lift->{grab_lift}  wrist {current_wrist}->{grab_wrist}  elbow {current_elbow}->{grab_elbow}")
            arm.setPosition([
                [SERVO_SHOULDER_LIFT, grab_lift],
                [SERVO_WRIST_FLEX,    grab_wrist],
                [SERVO_ELBOW_FLEX,    grab_elbow],
            ], duration=600, wait=True)
        except Exception as e:
            print(f"[ball] grab snap failed: {e}")

        # Open fully, close on the ball, detect stall + relax, then lift and
        # return home. Home pose deliberately excludes the gripper so the ball
        # isn't dropped.
        arm.setPosition(SERVO_GRIPPER, 60, duration=500, wait=True)
        arm.setPosition(SERVO_GRIPPER, GRIPPER_CLOSE_TARGET, duration=1400, wait=True)

        actual = arm.getPosition(SERVO_GRIPPER)
        if actual < GRIPPER_CLOSE_TARGET - GRIPPER_STALL_SLACK:
            relaxed = actual + GRIPPER_RELAX_OFFSET
            print(f"[ball] gripper stalled at {actual} (target {GRIPPER_CLOSE_TARGET}); relaxing to {relaxed}")
            arm.setPosition(SERVO_GRIPPER, relaxed, duration=200, wait=True)
            self.hold_target = relaxed
        else:
            self.hold_target = GRIPPER_CLOSE_TARGET

        # First raise the arm clear of the table, then move to home position.
        arm.setPosition([
            [SERVO_SHOULDER_LIFT, 350],
            [SERVO_ELBOW_FLEX,    350],
        ], duration=1500, wait=True)
        arm.setPosition(HOME_POSE_KEEP_GRIP, duration=1800, wait=True)

    def _maybe_predict(self):
        """Return a (bx, by, br) tuple extrapolated from recent motion, or None.

        On first call after a ball loss, estimates velocity from self.pos_buffer
        and seeds pred_x/pred_y. Subsequent calls step the prediction forward by
        the estimated per-frame velocity. Runs for at most MAX_PREDICT_FRAMES.
        """
        if not ENABLE_PREDICTION:
            return None
        if self.pred_frames_remaining > 0:
            self.pred_x += self.last_vel[0]
            self.pred_y += self.last_vel[1]
            self.pred_frames_remaining -= 1
            return (int(self.pred_x), int(self.pred_y), self.pred_r)

        if len(self.pos_buffer) < 2:
            return None
        fi_old, bx_old, by_old, _ = self.pos_buffer[0]
        fi_new, bx_new, by_new, br_new = self.pos_buffer[-1]
        dt = fi_new - fi_old
        if dt <= 0:
            return None
        vx = (bx_new - bx_old) / dt
        vy = (by_new - by_old) / dt
        if vx * vx + vy * vy < MIN_VELOCITY_PX_PER_FRAME ** 2:
            return None
        # account for frames that have already elapsed since the last real obs
        frames_since = max(1, self.frame_count - fi_new)
        self.last_vel = (vx, vy)
        self.pred_x = bx_new + vx * frames_since
        self.pred_y = by_new + vy * frames_since
        self.pred_r = br_new
        self.pred_frames_remaining = MAX_PREDICT_FRAMES - 1
        print(f"[ball] predicting: v=({vx:+.1f},{vy:+.1f}) px/frame, {MAX_PREDICT_FRAMES} frames")
        return (int(self.pred_x), int(self.pred_y), self.pred_r)

    def _maybe_send_telemetry(self, arm, send_telemetry, positions=None):
        now = time.time()
        if now - self.last_telemetry_at < TELEMETRY_INTERVAL_S:
            return
        try:
            send_telemetry(arm, extras={"ballTrack": self.telemetry()}, positions=positions)
            self.last_telemetry_at = now
        except Exception as e:
            print(f"[ball] telemetry failed: {e}")

    def get_state(self):
        return self.state

    def telemetry(self):
        """Build a flat dict of mode state for IoTConnect telemetry."""
        bx, by, br = self.last_ball
        pan_err, tilt_err, radius_err = self.last_errs
        d_pan, d_tilt, d_lift, d_elbow = self.last_deltas
        vx, vy = self.last_vel
        return {
            "state": self.state,
            "is_prediction": 1 if self.last_is_prediction else 0,
            "ball_x": bx,
            "ball_y": by,
            "ball_r": br,
            "pan_err": pan_err,
            "tilt_err": tilt_err,
            "radius_err": radius_err,
            "velocity_x": round(float(vx), 2),
            "velocity_y": round(float(vy), 2),
            "pred_frames_left": int(self.pred_frames_remaining),
            "no_ball_frames": int(self.no_ball_frames),
            "hold_target": int(self.hold_target) if self.hold_target is not None else 0,
            "d_pan": d_pan,
            "d_tilt": d_tilt,
            "d_lift": d_lift,
            "d_elbow": d_elbow,
        }

    def _log(self, msg):
        if msg != self.last_log:
            print(f"[ball] {msg}")
            self.last_log = msg
