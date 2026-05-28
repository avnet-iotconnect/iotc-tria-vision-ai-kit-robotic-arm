#!/usr/bin/env python3
"""Teach D_grab — the relative-depth value at the ball's centre when the arm
is in a known grab pose. Run this once, then the runtime depth-gated grab
uses the saved value (``grab_depth.json``) to decide when to close the gripper.

Workflow:
  1. Stop the running demo so /dev/video2 is free.
  2. Run ``./teach_grab.sh --model model/ball_best.tflite --web-port 8080``.
  3. SUPPORT THE ARM by hand, press Enter to release torque.
  4. Hand-pose the arm so the gripper is in the grab position, with a ball
     clearly visible to the wrist camera (slightly higher than touchdown so
     the ball isn't fully occluded by the gripper jaws).
  5. Watch the live D in the console + the browser preview.
  6. When happy, press 's' + Enter — captures ``--frames`` snapshots and
     saves the median D to ``grab_depth.json``.
  7. 'h' = hold (re-enable torque at current pose). 'r' = release again.
     'q' = quit (auto-holds for safety).

Ctrl-C re-enables torque before exiting.
"""

import argparse
import json
import os
import signal
import statistics
import sys
import threading
import time

import cv2
import xarm
from xarm import Servo

import main as app  # _FreshCamera applies camera_settings.json
from detectors import make_ball_detector, DepthDetector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, "grab_depth.json")

_ALL_SERVOS = [Servo(i) for i in range(1, 7)]
_arm_lock = threading.Lock()


def hold_current_pose(arm):
    """Re-enable torque by commanding each servo to its measured position."""
    with _arm_lock:
        arm.getPosition(_ALL_SERVOS)
        targets = [[s.servo_id, int(s.position)] for s in _ALL_SERVOS]
        arm.setPosition(targets, duration=1200, wait=True)


def release_torque(arm):
    with _arm_lock:
        arm.servoOff()


def parse_args():
    ap = argparse.ArgumentParser(description="Teach D_grab for depth-gated grab")
    ap.add_argument("--camera", type=int, default=2)
    ap.add_argument("--model", default="/etc/models/yolox_quantized.tflite",
                    help="YOLO model used to find the ball during teach")
    ap.add_argument("--conf", type=float, default=None,
                    help="confidence threshold (auto: 0.25 yolox, 0.7 custom)")
    ap.add_argument("--depth-model", default="/etc/models/midas_quantized.tflite")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU TFLite for both detectors (skip Hexagon HTP)")
    ap.add_argument("--frames", type=int, default=20,
                    help="frames to median over for the D_grab snapshot")
    ap.add_argument("--web-port", type=int, default=0,
                    help="serve a live preview on this port (0=off)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.conf is None:
        args.conf = 0.25 if "yolox" in os.path.basename(args.model).lower() else 0.7

    print("[teach-grab] connecting to arm...")
    arm = xarm.Controller("USB")

    def _on_sigint(*_):
        # Keep this handler NON-BLOCKING — it must complete fast so a second
        # Ctrl-C can actually escape. Calling hold_current_pose() here was a
        # bug: it does blocking USB reads/writes and re-entrant Ctrl-Cs got
        # queued inside the blocked call, hanging the process indefinitely.
        # If you need torque held on a hand-posed arm, use 'h' before quitting.
        print("\n[teach-grab] Ctrl-C — exiting "
              "(torque state UNCHANGED; use 'h' first if the arm should hold)")
        os._exit(130)
    signal.signal(signal.SIGINT, _on_sigint)

    print("[teach-grab] loading detectors (this takes a few seconds)...")
    ball_det = make_ball_detector(model_name=args.model, conf_thres=args.conf,
                                  use_npu=not args.cpu)
    depth_det = DepthDetector(args.depth_model, use_npu=not args.cpu)

    print(f"[teach-grab] opening camera index {args.camera}...")
    cam = app._FreshCamera(args.camera, 640, 480)
    if not cam.start():
        raise SystemExit("[teach-grab] camera failed to open (demo still running?)")

    web = None
    if args.web_port:
        try:
            from web_view import WebView
            web = WebView(port=args.web_port)
            print(f"[teach-grab] live preview: {web.url_hint()}")
        except Exception as e:
            print(f"[teach-grab] web view failed: {e}")

    # Wait for the first frame
    t0 = time.time()
    while cam.read() is None and time.time() - t0 < 3.0:
        time.sleep(0.02)

    state = {"last_D": None, "last_ball": None, "running": True}

    def loop():
        last_print = 0.0
        while state["running"]:
            f = cam.read()
            if f is None:
                time.sleep(0.02); continue
            dets = ball_det.detect(f)
            ball = max(dets, key=lambda d: d.r) if dets else None
            state["last_ball"] = ball
            D = None
            if ball is not None:
                dm = depth_det.infer(f)
                D = depth_det.at(dm, ball.cx, ball.cy, patch_r=max(5, int(ball.r / 3)))
            state["last_D"] = D

            disp = cv2.flip(f, 1)
            if ball is not None:
                cv2.circle(disp, (640 - ball.cx, ball.cy), int(ball.r), (0, 255, 0), 2)
                cv2.putText(disp, f"D={D:.1f}" if D is not None else "D=?",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(disp, f"r={ball.r:.0f}  conf={ball.conf:.2f}",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(disp, "no ball detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(disp, "TEACH GRAB DEPTH  (s=snapshot  h=hold  r=release  q=quit)",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if web is not None:
                try:
                    web.publish(disp, state="teach", mode="teach-grab-depth")
                except Exception:
                    pass

            now = time.time()
            if now - last_print >= 1.0:
                if ball is not None and D is not None:
                    print(f"[teach-grab] ball r={ball.r:.0f} conf={ball.conf:.2f} D={D:.1f}")
                else:
                    print("[teach-grab] no ball detected")
                last_print = now
            time.sleep(0.02)

    t = threading.Thread(target=loop, daemon=True)
    t.start()

    print("\n" + "=" * 60)
    print(" SAFETY: this will RELEASE the servos so you can hand-pose the arm.")
    print(" Support the arm BEFORE pressing Enter, or it will swing under")
    print(" gravity and could damage the mount / itself.")
    print("=" * 60)
    input("[teach-grab] Holding the arm? Press Enter to release torque... ")

    try:
        release_torque(arm)
        print("[teach-grab] torque OFF — hand-pose the arm so the ball is under")
        print("             the gripper and visible to the wrist camera.")
    except Exception as e:
        print(f"[teach-grab] servoOff failed: {e}")
        sys.exit(1)

    torque_on = False

    print("\n[teach-grab] commands: s=snapshot  h=hold  r=release  q=quit")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except EOFError:
            break

        if cmd == "s":
            print(f"[teach-grab] capturing {args.frames} frames at ~10 Hz...")
            d_vals, r_vals = [], []
            for _ in range(args.frames):
                ball, D = state["last_ball"], state["last_D"]
                if ball is not None and D is not None:
                    d_vals.append(D); r_vals.append(ball.r)
                time.sleep(0.1)
            if len(d_vals) < args.frames * 0.5:
                print(f"[teach-grab] only {len(d_vals)}/{args.frames} valid frames "
                      "(ball not consistently detected) — reposition + retry")
                continue
            D_grab = statistics.median(d_vals)
            r_grab = statistics.median(r_vals)
            stdev_d = statistics.stdev(d_vals) if len(d_vals) > 1 else 0.0
            if stdev_d < 0.5:
                print("[teach-grab] WARNING: D stdev is ~0 across all 20 frames.")
                print("             This almost always means torque was ON (arm")
                print("             locked) when you snapshotted, so you captured")
                print("             the SETTLED-after-hold pose, NOT your hand-pose.")
                print("             Hand-pose the arm (torque OFF), press 's' FIRST,")
                print("             THEN 'h'+'q'. Aborting save.")
                continue
            payload = {
                "D_grab": float(D_grab),
                "D_min": float(min(d_vals)),
                "D_max": float(max(d_vals)),
                "D_stdev": float(stdev_d),
                "ball_r_at_grab": float(r_grab),
                "n_frames": len(d_vals),
                "model": os.path.basename(args.model),
                "depth_model": os.path.basename(args.depth_model),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with open(OUT_FILE, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[teach-grab] D_grab = {D_grab:.1f}  "
                  f"(n={len(d_vals)}, stdev={payload['D_stdev']:.1f}, "
                  f"range {min(d_vals):.1f}..{max(d_vals):.1f})")
            print(f"[teach-grab] ball_r_at_grab = {r_grab:.1f}")
            print(f"[teach-grab] saved -> {OUT_FILE}")

        elif cmd == "h":
            print("[teach-grab] re-enabling torque at current pose...")
            try:
                hold_current_pose(arm)
                torque_on = True
                print("[teach-grab] torque ON")
            except Exception as e:
                print(f"[teach-grab] hold failed: {e}")

        elif cmd == "r":
            print("[teach-grab] releasing torque...")
            try:
                release_torque(arm)
                torque_on = False
                print("[teach-grab] torque OFF")
            except Exception as e:
                print(f"[teach-grab] release failed: {e}")

        elif cmd in ("q", "quit", "exit"):
            break

        elif cmd == "":
            ball, D = state["last_ball"], state["last_D"]
            if ball is not None and D is not None:
                print(f"[teach-grab] r={ball.r:.0f} conf={ball.conf:.2f} D={D:.1f}")
            else:
                print("[teach-grab] no ball detected")

        else:
            print(f"[teach-grab] unknown {cmd!r}; try s/h/r/q")

    state["running"] = False
    if not torque_on:
        print("[teach-grab] WARNING: torque is OFF — the arm may sag. "
              "Use 'h' BEFORE 'q' next time to lock the pose before exiting.")
    cam.release()
    if web is not None:
        web.stop()
    print("[teach-grab] done")


if __name__ == "__main__":
    main()
