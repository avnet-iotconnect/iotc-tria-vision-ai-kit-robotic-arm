#!/usr/bin/env python3
"""
IOTCONNECT XArm Gesture Control
Demonstrates IOTCONNECT integration with ASL gesture-controlled XArm 1S robot.
Features real-time telemetry transmission and remote command execution.
"""

import json
import os
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import systemdata
import torch
import xarm

from collections import deque
from datetime import datetime
from dataclasses import asdict

# IOTCONNECT SDK imports
try:
    from avnet.iotconnect.sdk.lite import Client, DeviceConfig, Callbacks, DeviceConfigError
    from avnet.iotconnect.sdk.sdklib.mqtt import C2dCommand, C2dAck
    IOTC_AVAILABLE = True
except ImportError:
    IOTC_AVAILABLE = False
    Client = None
    DeviceConfig = None
    Callbacks = None
    DeviceConfigError = None
    C2dCommand = None
    C2dAck = None

try:
    from point_net import PointNet
except ImportError:
    print("Warning: point_net module not found. Please ensure model/point_net.py exists and that the project root is on PYTHONPATH.")
    PointNet = None

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Device for PyTorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ASL character to integer mapping
char2int = {
    "A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "K":9, "L":10,
    "M":11, "N":12, "O":13, "P":14, "Q":15, "R":16, "S":17, "T":18, "U":19,
    "V":20, "W":21, "X":22, "Y":23
}

ACTION_LABELS = {
    'advance': "A : Advance",
    'backup': "B : Back-Up",
    'left': "L : Left",
    'right': "R : Right",
    'up': "U : Up",
    'down': "Y : Down",
    'home': "H : Home",
    'move_to': "Move To Position",
    'close_gripper': "A : Close Gripper",
    'open_gripper': "B : Open Gripper",
}

LEFT_GESTURE_TO_ACTION = {
    'A': 'advance',
    'B': 'backup',
    'L': 'left',
    'R': 'right',
    'U': 'up',
    'Y': 'down',
    'H': 'home',
}

RIGHT_GESTURE_TO_ACTION = {
    'A': 'close_gripper',
    'B': 'open_gripper',
}

IOTC_COMMAND_TO_ACTION = {
    'move_to': 'move_to',
    'move_to_home': 'home',
    'open_gripper': 'open_gripper',
    'close_gripper': 'close_gripper',
    'move_forward': 'advance',
    'move_backward': 'backup',
    'move_left': 'left',
    'move_right': 'right',
    'move_up': 'up',
    'move_down': 'down',
}

# Text overlay parameters
scale = 1.0
text_fontType = cv2.FONT_HERSHEY_SIMPLEX
text_fontSize = 0.75 * scale
text_color = (255, 0, 0)
text_lineSize = max(1, int(2 * scale))
text_lineType = cv2.LINE_AA
text_x = int(10 * scale)
text_y = int(30 * scale)


iotc_command_queue = deque()
command_queue_lock = threading.Lock()
iotc_publisher = None
iotc_warning_flags = {
    'sdk_missing': False,
    'not_connected': False,
}


def iotc_on_command(msg):
    print(f"Received IoTConnect command: {msg.command_name} args={msg.command_args} ack_id={msg.ack_id}")
    with command_queue_lock:
        iotc_command_queue.append(msg)


def iotc_on_disconnect(reason: str, disconnected_from_server: bool):
    print(f"IoTConnect disconnected ({'server' if disconnected_from_server else 'client'}): {reason}")


def _init_iotconnect_client():
    global iotc_publisher
    iotc_publisher = None
    if not IOTC_AVAILABLE:
        if not iotc_warning_flags['sdk_missing']:
            print('Warning: IoTConnect SDK not installed.')
            iotc_warning_flags['sdk_missing'] = True
        return

    try:
        device_config_path = 'iotcDeviceConfig.json'
        if not os.path.exists(device_config_path):
            print(f"Warning: IoTConnect device config file not found: {device_config_path}")
            return

        device_cert_path = 'device-cert.pem'
        device_pkey_path = 'device-pkey.pem'
        config = DeviceConfig.from_iotc_device_config_json_file(
            device_config_json_path=device_config_path,
            device_cert_path=device_cert_path,
            device_pkey_path=device_pkey_path
        )
       
        print('Device Config', config)
        callbacks = Callbacks(command_cb=iotc_on_command, disconnected_cb=iotc_on_disconnect)
        iotc_publisher = Client(config=config, callbacks=callbacks)
        iotc_publisher.connect()
        iotc_warning_flags['not_connected'] = False
        print('IOTCONNECT client connected')
    except Exception as e:
        iotc_warning_flags['not_connected'] = True
        print(f'Warning: Failed to initialize IoTConnect client: {e}.')
        iotc_publisher = None


def _parse_move_to_positions(command_args):
    if isinstance(command_args, dict):
        if 'positions' in command_args:
            raw_positions = command_args['positions']
        elif 'position' in command_args:
            raw_positions = command_args['position']
        else:
            raw_positions = ','.join(str(value) for value in command_args.values())
    else:
        raw_positions = command_args

    if isinstance(raw_positions, str):
        tokens = [token.strip() for token in raw_positions.split(',') if token.strip()]
    elif isinstance(raw_positions, (list, tuple)):
        tokens = [str(token).strip() for token in raw_positions if str(token).strip()]
    else:
        raise ValueError('move_to command_args must be a comma-separated string or list of positions')

    if len(tokens) != 6:
        raise ValueError(f'move_to requires 6 servo positions, received {len(tokens)}')

    positions = []
    for servo_id, token in enumerate(tokens, start=1):
        positions.append([servo_id, clamp_position(int(token))])

    return positions


def process_iotconnect_commands(arm):
    if not IOTC_AVAILABLE or iotc_publisher is None or not iotc_publisher.is_connected():
        return

    with command_queue_lock:
        queued_commands = list(iotc_command_queue)
        iotc_command_queue.clear()

    for msg in queued_commands:
        command_name = getattr(msg, 'command_name', '').lower()
        command_args = getattr(msg, 'command_args', {}) or {}
        print(f"Processing IoTConnect command '{command_name}' with args={command_args}")
        ack_status = C2dAck.CMD_SUCCESS_WITH_ACK
        ack_message = 'Executed command'
        try:
            action_name = IOTC_COMMAND_TO_ACTION.get(command_name)
            if action_name is None:
                ack_status = C2dAck.CMD_FAILURE_WITH_ACK
                ack_message = f"Unknown command '{command_name}'"
                print(ack_message)
            else:
                action_detected = execute_arm_action(arm, action_name, command_args)
                print(f"Action: {action_detected}")
                send_telemetry(arm)
        except Exception as e:
            ack_status = C2dAck.CMD_FAILURE_WITH_ACK
            ack_message = f"Command failed: {e}"
            print(ack_message)
        try:
            iotc_publisher.send_command_ack(msg, ack_status, ack_message)
        except Exception as e:
            print(f"Warning: Failed to send IoTConnect command ack: {e}")


def _send_iotconnect_telemetry(payload: dict):
    if not IOTC_AVAILABLE:
        if not iotc_warning_flags['sdk_missing']:
            print('Warning: IoTConnect SDK not installed.')
            iotc_warning_flags['sdk_missing'] = True
        return

    if iotc_publisher is None or not iotc_publisher.is_connected():
        if not iotc_warning_flags['not_connected']:
            print('Warning: IoTConnect client not connected.')
            iotc_warning_flags['not_connected'] = True
        return

    try:
        iotc_publisher.send_telemetry(payload)
    except Exception as e:
        print(f'IoTConnect telemetry publish error: {e}')


def send_telemetry(arm):
    telemetry = {
        'gripper': arm.getPosition(1),
        'wrist_roll': arm.getPosition(2),
        'wrist_flex': arm.getPosition(3),
        'elbow_flex': arm.getPosition(4),
        'shoulder_lift': arm.getPosition(5),
        'shoulder_pan': arm.getPosition(6),
        'systemdata': asdict(systemdata.collect_data())
    }

    message = json.dumps(telemetry)
    print(f"IoTConnect Telemetry: {message}")
    _send_iotconnect_telemetry(telemetry)


def clamp_position(value, minimum=0, maximum=1000):
    return max(minimum, min(maximum, int(value)))


def execute_arm_action(arm, action_name, command_args=None):
    if action_name == 'home':
        positions = [[1, 500], [2, 500], [3, 500], [4, 500], [5, 500], [6, 500]]
        arm.setPosition(positions, duration=1500, wait=True)
    elif action_name == 'move_to':
        positions = _parse_move_to_positions(command_args)
        arm.setPosition(positions, duration=1500, wait=True)
    elif action_name == 'open_gripper':
        arm.setPosition(1, clamp_position(arm.getPosition(1) - 100), duration=1000, wait=True)
    elif action_name == 'close_gripper':
        arm.setPosition(1, clamp_position(arm.getPosition(1) + 100), duration=1000, wait=True)
    elif action_name == 'advance':
        arm.setPosition(5, clamp_position(arm.getPosition(5) + 100), duration=1500, wait=True)
    elif action_name == 'backup':
        arm.setPosition(5, clamp_position(arm.getPosition(5) - 100), duration=1500, wait=True)
    elif action_name == 'left':
        arm.setPosition(6, clamp_position(arm.getPosition(6) - 100), duration=1500, wait=True)
    elif action_name == 'right':
        arm.setPosition(6, clamp_position(arm.getPosition(6) + 100), duration=1500, wait=True)
    elif action_name == 'up':
        arm.setPosition(4, clamp_position(arm.getPosition(4) - 100), duration=1500, wait=True)
    elif action_name == 'down':
        arm.setPosition(4, clamp_position(arm.getPosition(4) + 100), duration=1500, wait=True)
    else:
        raise ValueError(f"Unknown action '{action_name}'")

    return ACTION_LABELS[action_name]


def predict_asl_gesture(arm=None):
    """Detect ASL gestures from camera feed"""
    model_name = 'point_net_1.pth'
    model_path = './model'

    if PointNet is None:
        print("Error: PointNet model not available")
        return None

    try:
        model = torch.load(os.path.join(model_path, model_name), weights_only=False, map_location=device)
        print("PointNet model loaded successfully")
    except FileNotFoundError:
        print(f"Error: Model file {model_name} not found in {model_path}")
        return None

    input_video = 2
    cap = cv2.VideoCapture(input_video)
    frame_width = 640
    frame_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    print(f"[INFO] Camera input: {input_video} ({frame_width}, {frame_height})")

    last_gesture = None
    gesture_cooldown = 0

    while True:
        flag, frame = cap.read()
        if not flag:
            print("[ERROR] Camera read failed!")
            break

        cv2_image = cv2.flip(frame.copy(), 1)
        results = hands.process(cv2_image)

        image_height, image_width, _ = cv2_image.shape
        annotated_image = cv2_image.copy()
        current_gesture = None

        if results.multi_hand_landmarks:
            print(f"Detected Hands: {len(results.multi_hand_landmarks)}")

            for hand_id in range(len(results.multi_hand_landmarks)):
                hand_handedness = results.multi_handedness[hand_id]
                hand_landmarks = results.multi_hand_landmarks[hand_id]
                handedness = hand_handedness.classification[0].label
                print(f'[INFO] Detected Hand: "{handedness}"')

                hand_x = text_x
                hand_y = text_y
                hand_color = text_color

                if handedness == "Left":
                    hand_x = 10
                    hand_y = 30
                    hand_color = (0, 255, 0)
                    hand_msg = 'LEFT='
                elif handedness == "Right":
                    hand_x = image_width - 128
                    hand_y = 30
                    hand_color = (0, 0, 255)
                    hand_msg = 'RIGHT='

                points_raw = []
                for lm in hand_landmarks.landmark:
                    points_raw.append([lm.x, lm.y, lm.z])
                points_raw = np.array(points_raw)

                points_norm = points_raw.copy()
                min_x = np.min(points_raw[:, 0])
                max_x = np.max(points_raw[:, 0])
                min_y = np.min(points_raw[:, 1])
                max_y = np.max(points_raw[:, 1])

                for i in range(len(points_raw)):
                    if max_x > min_x:
                        points_norm[i][0] = (points_norm[i][0] - min_x) / (max_x - min_x)
                    if max_y > min_y:
                        points_norm[i][1] = (points_norm[i][1] - min_y) / (max_y - min_y)
                    if handedness == "Right":
                        points_norm[i][0] = 1.0 - points_norm[i][0]

                for hc in mp_hands.HAND_CONNECTIONS:
                    cv2.line(annotated_image,
                             (int(points_raw[hc[0]][0] * image_width),
                              int(points_raw[hc[0]][1] * image_height)),
                             (int(points_raw[hc[1]][0] * image_width),
                              int(points_raw[hc[1]][1] * image_height)),
                             hand_color, 2)

                try:
                    pointst = torch.tensor([points_norm]).float().to(device)
                    label = model(pointst)
                    label = label.detach().cpu().numpy()
                    asl_id = np.argmax(label)
                    confidence = float(label[0][asl_id])
                    asl_sign = list(char2int.keys())[list(char2int.values()).index(asl_id)]

                    asl_text = hand_msg + asl_sign
                    cv2.putText(annotated_image, asl_text, (hand_x, hand_y),
                               text_fontType, text_fontSize, hand_color, text_lineSize, text_lineType)

                    action_detected = None
                    action_name = None
                    if handedness == "Left":
                        action_name = LEFT_GESTURE_TO_ACTION.get(asl_sign)
                    elif handedness == "Right":
                        action_name = RIGHT_GESTURE_TO_ACTION.get(asl_sign)
                    if action_name:
                        action_detected = ACTION_LABELS[action_name]

                    if action_detected:
                        action_text = '[' + action_detected + ']'
                        cv2.putText(annotated_image, action_text,
                                    (hand_x, hand_y * 2),
                                    text_fontType, text_fontSize,
                                    hand_color, text_lineSize, text_lineType)
                        print(f"Detected ASL: {asl_sign}, Action: {action_detected}")

                    current_gesture = (handedness, asl_sign, confidence, points_raw)

                except Exception as e:
                    print(f"[ERROR] Exception during ASL classification: {e}")

        cv2.imshow("IOTCONNECT XArm Control", annotated_image)
        key = cv2.waitKey(10)

        if arm is not None:
            process_iotconnect_commands(arm)

        if key == 27 or key == 113:
            break

        if gesture_cooldown > 0:
            gesture_cooldown -= 1
        elif current_gesture:
            gesture_signature = current_gesture[:2]
            if gesture_signature != last_gesture:
                last_gesture = gesture_signature
                gesture_cooldown = 1
                control_arm_with_gesture(arm, current_gesture[0], current_gesture[1])

    cap.release()
    cv2.destroyAllWindows()
    return None


def control_arm_with_gesture(arm, handedness, gesture):
    """Control XArm based on detected ASL gesture and handedness"""
    try:
        action_name = None

        if handedness == 'Left':
            action_name = LEFT_GESTURE_TO_ACTION.get(gesture)
        elif handedness == 'Right':
            action_name = RIGHT_GESTURE_TO_ACTION.get(gesture)

        else:
            print(f"Unknown handedness '{handedness}' for gesture '{gesture}'")
            return

        if action_name:
            action_detected = execute_arm_action(arm, action_name)
            print(f"Action: {action_detected}")
            send_telemetry(arm)

    except Exception as e:
        print(f"Error controlling arm: {e}")


def main():
    try:
        print("Connecting to XArm 1S...")
        arm = xarm.Controller('USB')
        print("Connected to XArm 1S successfully!")

        print("Connecting to IoTConnect...")
        _init_iotconnect_client()
        if iotc_publisher is not None and iotc_publisher.is_connected():
            print("Connected to IoTConnect successfully!")
        else:
            print("IoTConnect is unavailable; running without cloud connectivity.")

        print("Initializing to home position...")
        home_positions = [[1,500], [2,500], [3,500], [4,500], [5,500], [6,500]]  # Servos 1-6
        arm.setPosition(home_positions, duration=2000, wait=True)
        print("Home position reached!")

        print("Starting ASL gesture control...")
        predict_asl_gesture(arm)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure XArm 1S is connected via USB and powered on.")
        print("Also ensure camera is available and PointNet model files exist.")

    finally:
        try:
            print("Turning off servos...")
            arm.servoOff()
        except:
            pass
        if iotc_publisher is not None:
            try:
                iotc_publisher.disconnect()
            except Exception:
                pass

if __name__ == "__main__":
    main()
