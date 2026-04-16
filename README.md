# TRIA Vision AI Kit 6490 + /IOTCONNECT XArm Gesture Control

This project showcases the **TRIA Vision AI Kit 6490** running **/IOTCONNECT** integration with Hiwonder XArm 1S robotic arm controlled via American Sign Language (ASL) gestures. This project demonstrates real-time AI-powered gesture recognition on the TRIA board with cloud telemetry transmission and remote command execution through /IOTCONNECT's IoT platform.

## Key Features

- **TRIA Vision AI Kit 6490**: Qualcomm QCS6490-powered edge AI platform for real-time gesture recognition
- **IoTConnect Cloud Integration**: Real-time telemetry transmission and remote command execution
- **ASL Gesture Control**: AI-powered American Sign Language recognition using MediaPipe + PointNet
- **Robotic Arm Control**: Hiwonder XArm 1S with 6-DOF movement and gripper control
- **Edge-to-Cloud Architecture**: Local AI inference on TRIA board with cloud connectivity via IoTConnect

## Setup

This project is designed to run on the **TRIA Vision AI Kit 6490** with **/IOTCONNECT** cloud integration, creating a powerful edge-to-cloud AI robotics solution.

### Why TRIA Vision AI Kit 6490 + /IOTCONNECT?

- **TRIA Vision AI Kit 6490**: Energy-efficient Qualcomm QCS6490 SOC with multi-camera support, perfect for real-time AI inference
- **IoTConnect Integration**: Seamless cloud connectivity for telemetry, remote monitoring, and command execution
- **Edge AI**: Run neural network inference locally on TRIA board while streaming results to the cloud
- **Industrial IoT**: Enterprise-grade IoT platform for robotics and automation applications

### Hardware Requirements

- **[TRIA Vision AI-KIT 6490](https://www.newark.com/avnet/sm2-sk-qcs6490-ep6-kit001/dev-kit-64bit-arm-cortex-a55-a78/dp/51AM9843)** - Main compute platform with Qualcomm QCS6490 SOC
- **[HiWonder xArm1S](https://www.amazon.com/LewanSoul-Programmable-Feedback-Parameter-Programming/dp/B0CHY63V9P?th=1)** - Robotic Arm connected vis USB
- USB-C Cable for flashing and USB-ADB debug (included with kit)
- USB-C 12VDC Power Supply and Cable (included with kit)
- Ethernet Cable (not included)
- Logitech Webcam or camera device for hand tracking
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


## Troubleshooting

- **XArm Connection Issues**: Ensure XArm 1S is connected to TRIA board's USB ports and powered on
- **HIDAPI Issues**: The xarm library uses hidapi for USB communication - ensure proper USB device permissions
- **Camera Not Detected**: Verify camera is connected to TRIA board and accessible
- **Model Loading Errors**: Ensure model files are downloaded and accessible in the `model/` directory
- **IoTConnect Connection**: Check ethernet connectivity and device onboarding status
- **Permission Issues**: Run applications with appropriate permissions for USB/serial access

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
- [/IOTCONNECT SDK](https://github.com/avnet-iotconnect/avnet-iotconnect-python-sdk) - IoTConnect Python SDK for cloud connectivity
- [/IOTCONNECT Device Onboarding](https://github.com/avnet-iotconnect/iotc-python-lite-sdk-demos/blob/main/common/general-guides/UI-ONBOARD.md) - Step-by-step device registration guide
- [/IOTCONNECT Overview](https://www.iotconnect.io/) - Enterprise IoT platform information

### Robotics & AI Components
- [xArm Python Library](https://github.com/xArm-Developer/xArm-Python-SDK) - Official Python SDK for Hiwonder XArm robotic arms
- [ASL MediaPipe PointNet](https://github.com/AlbertaBeef/asl_mediapipe_pointnet) - ASL gesture recognition using MediaPipe and PointNet neural network