# Gesture-based-diya-lighting
developed this code just for fun purpose
# Virtual Diya - README

This documentation is for `diya.py`, a Python script that simulates a virtual Diya (oil lamp) using computer vision hand tracking.

## Description
The Virtual Diya application allows you to interact with a digital oil lamp using hand gestures. You can light a virtual matchstick, ignite the Diya, and interact with the flame using a webcam.

## Prerequisites & Installation

To run this script, you need Python installed on your system along with the following libraries:

### Required PIP Packages
Install the necessary dependencies using pip:

```bash
pip install opencv-python mediapipe pygame numpy
```

### Dependencies Breakdown
- **opencv-python**: Used for capturing video from the webcam and image processing.
- **mediapipe**: Used for robust hand tracking and landmark detection.
- **pygame**: Used for rendering the graphical interface and handling the main game loop.
- **numpy**: Used for numerical operations and array manipluation.

## How to Run

1. Navigate to the directory containing `diya.py`:
   ```bash
   cd src/ui
   ```
2. Run the script:
   ```bash
   python diya.py
   ```

## Controls & Interaction

- **Lighting the Match**: 
  - Use your **Left Hand** (physically left, might appear mirrored).
  - Perform a "rubbing" gesture with your Thumb and Middle finger to ignite the virtual match held by your index finger.
- **Lighting the Diya**:
  - Once the match is lit, move it close to the Diya's wick to light it.
- **Extinguishing**:
  - Perform a "pinch" gesture (Thumb + Index finger touching) near the flame to extinguish it.
- **Reset**:
  - Press the **ESC** key or click the on-screen **Stop Button** to reset the Diya and Matchstick.
- **Quit**:
  - Press **'q'** or close the window to exit the application.

## Troubleshooting
- **Webcam Issues**: Use `cv2.VideoCapture(0)` (default) or `cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)` (macOS specific) if the camera doesn't open.
- **Performance**: Ensure your room is well-lit for better hand tracking accuracy.
