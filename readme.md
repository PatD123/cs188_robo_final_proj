# Multimodal Robot Control Simulation

A Python-based robot simulation that combines gesture recognition, speech-to-text (STT) commands, and robotic arm control using RoboSuite. Control a robotic arm using hand gestures for movement and voice commands for gripper control.

## Prerequisites

- Python 3.8 or higher
- Webcam (for gesture recognition)
- Microphone (for voice commands)
- [Ollama](https://ollama.ai/) with llama3.2 model installed

## Installation

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd cs188_robo_final_proj
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama and the required model**:
   ```bash
   # Install Ollama (visit https://ollama.ai/ for installation instructions)
   ollama pull llama3.2
   ```

5. **Verify MuJoCo installation**:
   The simulation requires MuJoCo physics engine, which should be installed with the requirements. If you encounter issues, visit [MuJoCo's official documentation](https://mujoco.org/).

## Usage

### Running the Simulation

1. **Start the simulation**:
   ```bash
   mjpython main.py
   ```

2. **Control the robot**:
   - **Hand Gestures**: Point your index finger in the direction you want the robot to move
     - Point up: Move robot UP
     - Point down: Move robot DOWN  
     - Point left: Move robot LEFT
     - Point right: Move robot RIGHT
   
   - **Voice Commands**: Speak clearly into your microphone 
     - "close" - Close gripper
     - "open" - Open gripper
     - "rotate right" - Rotate arm right
     - "rotate left" - Rotate arm left
     - "forward" - Move arm forward
     - "back" - Move arm backward
     - "increase speed" - Speed up movements
     - "decrease speed" - Slow down movements
     - "reset" - Reset to default settings
     - "stop" - Stop current action

### Gesture Recognition UI

To enable the gesture recognition UI window:
```python
gesture_recognizer = gestures.GestureRecognizer(show_UI=True)
```
Make sure this is set to False if running on MacOS

## Troubleshooting

### Common Issues

1. **OpenCV c++ exception**:
   - Make sure show_UI is set to false in the GestureRecognizer. MacOS cannot do multithreaded cv2 UI functions.

