# FruitGameOpenPose

## Description
FruitGameOpenPose is an interactive game that utilizes YOLOv8 Pose estimation to detect hand movements via a webcam. Players can "pluck" virtual fruits from a tree by moving their hands close to the fruit, as detected by the pose estimation model. The game ends when all fruits are plucked or the time runs out, and a QR code is displayed as a reward.

## Features
- **Real-time Hand Tracking**: Uses YOLOv8 Pose model to detect hands and track movements.
- **Interactive Fruit Plucking**: Detects when hands are near fruits and removes them from the screen.
- **Timer-Based Challenge**: The player has 120 seconds to pluck all the fruits.
- **QR Code Generation**: A scannable QR code is displayed at the end of the game for additional rewards.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/FruitGameOpenPose.git
cd FruitGameOpenPose
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```
If `requirements.txt` is missing, install these manually:
```bash
pip install opencv-python torch numpy ultralytics qrcode
```

### 3. Download YOLOv8 Pose Model
You need the YOLOv8 model file to run the pose estimation:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt
```

### 4. Prepare Game Assets
Ensure the following image files are available:
- `tree-removebg-preview.png` (Tree image)
- `Apple-removebg-preview.png` (Apple image)
- `milkyway-8190232_640.jpg` (Background image)

These files should be placed in the same directory as the script, or update their paths in the script accordingly.

## Usage

### 1. Start the Game
Run the script to start the game:
```bash
python QR_Code_FruitGame.py
```

### 2. Gameplay Instructions
- The webcam will activate and display a tree with apples.
- Move your hands in front of the camera to "pluck" the apples.
- Each successfully plucked fruit is removed from the screen.
- The game ends after 120 seconds or when all fruits are plucked.
- At the end of the game, a QR code appears, which can be scanned for a special reward.

### 3. Exiting the Game
- Press the `q` key at any time to exit the game.

## Dependencies
- Python 3.8+
- OpenCV (cv2) for real-time video processing
- PyTorch for running YOLOv8 Pose model
- Numpy for numerical operations
- YOLOv8 Pose model from Ultralytics
- qrcode library for QR code generation


## Computer Specification
- GPU Nvidia 3050 Ti (Please use this gpu and not the in-build gpu for smoother experience)

## Contribution
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new branch with a descriptive name (e.g., `feature-hand-detection-improvement`).
3. Make your changes and test them.
4. Submit a pull request for review.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

