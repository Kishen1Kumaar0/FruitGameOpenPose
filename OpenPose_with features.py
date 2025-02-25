import cv2
import numpy as np
import torch
from ultralytics import YOLO
import threading
import time
import logging
import qrcode

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Enable GPU for YOLOv8 Pose
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8m-pose.pt").to(device)
logging.info(f"YOLOv8-Pose model loaded on {device}.")

# Load Images
def load_image(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        logging.error(f"Failed to load image: {path}")
        return None
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if size:
        img = cv2.resize(img, size)
    return img

tree_img = load_image(r"C:\Users\VICTUS\OpenPose_Intern_Project\tree-removebg-preview.png")
apple_img = load_image(r"C:\Users\VICTUS\OpenPose_Intern_Project\Apple-removebg-preview.png", (30, 30))

# Generate Fruits Positions
fruit_positions = [(np.random.randint(300, 900), np.random.randint(50, 250)) for _ in range(15)]
remaining_fruits = fruit_positions.copy()

# Capture Frames
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Overlay Function
def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (1 - alpha) * background[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]
    return background

# QR Code Generation
def generate_qr():
    qr = qrcode.make("https://www.kidocode.com/")
    qr = np.array(qr.convert('RGB'))
    qr = cv2.resize(qr, (200, 200))
    return qr

# Pose Detection & Fruit Plucking
def process_frame():
    global remaining_fruits
    plucked_fruits = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        tree_canvas = np.ones_like(frame) * 255
        tree_canvas = overlay_transparent(tree_canvas, cv2.resize(tree_img, (1280, 720)), 0, 0)
        
        # Display remaining fruits
        for fx, fy in remaining_fruits:
            tree_canvas = overlay_transparent(tree_canvas, apple_img, fx, fy)
        
        # Display fruit counters
        cv2.putText(tree_canvas, f"Fruits on Tree: {len(remaining_fruits)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(tree_canvas, f"Fruits Plucked: {plucked_fruits}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # YOLO Pose Detection
        with torch.no_grad():
            results = yolo_model(frame, imgsz=640, conf=0.5, iou=0.45)[0]
        
        best_person = None
        max_conf = 0
        
        if results.keypoints is not None:
            for kp in results.keypoints.xy.cpu().numpy():
                confidence = np.sum(kp[:, 0] > 0)
                if confidence > max_conf:
                    max_conf = confidence
                    best_person = kp
        
        # Draw the best person only
        if best_person is not None:
            hands = [best_person[9], best_person[10]]  # Wrist keypoints
            
            remaining_fruits = [fruit for fruit in remaining_fruits if not any(
                np.linalg.norm(np.array(fruit) - hand) < 40 for hand in hands if hand[0] > 0
            )]
            
            plucked_fruits = 15 - len(remaining_fruits)
            
            # Draw skeleton with unique colors
            for i, kp in enumerate(best_person):
                if kp[0] > 0 and kp[1] > 0:
                    cv2.circle(tree_canvas, (int(kp[0]), int(kp[1])), 10, (255, 0, 0), -1)
            
            for (kp1, kp2) in [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
                if best_person[kp1][0] > 0 and best_person[kp2][0] > 0:
                    p1 = (int(best_person[kp1][0]), int(best_person[kp1][1]))
                    p2 = (int(best_person[kp2][0]), int(best_person[kp2][1]))
                    cv2.line(tree_canvas, p1, p2, (0, 255, 0), 5)
        
        # Display Game
        cv2.imshow("Game", cv2.cvtColor(tree_canvas, cv2.COLOR_BGRA2BGR))
        
        # End Game When All Fruits Are Plucked
        if len(remaining_fruits) == 0:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('x'):
            return
    
    # Display Game Over & QR Code
    qr_img = generate_qr()
    game_over_screen = np.ones((300, 400, 3), dtype=np.uint8) * 255
    game_over_screen[50:250, 100:300] = qr_img
    cv2.putText(game_over_screen, "Game Over", (120, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(game_over_screen, "Winner gets a Kidocode discount", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow("Game Over", game_over_screen)
    cv2.waitKey(100000)
    
    cv2.destroyAllWindows()
    cap.release()

# Start the game thread
game_thread = threading.Thread(target=process_frame, daemon=True)
game_thread.start()
game_thread.join()
logging.info("Game Over. Winner gets a Kidocode discount!")
