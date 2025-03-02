import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import logging
import qrcode

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

device = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = YOLO("yolov8m-pose.pt").to(device)
logging.info(f"YOLOv8-Pose model loaded on {device}.")

# Load image function
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

# Load necessary images
tree_img = load_image(r"C:\Users\VICTUS\OpenPose_Intern_Project\tree-removebg-preview.png")
apple_img = load_image(r"C:\Users\VICTUS\OpenPose_Intern_Project\Apple-removebg-preview.png", (30, 30))
background_img = load_image(r"C:\Users\VICTUS\OpenPose_Intern_Project\OpenPoseProject\OpenPoseProjectV2\milkyway-8190232_640.jpg", (1280, 720))

fruit_positions = [(np.random.randint(400, 900), np.random.randint(100, 300)) for _ in range(15)]
remaining_fruits = fruit_positions.copy()
fruits_plucked = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

def generate_qr():
    qr = qrcode.make("https://www.kidocode.com/trial-class")
    qr = np.array(qr.convert('RGB'))
    qr = cv2.cvtColor(qr, cv2.COLOR_RGB2BGRA)
    qr = cv2.resize(qr, (400, 400))
    return qr

def overlay_transparent(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (1 - alpha) * bg[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]
    return bg

def draw_rainbow_text(image, text, position, font, scale, thickness):
    x, y = position
    rainbow_colors = [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (75, 0, 130), (148, 0, 211)]
    for i, char in enumerate(text):
        color = rainbow_colors[i % len(rainbow_colors)]
        cv2.putText(image, char, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)
        (w, _), _ = cv2.getTextSize(char, font, scale, thickness)
        x += w

def process_frame():
    global fruits_plucked
    start_time = time.time()
    time_limit = 120
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        elapsed_time = int(time.time() - start_time)
        remaining_time = max(0, time_limit - elapsed_time)
        
        canvas = background_img.copy()
        canvas = overlay_transparent(canvas, tree_img, 400, 50)
        
        for fx, fy in remaining_fruits:
            canvas = overlay_transparent(canvas, apple_img, fx, fy)
        
        with torch.no_grad():
            results = yolo_model(frame, imgsz=640, conf=0.5, iou=0.45)[0]
        
        if results.keypoints is not None:
            for kp in results.keypoints.xy.cpu().numpy():
                if kp.shape[0] < 17:
                    continue
                
                left_wrist, right_wrist = kp[9], kp[10]
                hands = [left_wrist, right_wrist]
                
                new_fruits = [
                    fruit for fruit in remaining_fruits
                    if not any(
                        np.linalg.norm(np.array(fruit) - hand) < 40
                        for hand in hands if hand[0] > 0 and hand[1] > 0
                    )
                ]
                
                fruits_plucked += len(remaining_fruits) - len(new_fruits)
                remaining_fruits[:] = new_fruits
                
                body_connections = [
                    (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 12)
                ]
                
                for p1, p2 in body_connections:
                    if (kp[[p1, p2]] > 0).all():
                        pt1, pt2 = tuple(map(int, kp[p1])), tuple(map(int, kp[p2]))
                        color = tuple(np.random.randint(0, 255, 3).tolist())
                        cv2.line(canvas, pt1, pt2, color, 5)
                
                for hand in hands:
                    if hand[0] > 0 and hand[1] > 0:
                        cv2.circle(canvas, (int(hand[0]), int(hand[1])), 10, (0, 255, 0), -1)
        
        cv2.putText(canvas, f"Time Left: {remaining_time}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(canvas, f"Fruits Plucked: {fruits_plucked} / 15", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Fruit Plucking Game", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q') or remaining_time == 0 or len(remaining_fruits) == 0:
            break
    
    qr_img = generate_qr()
    canvas[160:560, 440:840] = qr_img
    draw_rainbow_text(canvas, "Congratulations!", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    draw_rainbow_text(canvas, "Scan for a KidoCode discount", (400, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    
    cv2.imshow("Game Over", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

process_frame()