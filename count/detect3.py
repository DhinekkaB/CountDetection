import cv2
import torch
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# === 1. ESP32-CAM Stream Setup ===
ESP32_STREAM_URL = 'http://192.168.1.3:81/stream'  # Your ESP32-CAM IP

# === 2. Firebase Setup ===
FIREBASE_URL = 'https://count-detect-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your Firebase URL
cred = credentials.Certificate("serviceAccountKey.json")  # Your Firebase Admin SDK JSON
firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
count_ref = db.reference('person_count')
count_ref.set(0)

# === 3. Load YOLOv5 Custom Model ===
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.conf = 0.3

# === 4. Detection Logic Variables ===
line_position = 250  # virtual line Y-position
counted = False
person_count = 0

def is_crossing_line(center_y, line_y, threshold=15):
    return abs(center_y - line_y) < threshold

# === 5. Start Video Stream ===
cap = cv2.VideoCapture(ESP32_STREAM_URL + 'video')

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame from ESP32-CAM")
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    for *xyxy, conf, cls in detections:
        label = results.names[int(cls)]
        if label == 'person':
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            if is_crossing_line(cy, line_position):
                if not counted:
                    person_count += 1
                    count_ref.set(person_count)
                    print(f"✅ Count Updated: {person_count}")
                    counted = True
            else:
                counted = False

    # Draw line & count
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255,0,0), 2)
    cv2.putText(frame, f'Count: {person_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    cv2.imshow("YOLO Person Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
