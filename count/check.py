import cv2
import time
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db

# === FIREBASE SETUP ===
cred = credentials.Certificate("serviceAccountKey.json")  # Your Firebase admin SDK JSON
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://count-detect-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your Firebase DB URL
})
firebase_ref = db.reference('bus1/person_count')

# === LOAD YOLOv8 MODEL ===
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")  # Replace with your trained model
print("Model loaded.")

# === CONNECT TO ESP32-CAM STREAM ===
ESP32_URL = 'http://10.157.212.87:81/stream'  # Update with your ESP32-CAM stream URL
print(f"Connecting to MJPEG stream at {ESP32_URL} ...")
cap = cv2.VideoCapture(ESP32_URL)
if not cap.isOpened():
    print("Failed to open video stream.")
    exit()
print("Stream connected.")

# === TRACKING SETTINGS ===
prev_count = 0
frame_skip = 3  # Process every 3rd frame for performance
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_index += 1
    if frame_index % frame_skip != 0:
        continue  # Skip frame to reduce lag

    results = model(frame, verbose=False)[0]
    detections = results.boxes.data.tolist()

    person_count = 0
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        class_id = int(cls_id)
        if model.names[class_id] == 'person':  # Or your custom label
            person_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show count on frame
    cv2.putText(frame, f'Persons Detected: {person_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('ESP32-CAM YOLOv8 Detection', frame)

    # === Firebase Update Only If Count Changes ===
    if person_count != prev_count:
        firebase_ref.set(person_count)
        prev_count = person_count
        print(f'[Firebase] Updated person count to: {person_count}')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
