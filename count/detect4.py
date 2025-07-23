import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import firebase_admin
from firebase_admin import credentials, db
import time

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")  # Replace with your Firebase Admin SDK file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://count-detect-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your URL
})

# Reference to Firebase node
firebase_ref = db.reference('people_count')

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Your custom trained model
cap = cv2.VideoCapture("hhttp://10.157.212.87:81/stream")  # ESP32-CAM stream

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Parameters
counted_ids = set()
total_count = 0

# Define line position (y-coordinate of the horizontal line)
line_position = 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_id = int(class_id)

        if score < 0.5:
            continue

        # Only track "person" class (or your box class if custom trained)
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, class_id))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        center_y = (y1 + y2) // 2

        # Count only if this ID hasn't been counted yet and it crosses the line
        if track_id not in counted_ids and center_y < line_position + 10 and center_y > line_position - 10:
            total_count += 1
            counted_ids.add(track_id)
            print(f"Person Counted: ID {track_id} | Total: {total_count}")
            firebase_ref.set(total_count)

    # Draw line
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)

    cv2.imshow("Smart Bus Person Counter", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
