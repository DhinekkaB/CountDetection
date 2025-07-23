import cv2
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import firebase_admin
from firebase_admin import credentials, db

# Firebase Setup
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://count-detect-default-rtdb.asia-southeast1.firebasedatabase.app/'
})
firebase_ref = db.reference('/')

# Load YOLO model
model = YOLO("yolov8n.pt")  # or your custom trained model
cap = cv2.VideoCapture("http://10.157.212.87:81/stream")  # ESP32-CAM stream

# Tracker
tracker = DeepSort(max_age=30)

# Count tracking
present_ids = {}
entry_count = 0
exit_count = 0

# Timers
entry_threshold = 3  # seconds
exit_threshold = 3

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    results = model(frame)[0]
    detections = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if int(class_id) != 0 or score < 0.5:  # Class 0 = person
            continue
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, int(class_id)))

    tracks = tracker.update_tracks(detections, frame=frame)

    current_frame_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        current_frame_ids.add(track_id)
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # If ID seen for first time
        if track_id not in present_ids:
            present_ids[track_id] = {"entry_time": time.time(), "counted": False, "last_seen": time.time()}
        else:
            present_ids[track_id]["last_seen"] = time.time()

        # Check for confirmed presence
        if not present_ids[track_id]["counted"] and time.time() - present_ids[track_id]["entry_time"] >= entry_threshold:
            entry_count += 1
            present_ids[track_id]["counted"] = True
            print(f"✅ Entry Detected | ID: {track_id} | Total Entered: {entry_count}")

    # Check for exits (IDs no longer in frame)
    for tid in list(present_ids.keys()):
        if tid not in current_frame_ids:
            if time.time() - present_ids[tid]["last_seen"] > exit_threshold:
                if present_ids[tid]["counted"]:
                    exit_count += 1
                    print(f"❌ Exit Detected | ID: {tid} | Total Exited: {exit_count}")
                del present_ids[tid]

    # Update Firebase
    firebase_ref.update({
        "entry_count": entry_count,
        "exit_count": exit_count,
        "current_inside": entry_count - exit_count
    })

    cv2.imshow("SmartBus Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
