import cv2
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
model = YOLO("yolov8n.pt")  # Replace with your custom trained model if available
print("✅ YOLOv8 model loaded")

# Stream Setup
stream_url = "http://10.157.212.87:81/stream"
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("❌ Failed to open ESP32-CAM stream")
    exit(1)
print("📹 ESP32-CAM stream connected")

# Tracker
tracker = DeepSort(max_age=30)

# Tracking state
present_ids = {}
entry_count = 0
exit_count = 0

# Timers (in seconds)
entry_threshold = 2.5
exit_threshold = 2.5

# Frame skip to reduce load
frame_skip = 3
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed")
        continue

    frame_id += 1
    if frame_id % frame_skip != 0:
        continue

    results = model(frame, verbose=False)[0]
    detections = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if int(class_id) != 0 or score < 0.5:
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
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw visual
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Entry logic
        if track_id not in present_ids:
            present_ids[track_id] = {
                "entry_time": time.time(),
                "last_seen": time.time(),
                "counted": False
            }
        else:
            present_ids[track_id]["last_seen"] = time.time()

        if not present_ids[track_id]["counted"] and (time.time() - present_ids[track_id]["entry_time"] >= entry_threshold):
            entry_count += 1
            present_ids[track_id]["counted"] = True
            print(f"✅ Entry | ID: {track_id} | IN: {entry_count}")

    # Exit check
    for tid in list(present_ids.keys()):
        if tid not in current_frame_ids:
            if time.time() - present_ids[tid]["last_seen"] > exit_threshold:
                if present_ids[tid]["counted"]:
                    exit_count += 1
                    print(f"❌ Exit | ID: {tid} | OUT: {exit_count}")
                del present_ids[tid]

    # Firebase update
    firebase_ref.update({
        "entry_count": entry_count,
        "exit_count": exit_count,
        "current_inside": max(0, entry_count - exit_count)
    })

    # UI display
    cv2.putText(frame, f'IN: {entry_count}  OUT: {exit_count}  NOW: {max(0, entry_count - exit_count)}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("SmartBus Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
