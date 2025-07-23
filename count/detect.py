from ultralytics import YOLO
import cv2
import firebase_admin
from firebase_admin import credentials, db
import time

# ---------------- FIREBASE SETUP ---------------- #
cred = credentials.Certificate("serviceAccountKey.json")  # your JSON key
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://count-detect-default-rtdb.asia-southeast1.firebasedatabase.app/'  # replace with your Firebase URL
})
ref = db.reference('/person_count')

# ---------------- VARIABLES ---------------- #
model = YOLO("yolov8n.pt")  # your YOLOv8 trained model
cam_url = "http://192.168.1.3:81/stream"  # replace with your ESP32-CAM stream URL

cap = cv2.VideoCapture(cam_url)

prev_ids = set()
person_count = 0

def update_count(count):
    ref.set(count)
    print(f"[FIREBASE] Updated count: {count}")

# ---------------- MAIN LOOP ---------------- #
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5)
    frame = results[0].plot()

    current_ids = set()
    for box in results[0].boxes:
        if box.id is not None:
            current_ids.add(int(box.id.item()))

    # Entry logic
    new_entries = current_ids - prev_ids
    if new_entries:
        person_count += len(new_entries)
        update_count(person_count)

    prev_ids = current_ids

    cv2.imshow("Person Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
