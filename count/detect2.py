import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# ESP32-CAM stream URL (corrected)
stream_url = 'http://10.157.212.87:81/stream'
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ Cannot open ESP32-CAM stream")
    exit()

print("✅ Stream opened successfully. Starting detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to decode frame")
        continue

    # Run YOLO inference
    results = model(frame)

    # Visualize detection
    annotated_frame = results[0].plot()
    cv2.imshow("ESP32-CAM Detection", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

