
from ultralytics import YOLO
import cv2

# Load trained modelqqqqq
model = YOLO('runs/detect/train2/weights/best.pt')  # or your custom path

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict on frame
    results = model(frame, imgsz=640, conf=0.5)

    # Plot the results
    annotated_frame = results[0].plot()

    # Show the output
    cv2.imshow("Real-Time Waste Detection", annotated_frame)
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




