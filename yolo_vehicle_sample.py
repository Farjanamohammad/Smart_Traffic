import cv2
import os
from ultralytics import YOLO

# ✅ Load pre-trained YOLOv8 model (vehicle detection)
model = YOLO(r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\yolov8.pt")  # Replace with your custom weights if needed

# ✅ Load sample video
video_path = r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\videos\test3.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# ✅ Create output folder
os.makedirs("media/sample_outputs", exist_ok=True)

frame_saved = False
frame_number_to_save = 50  # Save the 50th frame (you can change it)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # ✅ Perform detection
    results = model(frame)
    annotated_frame = results[0].plot()  # Draw bounding boxes

    # ✅ Save only one sample frame
    if frame_number == frame_number_to_save and not frame_saved:
        output_path = "media/sample_outputs/yolov8_vehicle_detection_output.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"✅ Sample detection image saved at: {output_path}")
        frame_saved = True

    # ✅ Optional: Show detection
    cv2.imshow("YOLOv8 Vehicle Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
