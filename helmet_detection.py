# import cv2
# import os
# from ultralytics import YOLO

# # ✅ Load your custom helmet detection model
# helmet_model = YOLO(r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\best.pt")  # Change path if needed

# # ✅ Load a sample traffic video
# video_path = r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\videos\test3.mp4"  # Change to your test video path
# cap = cv2.VideoCapture(video_path)

# # ✅ Create output directory
# os.makedirs("media/sample_outputs", exist_ok=True)

# frame_saved = False
# frame_number_to_save = 60  # Save output at this frame (adjust if needed)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

#     # ✅ Detect helmets
#     results = helmet_model(frame)
#     annotated_frame = results[0].plot()  # Draw bounding boxes

#     # ✅ Save a sample frame with bounding boxes
#     if frame_number == frame_number_to_save and not frame_saved:
#         output_path = "media/sample_outputs/helmet_detection_output.jpg"
#         cv2.imwrite(output_path, annotated_frame)
#         print(f"✅ Helmet detection sample saved at: {output_path}")
#         frame_saved = True

#     # ✅ Optional: Display window
#     cv2.imshow("Helmet Detection Sample", cv2.resize(annotated_frame, (960, 540)))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# === Load YOLOv8 Helmet Model === #
helmet_model = YOLO(r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\best.pt")  # Replace with your model path

# === Load Your Video === #
video_path = r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\videos\test3.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# === Detection Counters === #
total_frames_checked = 0
helmet_detected = 0
no_helmet_detected = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames_checked += 1

    # Run helmet detection model
    results = helmet_model(frame)
    detected_classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []

    if 0 in detected_classes:
        helmet_detected += 1
    if 1 in detected_classes:
        no_helmet_detected += 1

cap.release()

# === Print Detection Summary === #
print("🎥 Helmet Detection Analysis from Video:")
print(f"📌 Total Frames Analyzed: {total_frames_checked}")
print(f"✅ Frames with Helmet Detected: {helmet_detected}")
print(f"🚫 Frames with No-Helmet Detected: {no_helmet_detected}")

# === Calculate Percentages === #
total_detections = helmet_detected + no_helmet_detected
helmet_percent = (helmet_detected / total_detections) * 100 if total_detections else 0
no_helmet_percent = (no_helmet_detected / total_detections) * 100 if total_detections else 0

print(f"\n📊 Estimated Helmet Rate: {helmet_percent:.2f}%")
print(f"📊 Estimated No-Helmet Violation Rate: {no_helmet_percent:.2f}%")

# === Plot and Save Bar Chart === #
labels = ['With Helmet', 'Without Helmet']
counts = [helmet_detected, no_helmet_detected]
colors = ['green', 'red']

plt.figure(figsize=(6, 5))
plt.bar(labels, counts, color=colors)
plt.title('Helmet Detection Summary from Video')
plt.xlabel('Detection Category')
plt.ylabel('Number of Frames')
plt.grid(axis='y')

for i, count in enumerate(counts):
    plt.text(i, count + 5, str(count), ha='center', fontsize=12)

# Save chart
output_folder = r"media\sample_outputs"
os.makedirs(output_folder, exist_ok=True)
chart_path = os.path.join(output_folder, "helmet_detection_bar_chart.jpg")
plt.savefig(chart_path)
plt.show()

print(f"\n📈 Bar chart saved at: {chart_path}")
