# import cv2
# from ultralytics import YOLO
# from paddleocr import PaddleOCR
# import os

# # === Load Models ===
# yolo_model = YOLO(r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\best_LPR.pt")  # replace with your license plate YOLOv8 model
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # === Image Input ===
# image_path = r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\videos\tttttttt.jpg"  # change to your image file
# img = cv2.imread(image_path)

# # === Run YOLO Detection ===
# results = yolo_model(image_path)
# boxes = results[0].boxes.xyxy.cpu().numpy()

# if not boxes.any():
#     print("No license plate detected.")
# else:
#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = map(int, box)
#         plate_crop = img[y1:y2, x1:x2]

#         # Save cropped image (optional)
#         cropped_path = f"plate_crop_{i}.jpg"
#         cv2.imwrite(cropped_path, plate_crop)

#         # === OCR on Plate ===
#         ocr_result = ocr.ocr(plate_crop, cls=True)
#         if ocr_result and ocr_result[0]:
#             plate_text = ocr_result[0][0][1][0].replace(" ", "")
#             confidence = ocr_result[0][0][1][1]
#             print(f"[{i+1}] Plate: {plate_text} | Confidence: {round(confidence*100, 2)}%")
#         else:
#             print(f"[{i+1}] No text detected on cropped plate.")


# import cv2
# from ultralytics import YOLO
# from paddleocr import PaddleOCR
# import time
# import os

# # === Config ===
# video_path = r'C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\videos\test8.mp4'  # Change to your video file
# model_path = r'C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\best_LPR.pt'         # Your YOLOv8 license plate model
# save_crops = True              # Set to False if you don't want cropped plates saved

# # === Load Models ===
# yolo_model = YOLO(model_path)
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # === Create Output Directory ===
# if save_crops:
#     os.makedirs("detected_plates", exist_ok=True)

# # === Open Video ===
# cap = cv2.VideoCapture(video_path)
# frame_id = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_id += 1

#     # === Detect Plates with YOLO ===
#     results = yolo_model(frame)
#     boxes = results[0].boxes.xyxy.cpu().numpy()

#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = map(int, box)
#         plate_crop = frame[y1:y2, x1:x2]

#         # Optional: Save crop
#         if save_crops:
#             crop_name = f"detected_plates/frame{frame_id}_plate{i}.jpg"
#             cv2.imwrite(crop_name, plate_crop)

#         # === OCR Detection ===
#         ocr_result = ocr.ocr(plate_crop, cls=True)
#         if ocr_result and ocr_result[0]:
#             plate_text = ocr_result[0][0][1][0].replace(" ", "")
#             confidence = round(ocr_result[0][0][1][1] * 100, 2)

#             print(f"[Frame {frame_id}] Plate {i+1}: {plate_text} | Confidence: {confidence}%")

#             # Optional: draw on frame
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, plate_text, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#     # === Display Frame (Optional) ===
#     cv2.imshow('License Plate Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

# === Load OCR & LPR Model === #
ocr = PaddleOCR(use_angle_cls=True, lang='en')
lpr_model = YOLO(r'C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\best_LPR.pt')  # Replace with your LPR model path

# === Load Video === #
video_path = r'C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\videos\test9.mp4' # Replace with your video
cap = cv2.VideoCapture(video_path)

# === Create Output Folder === #
output_folder = "media/sample_outputs"
os.makedirs(output_folder, exist_ok=True)

# === Counters for Accuracy === #
total_plates = 0
correct_plates = 0
missed_plates = 0

frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # === Run License Plate Detection === #
    results = lpr_model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []

    for (x1, y1, x2, y2) in detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        plate_img = frame[int(y1):int(y2), int(x1):int(x2)]

        if plate_img.size == 0:
            continue

        total_plates += 1

        # === Run OCR === #
        ocr_result = ocr.ocr(plate_img, cls=True)
        if ocr_result and ocr_result[0]:
            plate_number = ''.join([line[1][0] for line in ocr_result[0]]).replace(" ", "").upper()
            if len(plate_number) >= 6 and any(char.isdigit() for char in plate_number):
                correct_plates += 1
                print(f"✅ Plate {plate_number} recognized at frame {frame_number}")
            else:
                missed_plates += 1
                print("❌ Plate detected but OCR failed")
        else:
            missed_plates += 1
            print("❌ No plate text recognized")

cap.release()

# === Accuracy Summary === #
print("\n📊 License Plate Recognition Summary:")
print(f"📦 Total Plates Detected: {total_plates}")
print(f"✅ Correctly Recognized: {correct_plates}")
print(f"❌ Failed/Unknown: {missed_plates}")

accuracy_percent = (correct_plates / total_plates) * 100 if total_plates else 0
failure_percent = 100 - accuracy_percent

print(f"\n📈 Accuracy: {accuracy_percent:.2f}%")
print(f"📉 Failure Rate: {failure_percent:.2f}%")

# === Plot & Save Accuracy Chart === #
labels = ['Correctly Recognized', 'Failed/Unknown']
values = [correct_plates, missed_plates]
colors = ['green', 'red']

plt.figure(figsize=(6, 5))
plt.bar(labels, values, color=colors)
plt.title(f'LPR Accuracy (Total Plates: {total_plates})')
plt.ylabel('Number of Plates')
plt.grid(axis='y')

for i, value in enumerate(values):
    plt.text(i, value + 1, str(value), ha='center', fontsize=12)

chart_path = os.path.join(output_folder, "lpr_accuracy_chart.jpg")
plt.savefig(chart_path)
plt.show()

print(f"\n🖼️ Accuracy chart saved at: {chart_path}")
