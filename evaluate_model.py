# import cv2
# import torch
# import numpy as np
# import pytesseract
# from ultralytics import YOLO
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Set Tesseract OCR path (update if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Load trained YOLO model
# model = YOLO(r'C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\best_LPR.pt')

# # Load test image
# image_path = r'C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\overspeed_vehicle_2.jpg'
# image = cv2.imread(image_path)

# # Perform inference
# results = model(image_path)

# # Store predictions
# pred_boxes = []
# pred_classes = []
# pred_confidences = []
# license_plates = []  # To store extracted license plate texts

# # Extract bounding boxes and class IDs
# for result in results:
#     detections = result.boxes.data.cpu().numpy()
    
#     for det in detections:
#         x_min, y_min, x_max, y_max, confidence, class_id = det
#         pred_boxes.append([x_min, y_min, x_max, y_max])
#         pred_classes.append(int(class_id))
#         pred_confidences.append(confidence)

# # Load ground truth labels for this image
# # Format: [class_id, x_min, y_min, x_max, y_max]
# ground_truth_boxes = [[0, 340, 580, 580, 650]]  # Replace with actual values
# ground_truth_classes = [0]  # Corresponding class IDs

# # Convert to NumPy arrays
# pred_boxes = np.array(pred_boxes)
# ground_truth_boxes = np.array([gt[1:] for gt in ground_truth_boxes])
# pred_classes = np.array(pred_classes)
# ground_truth_classes = np.array(ground_truth_classes)

# # Compute IoU (Intersection over Union)
# def compute_iou(box1, box2):
#     """Compute IoU between two bounding boxes."""
#     x_min1, y_min1, x_max1, y_max1 = box1
#     x_min2, y_min2, x_max2, y_max2 = box2

#     # Calculate intersection
#     x_left = max(x_min1, x_min2)
#     y_top = max(y_min1, y_min2)
#     x_right = min(x_max1, x_max2)
#     y_bottom = min(y_max1, y_max2)

#     if x_right < x_left or y_bottom < y_top:
#         return 0.0

#     intersection_area = (x_right - x_left) * (y_bottom - y_top)

#     # Calculate areas of both boxes
#     box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
#     box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

#     # Compute IoU
#     iou = intersection_area / float(box1_area + box2_area - intersection_area)
#     return iou

# # Compute IoU scores
# iou_scores = [compute_iou(pred, gt) for pred, gt in zip(pred_boxes, ground_truth_boxes)]

# # Apply IoU threshold to count correct detections
# iou_threshold = 0.5
# correct_detections = [iou > iou_threshold for iou in iou_scores]

# # Compute evaluation metrics
# accuracy = accuracy_score(ground_truth_classes, pred_classes)
# precision = precision_score(ground_truth_classes, pred_classes, average='weighted', zero_division=1)
# recall = recall_score(ground_truth_classes, pred_classes, average='weighted', zero_division=1)
# f1 = f1_score(ground_truth_classes, pred_classes, average='weighted', zero_division=1)

# # License Plate OCR
# for i, pred in enumerate(pred_boxes):
#     x_min, y_min, x_max, y_max = map(int, pred)

#     # Crop the detected license plate region
#     plate_img = image[y_min:y_max, x_min:x_max]

#     # Convert to grayscale and apply thresholding
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Perform OCR
#     plate_text = pytesseract.image_to_string(thresh, config="--psm 8 --oem 3").strip()
    
#     # Store extracted text
#     license_plates.append(plate_text)
    
#     # Draw bounding box and text on image
#     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#     cv2.putText(image, plate_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#     print(f"Detected License Plate {i+1}: {plate_text}")

# # Print evaluation metrics
# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-score: {f1:.2f}")
# print(f"IoU Scores: {iou_scores}")

# # Save and display results
# cv2.imwrite("output_image.jpg", image)
# cv2.imshow("Detection Result", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# speed

import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO(r'C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\yolov8.pt')  # Use 'yolov8n.pt' or your trained model

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Optical Flow Parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
prev_gray = None
prev_points = {}

# Video capture
cap = cv2.VideoCapture(r'C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\videos\test5.mp4')  # Replace with your video file
fps = cap.get(cv2.CAP_PROP_FPS)
pixels_per_meter = 8  # Approximate scale factor (needs calibration)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for Optical Flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # YOLOv8 Object Detection
    results = yolo_model(frame)[0]
    detections = []
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        if int(cls) in [2, 3, 5, 7]:  # Car, Motorcycle, Bus, Truck
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))
    
    # Track vehicles
    tracks = tracker.update_tracks(detections, frame=frame)
    
    new_points = {}
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = track.to_ltrb()
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        new_points[track_id] = (center_x, center_y)
        
        # Calculate Speed using Optical Flow
        if prev_gray is not None and track_id in prev_points:
            prev_pt = np.array([prev_points[track_id]], dtype=np.float32)
            new_pt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None, **lk_params)
            if status[0] == 1:
                dx, dy = new_pt[0] - prev_pt[0]
                pixel_distance = np.sqrt(dx**2 + dy**2)
                speed_kmh = (pixel_distance * fps) / pixels_per_meter * 3.6  # Convert to km/h
                
                # Display Speed
                cv2.putText(frame, f"{int(speed_kmh)} km/h", (center_x, center_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    prev_gray = gray.copy()
    prev_points = new_points
    
    # Display the frame
    # Resize frame to fit the screen properly
    frame_resized = cv2.resize(frame, (1280, 720))  # Adjust resolution as needed
    cv2.imshow('AI-Powered Traffic Monitoring', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
