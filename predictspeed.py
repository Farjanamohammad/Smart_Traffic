# import cv2
# import numpy as np
# import os
# import django
# import time
# import re
# import argparse
# from collections import deque
# from ultralytics import YOLO
# from paddleocr import PaddleOCR
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from django.core.mail import EmailMessage
# from django.conf import settings
# from django.core.files.base import ContentFile
# from django.utils.timezone import now
# import torch
# import paddleocr



# # ✅ Load Django settings
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'platevision.settings')
# django.setup()
# from app.models import SpeedViolation,HelmetViolation
# from datetime import datetime

# # ✅ Parse command-line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--video", required=True, help="Path to the video file")
# args = parser.parse_args()

# # ✅ Load YOLOv8 Model for Vehicle Detection
# vehicle_model = YOLO(r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\yolov8.pt")
# # ✅ Load YOLOv8 Model for helmate Detection
# helmet_model = YOLO(r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\best.pt")
# # ✅ Load Custom YOLO LPR Model
# lpr_model = YOLO(r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\best_LPR.pt")
# vehicle_classes = [2, 3, 5, 7]   

# # ✅ Load DeepSORT Tracker
# tracker = DeepSort(max_age=50, n_init=3, nms_max_overlap=1.0)

# # ✅ Initialize PaddleOCR for License Plate Recognition
# ocr = PaddleOCR(use_angle_cls=True, lang="en")

# # ✅ Open Video File
# video_path = args.video
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Error: Could not open video file.")
#     exit()

# # ✅ Get Video Properties
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))  # ✅ Get FPS only once

# print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps}")

# # ✅ Video Writer
# # out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))
# out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# # ✅ Speed Limit Configuration
# SPEED_LIMIT = 80  # Set realistic speed limit in km/h
# alerted_vehicles = {}  # ✅ Track vehicles that were alerted
# ALERT_COOLDOWN = 10  # ✅ Prevent multiple alerts within 10 seconds

# # ✅ Real-world reference values
# REAL_WORLD_DISTANCE_METERS = 10  # Approximate distance between two reference points on the road
# PIXEL_DISTANCE_REFERENCE = 200  # Manually measured pixel distance for that real-world distance
# meters_per_pixel = REAL_WORLD_DISTANCE_METERS / PIXEL_DISTANCE_REFERENCE  # ✅ Calculate only once

# # ✅ Vehicle Speed Tracking
# vehicle_speeds = {}
# speed_history = {}
# helmet_violations = {}

# detected_vehicles = {}  # Stores detected plates with timestamp

# def is_new_detection(plate_number):
#     current_time = time.time()
    
#     # If the plate is detected for the first time OR last detected >5s ago, accept it
#     if plate_number not in detected_vehicles or (current_time - detected_vehicles[plate_number]) > 5:
#         detected_vehicles[plate_number] = current_time  # Update timestamp
#         return True
#     return False

# violation_emails_sent = {}  # Stores plates and last email time

# def should_send_email(plate_number):
#     current_time = time.time()
    
#     # If no email was sent before OR last email was sent more than 10 seconds ago
#     if plate_number not in violation_emails_sent or (current_time - violation_emails_sent[plate_number]) > 10:
#         violation_emails_sent[plate_number] = current_time  # Update timestamp
#         return True
#     return False

# def clean_plate(ocr_text):
#     # Pattern for a valid Indian license plate
#     pattern = r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}'
#     match = re.findall(pattern, ocr_text)  # Find valid plates
#     return match[0] if match else None  # Return the first valid match

# def preprocess_plate_image(plate_img):
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (1, 1), 0)  # Reduce blur
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     sharp = cv2.Laplacian(thresh, cv2.CV_64F).var()  # Check sharpness
#     if sharp < 50:  # If too blurry, adjust
#         gray = cv2.equalizeHist(gray)  # Enhance contrast
#     return gray




# def extract_plate_text(plate_img):
#     """Extract license plate text using PaddleOCR."""
#     result = ocr.ocr(plate_img, cls=True)
#     if result and result[0]:
#         plate_number = ''.join([word[1][0] for word in result[0]])
#         return plate_number.replace(" ", "").upper()
#     return "UNKNOWN"

# def detect_license_plate(frame):
#     """Detect license plates using YOLO and extract OCR text"""
#     results = lpr_model.predict(source=frame, conf=0.35)  # Lower confidence threshold
#     detections = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []

    
#     for (x1, y1, x2, y2) in detections:
#         plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
        
#         if plate_img.size == 0:
#             continue
        
#         # Preprocess image for OCR
#         plate_img = preprocess_plate_image(plate_img)
        
#         # Extract text from plate image
#         plate_number = extract_plate_text(plate_img)
        
#         if plate_number and len(plate_number) > 6 and any(char.isdigit() for char in plate_number):
#             print(f"✅ Detected Plate: {plate_number}")
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, plate_number, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         else:
#             print("❌ Invalid plate detected, skipping...")
#     return frame

# def save_violation(plate_number, speed, image_path):
#     """ ✅ Save speed violation details to the database """
#     try:
#         with open(image_path, "rb") as image_file:
#             image_bytes = image_file.read()
        
#         violation = SpeedViolation(plate_number=plate_number.upper(), speed=speed)
#         violation.image.save(f"overspeed_vehicle_{plate_number}.jpg", ContentFile(image_bytes))
#         violation.save()
#         print(f"✅ Speed Violation Saved: {plate_number} - {speed} km/h")
#     except Exception as e:
#         print(f"❌ ERROR SAVING: {e}")

# def send_email_alert(plate_number, speed, image_path):
#     try:
#         subject = "🚨 Speed Violation Alert!"
#         body = f"Alert! Vehicle {plate_number} exceeded {speed} km/h!"
#         recipient_list = ["mohammadfarjana51@gmail.com"]  

#         email = EmailMessage(subject, body, settings.DEFAULT_FROM_EMAIL, recipient_list)
#         email.attach_file(image_path)  # Attach high-quality vehicle image
#         email.send()

#         print(f"✅ Email sent with image for vehicle {plate_number} exceeding {speed} km/h")
    
#     except Exception as e:
#         print(f"❌ Error sending email: {str(e)}")

# def save_helmet_violation(plate_number, image_path):
#     """ ✅ Save helmet violation details to the database """
#     try:
#         with open(image_path, "rb") as image_file:
#             image_bytes = image_file.read()

#         violation = HelmetViolation(plate_number=plate_number)
#         violation.image.save(f"helmet_violation_{plate_number}.jpg", ContentFile(image_bytes))
#         violation.save()
#         print(f"✅ Helmet Violation Saved: {plate_number}")
#     except Exception as e:
#         print(f"❌ ERROR SAVING: {e}")

# def send_helmet_email_alert(plate_number, image_path):
#     try:
#         subject = "🚨 Helmet Violation Alert!"
#         body = f"Alert! Vehicle {plate_number} rider is not wearing a helmet!"
#         recipient_list = ["mohammadfarjana51@gmail.com"]

#         email = EmailMessage(subject, body, settings.DEFAULT_FROM_EMAIL, recipient_list)
#         email.attach_file(image_path)
#         email.send()

#         print(f"✅ Email sent for helmet violation: {plate_number}")
#     except Exception as e:
#         print(f"❌ Error sending email: {str(e)}")


# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     original_frame = frame.copy()
#     results = vehicle_model(frame)
#     helmet_results = helmet_model(frame)
#     detections = []
#     helmet_detections = []

#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy()
#         confs = result.boxes.conf.cpu().numpy()
#         clss = result.boxes.cls.cpu().numpy()

#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = map(int, box)
#             class_id = int(clss[i])
#             conf = confs[i]

#             if class_id in vehicle_classes and conf > 0.5:
#                 detections.append((np.array([x1, y1, x2, y2]), conf, class_id))  # Ensure class ID is passed


#     tracks = tracker.update_tracks(detections, frame=frame)
#     # ✅ Helmet Detection
#     for result in helmet_results:
#         for box in result.boxes.xyxy.cpu().numpy():
#             x1, y1, x2, y2 = map(int, box)
#             helmet_detections.append((x1, y1, x2, y2))

#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         track_id = track.track_id
#         x1, y1, x2, y2 = map(int, track.to_tlbr())
#         color = (0, 255, 0)
#         thickness = 4
#         # ✅ Check if motorcycle rider is wearing a helmet
#          # Fix: Using track.det_class instead of track.class_id
#         if class_id == 3:  # ✅ Corrected: Motorcycle class check
#             helmet_detected = False  # Assume no helmet
#             for hx1, hy1, hx2, hy2 in helmet_detections:
#                 if hx1 < (x1 + x2) / 2 < hx2 and hy1 < (y1 + y2) / 2 < hy2:
#                     helmet_detected = True  # Helmet is present
#             if not helmet_detected and track_id not in helmet_violations:
#                 helmet_violations[track_id] = time.time()
#                 # Extract plate number first
#                 vehicle_img = original_frame[y1:y2, x1:x2]
#                 plate_number = "UNKNOWN"
#                 if vehicle_img is not None and vehicle_img.size > 0 and vehicle_img.shape[1] > 0:
#                     ocr_result = ocr.ocr(vehicle_img, cls=True)
#                 else:
#                     print("❌ Skipping OCR: Empty or Invalid Image")
#                     ocr_result = None

#                 if ocr_result and ocr_result[0]:
#                     plate_number = ocr_result[0][0][1][0]
#                 # Save image to media folder
#                 helmet_violation_dir = "media/violations/"
#                 os.makedirs(helmet_violation_dir, exist_ok=True)
#                 image_path = os.path.join(helmet_violation_dir, f"helmet_violation_{plate_number}.jpg")
#                 cv2.imwrite(image_path, vehicle_img)
#                 # Save and send alert
#                 save_helmet_violation(plate_number, image_path)
#                 send_helmet_email_alert(plate_number, image_path)
#                 print(f"🚨 Helmet Violation: {plate_number} - Image saved.")
#                 color = (0, 0, 255)
#         # ✅ Speed Calculation with Fixes
#         if track_id not in vehicle_speeds:
#             vehicle_speeds[track_id] = (cap.get(cv2.CAP_PROP_POS_FRAMES), x1)
#             speed_history[track_id] = deque(maxlen=5)
#         else:
#             prev_frame, prev_x1 = vehicle_speeds[track_id]
#             current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
#             frame_diff = max(1, current_frame - prev_frame)  # ✅ Prevent Zero Division

#             if frame_diff > 0:
#                 frame_distance = abs(x1 - prev_x1)
#                 real_distance = frame_distance * meters_per_pixel
#                 time_seconds = frame_diff / fps  # ✅ Correct Time Calculation
#                 speed = (real_distance / time_seconds) * 3.6  # ✅ Speed in km/h
#                 frame_time = frame_diff / fps  # Convert frame difference to seconds

#                 # ✅ Filter Out Unrealistic Speeds
#                 if 0 < speed < 200:  # Set a maximum limit (e.g., 200 km/h)
#                     speed_history[track_id].append(speed)

#                 smooth_speed = int(np.mean(speed_history[track_id])) if speed_history[track_id] else 0

#                 # ✅ Update vehicle_speeds to track latest frame and position
#                 vehicle_speeds[track_id] = (current_frame, x1)

#                 # ✅ Debugging: Print Speed Data
#                 print(f"Track ID: {track_id} | Speed: {smooth_speed} km/h")

#                 # ✅ Detect Speed Violation
#                 if smooth_speed > SPEED_LIMIT and (track_id not in alerted_vehicles or time.time() - alerted_vehicles[track_id] > ALERT_COOLDOWN):
#                     color = (0, 0, 255)  
#                     alerted_vehicles[track_id] = time.time()

#                     # ✅ License Plate Recognition (LPR) Fix
#                     # Extract License Plate Region
#                     if y2 > y1 and x2 > x1:  # ✅ Ensure valid cropping
#                         plate_img = frame[y1:y2, x1:x2]
#                         if plate_img.size > 0:  # ✅ Ensure image is not empty
#                             plate_img = preprocess_plate_image(plate_img)  # Preprocess for better OCR
#                             plate_number = extract_plate_text(plate_img)  
#                         # Ensure extracted plate number is valid
#                     if 'plate_number' in locals() and plate_number and len(plate_number) >= 4 and any(char.isdigit() for char in plate_number):

#                         print(f"✅ Final Plate Number: {plate_number}")
#                     else:
#                         plate_number = "UNKNOWN"

#                     # ✅ Perform LPR only if vehicle_img is valid
#                     if y2 > y1 and x2 > x1:  # ✅ Ensure valid region before capturing
#                         vehicle_img = frame[y1:y2, x1:x2]
#                         if vehicle_img.shape[0] > 0 and vehicle_img.shape[1] > 0:  # ✅ Ensure non-empty image
#                             vehicle_img = cv2.resize(vehicle_img, (600, 300))  # Keep more details
#                             vehicle_img = cv2.resize(vehicle_img, (600, 300))  # Keep more details 
#                             lpr_results = lpr_model(vehicle_img)  # Run YOLO-based plate detection
#                         # ✅ Process LPR Model Output
                    
#                     if 'lpr_results' in locals() and lpr_results and len(lpr_results) > 0:

#                         for lpr_result in lpr_results:
#                             for box in lpr_result.boxes.xyxy.cpu().numpy():
#                                 x1_p, y1_p, x2_p, y2_p = map(int, box)
#                                 # Ensure plate box is within vehicle image bounds
#                                 x1_p = max(0, x1_p)
#                                 y1_p = max(0, y1_p)
#                                 x2_p = min(vehicle_img.shape[1] - 1, x2_p)
#                                 y2_p = min(vehicle_img.shape[0] - 1, y2_p)
#                                 if x1_p < x2_p and y1_p < y2_p:
#                                     plate_img = vehicle_img[y1_p:y2_p, x1_p:x2_p]
#                                     # ✅ Ensure plate image is not empty
#                                     if plate_img is None or plate_img.size == 0:
#                                         print("❌ ERROR: Empty license plate region detected, setting plate as UNKNOWN.")
#                                         plate_number = "UNKNOWN"
#                                     else:
#                                     # ✅ Preprocess for OCR
#                                         plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#                                         plate_img = cv2.resize(plate_img, (200, 50))
#                                         _, plate_img = cv2.threshold(plate_img, 100, 255, cv2.THRESH_BINARY)  # Binarization
#                                         ocr_result = ocr.ocr(plate_img, cls=True)
#                                         if ocr_result and ocr_result[0]:
#                                             plate_number = ocr_result[0][0][1][0]  # Extract recognized plate number
#                                             print(f"✅ License Plate Recognized: {plate_number}")

#                     # ✅ Save Violation Image
#                     image_path = os.path.join(settings.MEDIA_ROOT, 'violations', f"overspeed_vehicle_{plate_number}.jpg")
#                     cv2.imwrite(image_path, vehicle_img)


#                     # ✅ Save to Database
#                     save_violation(plate_number, smooth_speed, image_path)

#                     # ✅ Send Email Alert
#                     send_email_alert(plate_number, smooth_speed, image_path)

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
#         cv2.putText(frame, f"ID {track_id}", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

#     cv2.imshow("Traffic System", cv2.resize(frame, (1024, 576)))
#     out.write(frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import django
import time
import re
import argparse


def extract_plate_number(vehicle_img, ocr):
    """
    Crops bottom region of vehicle, runs OCR, and returns best plate-like text.
    """
    h, w, _ = vehicle_img.shape
    # Focus only on bottom center region of the vehicle box
    plate_region = vehicle_img[int(0.6*h):h, int(0.2*w):int(0.8*w)]

    # Run OCR on cropped region
    result = ocr.ocr(plate_region, cls=True)

    if not result or not result[0]:
        return None

    best_text = None
    best_conf = 0
    for line in result[0]:
        text, conf = line[1][0], line[1][1]
        if conf > best_conf and 5 <= len(text) <= 15:  # Typical plate length
            best_text = text
            best_conf = conf
    return best_text

from collections import deque
from ultralytics import YOLO
from paddleocr import PaddleOCR
from deep_sort_realtime.deepsort_tracker import DeepSort
from django.core.mail import EmailMessage
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils.timezone import now
import torch
import paddleocr



# ✅ Load Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'platevision.settings')
django.setup()
from app.models import SpeedViolation,HelmetViolation
from datetime import datetime

# ✅ Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Path to the video file")
args = parser.parse_args()

# ✅ Load YOLOv8 Model for Vehicle Detection
vehicle_model = YOLO(r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\yolov8.pt")
# ✅ Load YOLOv8 Model for helmate Detection
helmet_model = YOLO(r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\best.pt")
# ✅ Load Custom YOLO LPR Model
lpr_model = YOLO(r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\best_LPR.pt")
vehicle_classes = [2, 3, 5, 7]   

# ✅ Load DeepSORT Tracker
tracker = DeepSort(max_age=50, n_init=3, nms_max_overlap=1.0)

# ✅ Initialize PaddleOCR for License Plate Recognition
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# ✅ Open Video File
video_path = args.video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# ✅ Get Video Properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # ✅ Get FPS only once

print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps}")

# ✅ Video Writer
# out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# ✅ Speed Limit Configuration
SPEED_LIMIT = 80  # Set realistic speed limit in km/h
alerted_vehicles = {}  # ✅ Track vehicles that were alerted
ALERT_COOLDOWN = 10  # ✅ Prevent multiple alerts within 10 seconds

# ✅ Real-world reference values
REAL_WORLD_DISTANCE_METERS = 10  # Approximate distance between two reference points on the road
PIXEL_DISTANCE_REFERENCE = 200  # Manually measured pixel distance for that real-world distance
meters_per_pixel = REAL_WORLD_DISTANCE_METERS / PIXEL_DISTANCE_REFERENCE  # ✅ Calculate only once

# ✅ Vehicle Speed Tracking
vehicle_speeds = {}
speed_history = {}
helmet_violations = {}

detected_vehicles = {}  # Stores detected plates with timestamp

def is_new_detection(plate_number):
    current_time = time.time()
    
    # If the plate is detected for the first time OR last detected >5s ago, accept it
    if plate_number not in detected_vehicles or (current_time - detected_vehicles[plate_number]) > 5:
        detected_vehicles[plate_number] = current_time  # Update timestamp
        return True
    return False

violation_emails_sent = {}  # Stores plates and last email time

def should_send_email(plate_number):
    current_time = time.time()
    
    # If no email was sent before OR last email was sent more than 10 seconds ago
    if plate_number not in violation_emails_sent or (current_time - violation_emails_sent[plate_number]) > 10:
        violation_emails_sent[plate_number] = current_time  # Update timestamp
        return True
    return False

def clean_plate(ocr_text):
    # Pattern for a valid Indian license plate
    pattern = r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}'
    match = re.findall(pattern, ocr_text)  # Find valid plates
    return match[0] if match else None  # Return the first valid match

def preprocess_plate_image(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)  # Reduce blur
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    sharp = cv2.Laplacian(thresh, cv2.CV_64F).var()  # Check sharpness
    if sharp < 50:  # If too blurry, adjust
        gray = cv2.equalizeHist(gray)  # Enhance contrast
    return gray




def extract_plate_text(plate_img):
    """Extract license plate text using PaddleOCR."""
    result = ocr.ocr(plate_img, cls=True)
    if result and result[0]:
        plate_number = ''.join([word[1][0] for word in result[0]])
        return plate_number.replace(" ", "").upper()
    return "UNKNOWN"

def detect_license_plate(frame):
    """Detect license plates using YOLO and extract OCR text"""
    results = lpr_model.predict(source=frame, conf=0.35)  # Lower confidence threshold
    detections = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []

    
    for (x1, y1, x2, y2) in detections:
        plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if plate_img.size == 0:
            continue
        
        # Preprocess image for OCR
        plate_img = preprocess_plate_image(plate_img)
        
        # Extract text from plate image
        plate_number = extract_plate_text(plate_img)
        
        if plate_number and len(plate_number) > 6 and any(char.isdigit() for char in plate_number):
            print(f"✅ Detected Plate: {plate_number}")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, plate_number, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            print("❌ Invalid plate detected, skipping...")
    return frame

def save_violation(plate_number, speed, image_path):
    """ ✅ Save speed violation details to the database """
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        violation = SpeedViolation(plate_number=plate_number.upper(), speed=speed)
        violation.image.save(f"overspeed_vehicle_{plate_number}.jpg", ContentFile(image_bytes))
        violation.save()
        print(f"✅ Speed Violation Saved: {plate_number} - {speed} km/h")
    except Exception as e:
        print(f"❌ ERROR SAVING: {e}")

def send_email_alert(plate_number, speed, image_path):
    try:
        subject = "🚨 Speed Violation Alert!"
        body = f"Alert! Vehicle {plate_number} exceeded {speed} km/h!"
        recipient_list = ["mohammadfarjana51@gmail.com"]  

        email = EmailMessage(subject, body, settings.DEFAULT_FROM_EMAIL, recipient_list)
        email.attach_file(image_path)  # Attach high-quality vehicle image
        email.send()

        print(f"✅ Email sent with image for vehicle {plate_number} exceeding {speed} km/h")
    
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")

def save_helmet_violation(plate_number, image_path):
    """ ✅ Save helmet violation details to the database """
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        violation = HelmetViolation(plate_number=plate_number)
        violation.image.save(f"helmet_violation_{plate_number}.jpg", ContentFile(image_bytes))
        violation.save()
        print(f"✅ Helmet Violation Saved: {plate_number}")
    except Exception as e:
        print(f"❌ ERROR SAVING: {e}")

def send_helmet_email_alert(plate_number, image_path):
    try:
        subject = "🚨 Helmet Violation Alert!"
        body = f"Alert! Vehicle {plate_number} rider is not wearing a helmet!"
        recipient_list = ["mohammadfarjana51@gmail.com"]

        email = EmailMessage(subject, body, settings.DEFAULT_FROM_EMAIL, recipient_list)
        email.attach_file(image_path)
        email.send()

        print(f"✅ Email sent for helmet violation: {plate_number}")
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy()
    results = vehicle_model(frame)
    helmet_results = helmet_model(frame)
    detections = []
    helmet_detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(clss[i])
        conf = confs[i]

        if class_id in vehicle_classes and conf > 0.5:
            w = x2 - x1
            h = y2 - y1
            aspect_ratio = w / h if h != 0 else 0

            # If box is too wide, assume it's covering more than one vehicle
            if w > 220 or aspect_ratio > 1.6:
                num_vehicles = round(w / 120)  # estimate how many vehicles are merged (120px width per vehicle)
                sub_box_width = w // num_vehicles

                for j in range(num_vehicles):
                    sub_x1 = x1 + j * sub_box_width
                    sub_x2 = sub_x1 + sub_box_width
                    detections.append((np.array([sub_x1, y1, sub_x2, y2]), conf, class_id))
            else:
                detections.append((np.array([x1, y1, x2, y2]), conf, class_id))




    tracks = tracker.update_tracks(detections, frame=frame)
    # ✅ Helmet Detection
    for result in helmet_results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            helmet_detections.append((x1, y1, x2, y2))

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        color = (0, 255, 0)
        thickness = 4
        # ✅ Check if motorcycle rider is wearing a helmet
         # Fix: Using track.det_class instead of track.class_id
        if class_id == 3:  # ✅ Corrected: Motorcycle class check
            helmet_detected = False  # Assume no helmet
            for hx1, hy1, hx2, hy2 in helmet_detections:
                if hx1 < (x1 + x2) / 2 < hx2 and hy1 < (y1 + y2) / 2 < hy2:
                    helmet_detected = True  # Helmet is present
            if not helmet_detected and track_id not in helmet_violations:
                helmet_violations[track_id] = time.time()
                # Extract plate number first
                vehicle_img = original_frame[y1:y2, x1:x2]
                plate_number = "UNKNOWN"
                if vehicle_img is not None and vehicle_img.size > 0 and vehicle_img.shape[1] > 0:
                    ocr_result = ocr.ocr(vehicle_img, cls=True)
                else:
                    print("❌ Skipping OCR: Empty or Invalid Image")
                    ocr_result = None

                if ocr_result and ocr_result[0]:
                    plate_number = ocr_result[0][0][1][0]
                # Save image to media folder
                helmet_violation_dir = "media/violations/"
                os.makedirs(helmet_violation_dir, exist_ok=True)
                image_path = os.path.join(helmet_violation_dir, f"helmet_violation_{plate_number}.jpg")
                cv2.imwrite(image_path, vehicle_img)
                # Save and send alert
                save_helmet_violation(plate_number, image_path)
                send_helmet_email_alert(plate_number, image_path)
                print(f"🚨 Helmet Violation: {plate_number} - Image saved.")
                color = (0, 0, 255)
        # ✅ Speed Calculation with Fixes
        if track_id not in vehicle_speeds:
            vehicle_speeds[track_id] = (cap.get(cv2.CAP_PROP_POS_FRAMES), x1)
            speed_history[track_id] = deque(maxlen=5)
        else:
            prev_frame, prev_x1 = vehicle_speeds[track_id]
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            frame_diff = max(1, current_frame - prev_frame)  # ✅ Prevent Zero Division

            if frame_diff > 0:
                frame_distance = abs(x1 - prev_x1)
                real_distance = frame_distance * meters_per_pixel
                time_seconds = frame_diff / fps  # ✅ Correct Time Calculation
                speed = (real_distance / time_seconds) * 3.6  # ✅ Speed in km/h
                frame_time = frame_diff / fps  # Convert frame difference to seconds

                # ✅ Filter Out Unrealistic Speeds
                if 0 < speed < 200:  # Set a maximum limit (e.g., 200 km/h)
                    speed_history[track_id].append(speed)

                smooth_speed = int(np.mean(speed_history[track_id])) if speed_history[track_id] else 0

                # ✅ Update vehicle_speeds to track latest frame and position
                vehicle_speeds[track_id] = (current_frame, x1)

                # ✅ Debugging: Print Speed Data
                print(f"Track ID: {track_id} | Speed: {smooth_speed} km/h")

                # ✅ Detect Speed Violation
                if smooth_speed > SPEED_LIMIT and (track_id not in alerted_vehicles or time.time() - alerted_vehicles[track_id] > ALERT_COOLDOWN):
                    color = (0, 0, 255)  
                    alerted_vehicles[track_id] = time.time()

                    # ✅ License Plate Recognition (LPR) Fix
                    # Extract License Plate Region
                    if y2 > y1 and x2 > x1:  # ✅ Ensure valid cropping
                        plate_img = frame[y1:y2, x1:x2]
                        if plate_img.size > 0:  # ✅ Ensure image is not empty
                            plate_img = preprocess_plate_image(plate_img)  # Preprocess for better OCR
                            plate_number = extract_plate_text(plate_img)  
                        # Ensure extracted plate number is valid
                    if 'plate_number' in locals() and plate_number and len(plate_number) >= 4 and any(char.isdigit() for char in plate_number):

                        print(f"✅ Final Plate Number: {plate_number}")
                    else:
                        plate_number = "UNKNOWN"

                    # ✅ Perform LPR only if vehicle_img is valid
                    if y2 > y1 and x2 > x1:  # ✅ Ensure valid region before capturing
                        vehicle_img = frame[y1:y2, x1:x2]
                        if vehicle_img.shape[0] > 0 and vehicle_img.shape[1] > 0:  # ✅ Ensure non-empty image
                            vehicle_img = cv2.resize(vehicle_img, (600, 300))  
                            # ✅ Split the vehicle image if it's too wide
                            v_h, v_w = vehicle_img.shape[:2]
                            plate_images = []
                            if v_w > 300:
                                # If vehicle image is wide, split it in half to avoid merged plates\
                                split_w = v_w // 2
                                left_crop = vehicle_img[:, :split_w]
                                plate_images.extend([left_crop, right_crop])
                            else:
                                plate_images.append(vehicle_img)
                                # ✅ Run LPR on all cropped regions
                            for img in plate_images:
                                lpr_results = lpr_model(img)
                                for result in lpr_results:
                                    for box in result.boxes.xyxy.cpu().numpy():
                                        x1_p, y1_p, x2_p, y2_p = map(int, box)
                                        x1_p = max(0, x1_p)
                                        y1_p = max(0, y1_p)
                                        x2_p = min(img.shape[1] - 1, x2_p)
                                        y2_p = min(img.shape[0] - 1, y2_p)
                                        if x1_p < x2_p and y1_p < y2_p:
                                            plate_img = img[y1_p:y2_p, x1_p:x2_p]
                                            if plate_img is not None and plate_img.size > 0:
                                                plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                                                plate_img = cv2.resize(plate_img, (200, 50))
                                                _, plate_img = cv2.threshold(plate_img, 100, 255, cv2.THRESH_BINARY)
                                                ocr_result = ocr.ocr(plate_img, cls=True)
                                                if ocr_result and ocr_result[0]:
                                                    plate_number = ocr_result[0][0][1][0]
                                                    print(f"✅ License Plate Recognized: {plate_number}")

                        # ✅ Process LPR Model Output
                    
                    if 'lpr_results' in locals() and lpr_results and len(lpr_results) > 0:

                        for lpr_result in lpr_results:
                            for box in lpr_result.boxes.xyxy.cpu().numpy():
                                x1_p, y1_p, x2_p, y2_p = map(int, box)
                                # Ensure plate box is within vehicle image bounds
                                x1_p = max(0, x1_p)
                                y1_p = max(0, y1_p)
                                x2_p = min(vehicle_img.shape[1] - 1, x2_p)
                                y2_p = min(vehicle_img.shape[0] - 1, y2_p)
                                if x1_p < x2_p and y1_p < y2_p:
                                    plate_img = vehicle_img[y1_p:y2_p, x1_p:x2_p]
                                    # ✅ Ensure plate image is not empty
                                    if plate_img is None or plate_img.size == 0:
                                        print("❌ ERROR: Empty license plate region detected, setting plate as UNKNOWN.")
                                        plate_number = "UNKNOWN"
                                    else:
                                    # ✅ Preprocess for OCR
                                        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                                        plate_img = cv2.resize(plate_img, (200, 50))
                                        _, plate_img = cv2.threshold(plate_img, 100, 255, cv2.THRESH_BINARY)  # Binarization
                                        ocr_result = ocr.ocr(plate_img, cls=True)
                                        if ocr_result and ocr_result[0]:
                                            plate_number = ocr_result[0][0][1][0]  # Extract recognized plate number
                                            print(f"✅ License Plate Recognized: {plate_number}")

                    # ✅ Save Violation Image
                    image_path = os.path.join(settings.MEDIA_ROOT, 'violations', f"overspeed_vehicle_{plate_number}.jpg")
                    cv2.imwrite(image_path, vehicle_img)


                    # ✅ Save to Database
                    save_violation(plate_number, smooth_speed, image_path)

                    # ✅ Send Email Alert
                    send_email_alert(plate_number, smooth_speed, image_path)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("Traffic System", cv2.resize(frame, (1024, 576)))
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()