# import cv2
# import os
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # === SETUP === #
# video_path = r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\videos\test5.mp4"  # Replace with your own traffic video path
# helmet_model_path = r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\best.pt"        # Helmet detection YOLOv8 model
# vehicle_model_path = r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\yolov8.pt"    # Vehicle detection YOLOv8 model

# save_dir = "media/sample_outputs"
# os.makedirs(save_dir, exist_ok=True)

# # === Load Models === #
# vehicle_model = YOLO(vehicle_model_path)
# helmet_model = YOLO(helmet_model_path)
# tracker = DeepSort()

# # === Start Video === #
# cap = cv2.VideoCapture(video_path)

# frame_saved = {
#     "input": False,
#     "tracking": False,
#     "speed_1": False,
#     "speed_2": False,
#     "helmet": False
# }

# frame_number = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_number += 1

#     # === 3.2: Save Uploaded Input Frame === #
#     if not frame_saved["input"] and frame_number == 1:
#         cv2.imwrite(f"{save_dir}/3.2_uploaded_video_frame.jpg", frame)
#         frame_saved["input"] = True

#     # === Vehicle Detection & Tracking === #
#     vehicle_results = vehicle_model(frame)
#     boxes = vehicle_results[0].boxes.xyxy.cpu().numpy()
#     confs = vehicle_results[0].boxes.conf.cpu().numpy()
#     clss = vehicle_results[0].boxes.cls.cpu().numpy()

#     detections = []
#     for i in range(len(boxes)):
#         x1, y1, x2, y2 = boxes[i].astype(int)
#         conf = confs[i]
#         class_id = int(clss[i])
#         if conf > 0.5:
#             detections.append((boxes[i], conf, class_id))

#     tracks = tracker.update_tracks(detections, frame=frame)

#     for track in tracks:
#         if not track.is_confirmed():
#             continue
#         track_id = track.track_id
#         x1, y1, x2, y2 = map(int, track.to_ltrb())
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#         cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

#     # === 3.3: Save DeepSORT Tracking Frame === #
#     if not frame_saved["tracking"] and frame_number == 30:
#         cv2.imwrite(f"{save_dir}/3.3_vehicle_tracking_deepsort.jpg", frame)
#         frame_saved["tracking"] = True

#     # === 3.4: Save Speed Estimation Frames === #
#     if not frame_saved["speed_1"] and frame_number == 40:
#         cv2.imwrite(f"{save_dir}/3.4_speed_frame_start.jpg", frame)
#         frame_saved["speed_1"] = True

#     if not frame_saved["speed_2"] and frame_number == 55:
#         cv2.imwrite(f"{save_dir}/3.4_speed_frame_end.jpg", frame)
#         frame_saved["speed_2"] = True

#     # === Helmet Detection === #
#     helmet_results = helmet_model(frame)
#     for result in helmet_results:
#         for box in result.boxes.xyxy.cpu().numpy():
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, "Helmet", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#     # === 3.5: Save Helmet Detection Flow Frame === #
#     if not frame_saved["helmet"] and frame_number == 60:
#         cv2.imwrite(f"{save_dir}/3.5_helmet_detection_flow.jpg", frame)
#         frame_saved["helmet"] = True

#     # Display (Optional)
#     cv2.imshow("YOLOv8 + DeepSORT + Helmet", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import os

# cap = cv2.VideoCapture(r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\videos\test3.mp4")  # Replace with your video path
# os.makedirs("media/sample_outputs", exist_ok=True)

# ret, frame = cap.read()
# if ret:
#     cv2.imwrite("media/sample_outputs/4.1_sample_input_video_frame.jpg", frame)
#     print("✅ Frame saved as 4.1_sample_input_video_frame.jpg")
# cap.release()

import cv2
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === CONFIG === #
video_path = r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\videos\test3.mp4"  # Replace with your input video path
yolo_model_path = r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\weights\yolov8.pt"  # Replace with your trained YOLOv8 weights
save_dir = "media/sample_outputs"
save_frame_number = 80  # Save the annotated frame at this frame count
os.makedirs(save_dir, exist_ok=True)

# === MODEL LOAD === #
model = YOLO(yolo_model_path)
tracker = DeepSort()

# === VIDEO SETUP === #
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# === SPEED CALC SETUP === #
vehicle_positions = {}
speed_history = {}

# Assumed real-world pixel-to-meter mapping
REAL_WORLD_DISTANCE_METERS = 10
PIXEL_DISTANCE_REFERENCE = 200
meters_per_pixel = REAL_WORLD_DISTANCE_METERS / PIXEL_DISTANCE_REFERENCE

frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # === Detection === #
    results = model(frame)
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(clss[i])
            conf = confs[i]
            if conf > 0.5 and class_id in [2, 3, 5, 7]:  # vehicles
                detections.append((box, conf, class_id))

    # === Tracking === #
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        center_x = (x1 + x2) // 2

        # === Speed Calculation === #
        if track_id not in vehicle_positions:
            vehicle_positions[track_id] = (frame_number, center_x)
            speed_history[track_id] = []
            smooth_speed = 0
        else:
            prev_frame, prev_x = vehicle_positions[track_id]
            frame_diff = frame_number - prev_frame
            pixel_diff = abs(center_x - prev_x)
            real_distance = pixel_diff * meters_per_pixel
            time_seconds = frame_diff / fps
            if time_seconds > 0:
                speed = (real_distance / time_seconds) * 3.6
                if 0 < speed < 200:
                    speed_history[track_id].append(speed)
                vehicle_positions[track_id] = (frame_number, center_x)
            smooth_speed = int(np.mean(speed_history[track_id])) if speed_history[track_id] else 0
            

        # === Draw Bounding Box & Speed === #
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"Speed: {smooth_speed} km/h", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # === Save Frame === #
    if frame_number == save_frame_number:
        save_path = os.path.join(save_dir, "detected_vehicle_with_speed_overlay.jpg")
        cv2.imwrite(save_path, frame)
        print(f"✅ Image saved at: {save_path}")

    # Optional display
    cv2.imshow("Vehicle Detection with Speed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
