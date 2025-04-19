# import pytesseract
# from PIL import Image

# # Path to the Tesseract executable (change it according to your installation)
# pytesseract.pytesseract.tesseract_cmd = r"C:/Users/aksha/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"

# # Path to your image file
# image_path = "C:/Users/aksha/OneDrive/Desktop/IMG_4139_00236.jpg"

# # Open the image using PIL (Python Imaging Library)
# image = Image.open(image_path)

# # Use Tesseract to recognize text from the image
# text = pytesseract.image_to_string(image)

# # Print the extracted text
# print(text)









# # # Ultralytics YOLO 🚀, GPL-3.0 license

# # import hydra
# # import torch
# # import argparse
# # import time
# # from pathlib import Path
# # import math
# # import cv2
# # import torch
# # import torch.backends.cudnn as cudnn
# # from numpy import random
# # from ultralytics.yolo.engine.predictor import BasePredictor
# # from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# # from ultralytics.yolo.utils.checks import check_imgsz
# # from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# # import cv2
# # from deep_sort_pytorch.utils.parser import get_config
# # from deep_sort_pytorch.deep_sort import DeepSort
# # from collections import deque
# # import numpy as np


# # palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
# # data_deque = {}

# # deepsort = None

# # object_counter = {}

# # object_counter1 = {}

# # line = [(100, 500), (1050, 500)]
# # speed_line_queue = {}
# # def estimatespeed(Location1, Location2):
# #     #Euclidean Distance Formula
# #     d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
# #     # defining thr pixels per meter
# #     ppm = 8
# #     d_meters = d_pixel/ppm
# #     time_constant = 15*3.6
# #     #distance = speed/time
# #     speed = d_meters * time_constant

# #     return int(speed)
# # def init_tracker():
# #     global deepsort
# #     cfg_deep = get_config()
# #     cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

# #     deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
# #                             max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
# #                             nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
# #                             max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
# #                             use_cuda=True)
# # ##########################################################################################
# # def xyxy_to_xywh(*xyxy):
# #     """" Calculates the relative bounding box from absolute pixel values. """
# #     bbox_left = min([xyxy[0].item(), xyxy[2].item()])
# #     bbox_top = min([xyxy[1].item(), xyxy[3].item()])
# #     bbox_w = abs(xyxy[0].item() - xyxy[2].item())
# #     bbox_h = abs(xyxy[1].item() - xyxy[3].item())
# #     x_c = (bbox_left + bbox_w / 2)
# #     y_c = (bbox_top + bbox_h / 2)
# #     w = bbox_w
# #     h = bbox_h
# #     return x_c, y_c, w, h

# # def xyxy_to_tlwh(bbox_xyxy):
# #     tlwh_bboxs = []
# #     for i, box in enumerate(bbox_xyxy):
# #         x1, y1, x2, y2 = [int(i) for i in box]
# #         top = x1
# #         left = y1
# #         w = int(x2 - x1)
# #         h = int(y2 - y1)
# #         tlwh_obj = [top, left, w, h]
# #         tlwh_bboxs.append(tlwh_obj)
# #     return tlwh_bboxs

# # def compute_color_for_labels(label):
# #     """
# #     Simple function that adds fixed color depending on the class
# #     """
# #     if label == 2: # Car
# #         color = (222,82,175)
# #     elif label == 3:  # Motobike
# #         color = (0, 204, 255)
# #     elif label == 5:  # Bus
# #         color = (0, 149, 255)
# #     else:
# #         color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
# #     return tuple(color)

# # def draw_border(img, pt1, pt2, color, thickness, r, d):
# #     x1,y1 = pt1
# #     x2,y2 = pt2
# #     # Top left
# #     cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
# #     cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
# #     cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
# #     # Top right
# #     cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
# #     cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
# #     cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
# #     # Bottom left
# #     cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
# #     cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
# #     cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
# #     # Bottom right
# #     cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
# #     cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
# #     cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# #     cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
# #     cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
# #     cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
# #     cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
# #     cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
# #     cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
# #     return img

# # def UI_box(x, img, color=None, label=None, line_thickness=None):
# #     # Plots one bounding box on image img
# #     tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
# #     color = color or [random.randint(0, 255) for _ in range(3)]
# #     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
# #     cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
# #     if label:
# #         tf = max(tl - 1, 1)  # font thickness
# #         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

# #         img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

# #         cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# # def intersect(A,B,C,D):
# #     return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# # def ccw(A,B,C):
# #     return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


# # def get_direction(point1, point2):
# #     direction_str = ""

# #     # calculate y axis direction
# #     if point1[1] > point2[1]:
# #         direction_str += "South"
# #     elif point1[1] < point2[1]:
# #         direction_str += "North"
# #     else:
# #         direction_str += ""

# #     # calculate x axis direction
# #     if point1[0] > point2[0]:
# #         direction_str += "East"
# #     elif point1[0] < point2[0]:
# #         direction_str += "West"
# #     else:
# #         direction_str += ""

# #     return direction_str
# # def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
# #     cv2.line(img, line[0], line[1], (46,162,112), 3)

# #     height, width, _ = img.shape
# #     # remove tracked point from buffer if object is lost
# #     for key in list(data_deque):
# #       if key not in identities:
# #         data_deque.pop(key)

# #     for i, box in enumerate(bbox):
# #         x1, y1, x2, y2 = [int(i) for i in box]
# #         x1 += offset[0]
# #         x2 += offset[0]
# #         y1 += offset[1]
# #         y2 += offset[1]
# #         label=' '
# #         color = (255, 255, 255)
        
# #         if object_id[i] == 2 or object_id[i] == 3:  # Check if object is car (2) or bike (3)
# #             color = compute_color_for_labels(object_id[i])
# #             try:
# #                 object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
# #                 speed_line_queue[id].append(object_speed)
                
# #                 # Check if speed exceeds 60 kmph
# #                 if sum(speed_line_queue[id])//len(speed_line_queue[id]) > 60:
# #                     # Cropping and saving the image
# #                     crop_img = img[y1:y2, x1:x2]
# #                     cv2.imwrite(f'over_speed_{id}.jpg', crop_img)
                
# #                 label = '{}{:d}'.format("", id) + ":" + names[object_id[i]] + " " + str(sum(speed_line_queue[id])//len(speed_line_queue[id])) + "km/h"
# #             except Exception as e:
# #                 print(f"Error calculating speed or saving image: {e}")

# #         UI_box(box, img, label=label, color=color, line_thickness=2)


# #         # code to find center of bottom edge
# #         center = (int((x2+x1)/ 2), int((y2+y2)/2))

# #         # get ID of object
# #         id = int(identities[i]) if identities is not None else 0

# #         # create new buffer for new object
# #         if id not in data_deque:  
# #           data_deque[id] = deque(maxlen= 64)
# #           speed_line_queue[id] = []
# #         color = compute_color_for_labels(object_id[i])
# #         obj_name = names[object_id[i]]
# #         label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

# #         # add center to buffer
# #         data_deque[id].appendleft(center)
# #         if len(data_deque[id]) >= 2:
# #           direction = get_direction(data_deque[id][0], data_deque[id][1])
# #           object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
# #           speed_line_queue[id].append(object_speed)
# #           if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
# #               cv2.line(img, line[0], line[1], (255, 255, 255), 3)
# #               if "South" in direction:
# #                 if obj_name not in object_counter:
# #                     object_counter[obj_name] = 1
# #                 else:
# #                     object_counter[obj_name] += 1
# #               if "North" in direction:
# #                 if obj_name not in object_counter1:
# #                     object_counter1[obj_name] = 1
# #                 else:
# #                     object_counter1[obj_name] += 1

# #         try:
# #             label = label + " " + str(sum(speed_line_queue[id])//len(speed_line_queue[id])) + "km/h"
# #         except:
# #             pass
# #         UI_box(box, img, label=label, color=color, line_thickness=2)
# #         # draw trail
# #         for i in range(1, len(data_deque[id])):
# #             # check if on buffer value is none
# #             if data_deque[id][i - 1] is None or data_deque[id][i] is None:
# #                 continue
# #             # generate dynamic thickness of trails
# #             thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
# #             # draw trails
# #             cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    
# #     #4. Display Count in top right corner
# #         for idx, (key, value) in enumerate(object_counter1.items()):
# #             cnt_str = str(key) + ":" +str(value)
# #             cv2.line(img, (width - 500,25), (width,25), [85,45,255], 40)
# #             cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
# #             cv2.line(img, (width - 150, 65 + (idx*40)), (width, 65 + (idx*40)), [85, 45, 255], 30)
# #             cv2.putText(img, cnt_str, (width - 150, 75 + (idx*40)), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)

# #         for idx, (key, value) in enumerate(object_counter.items()):
# #             cnt_str1 = str(key) + ":" +str(value)
# #             cv2.line(img, (20,25), (500,25), [85,45,255], 40)
# #             cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
# #             cv2.line(img, (20,65+ (idx*40)), (127,65+ (idx*40)), [85,45,255], 30)
# #             cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    
    
    
# #     return img

# # class DetectionPredictor(BasePredictor):

# #     def get_annotator(self, img):
# #         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

# #     def preprocess(self, img):
# #         img = torch.from_numpy(img).to(self.model.device)
# #         img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
# #         img /= 255  # 0 - 255 to 0.0 - 1.0
# #         return img

# #     def postprocess(self, preds, img, orig_img):
# #         preds = ops.non_max_suppression(preds,
# #                                         self.args.conf,
# #                                         self.args.iou,
# #                                         agnostic=self.args.agnostic_nms,
# #                                         max_det=self.args.max_det)

# #         for i, pred in enumerate(preds):
# #             shape = orig_img[i].shape if self.webcam else orig_img.shape
# #             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

# #         return preds

# #     def write_results(self, idx, preds, batch):
# #         p, im, im0 = batch
# #         all_outputs = []
# #         log_string = ""
# #         if len(im.shape) == 3:
# #             im = im[None]  # expand for batch dim
# #         self.seen += 1
# #         im0 = im0.copy()
# #         if self.webcam:  # batch_size >= 1
# #             log_string += f'{idx}: '
# #             frame = self.dataset.count
# #         else:
# #             frame = getattr(self.dataset, 'frame', 0)

# #         self.data_path = p
# #         save_path = str(self.save_dir / p.name)  # im.jpg
# #         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
# #         log_string += '%gx%g ' % im.shape[2:]  # print string
# #         self.annotator = self.get_annotator(im0)

# #         det = preds[idx]
# #         det = det[(det[:, 5] == 2) | (det[:, 5] == 3)]
# #         all_outputs.append(det)
# #         if len(det) == 0:
# #             return log_string
# #         for c in det[:, 5].unique():
# #             n = (det[:, 5] == c).sum()  # detections per class
# #             log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
# #         # write
# #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
# #         xywh_bboxs = []
# #         confs = []
# #         oids = []
# #         outputs = []
# #         for *xyxy, conf, cls in reversed(det):
# #             x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
# #             xywh_obj = [x_c, y_c, bbox_w, bbox_h]
# #             xywh_bboxs.append(xywh_obj)
# #             confs.append([conf.item()])
# #             oids.append(int(cls))
# #         xywhs = torch.Tensor(xywh_bboxs)
# #         confss = torch.Tensor(confs)
          
# #         outputs = deepsort.update(xywhs, confss, oids, im0)
# #         if len(outputs) > 0:
# #             bbox_xyxy = outputs[:, :4]
# #             identities = outputs[:, -2]
# #             object_id = outputs[:, -1]
            
# #             draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)

# #         return log_string


# # @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# # def predict(cfg):
# #     init_tracker()
# #     cfg.model = cfg.model or "yolov8n.pt"
# #     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # Check image size
# #     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
# #     predictor = DetectionPredictor(cfg)
# #     predictor()

# # # Main execution logic
# # if __name__ == "__main__":
# #     predict()









# # import cv2
# # import numpy as np
# # import os
# # import django
# # import time
# # from collections import deque
# # from ultralytics import YOLO
# # from paddleocr import PaddleOCR
# # from deep_sort_realtime.deepsort_tracker import DeepSort
# # from django.core.mail import EmailMessage
# # from django.conf import settings

# # # ✅ Load Django settings
# # os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'platevision.settings')
# # django.setup()

# # # ✅ Load YOLOv8 Model for Vehicle Detection
# # model = YOLO("yolov8n.pt")
# # vehicle_classes = [2, 3, 5, 7]  

# # # ✅ Load DeepSORT Tracker
# # tracker = DeepSort(max_age=30)

# # # ✅ Initialize PaddleOCR for License Plate Recognition
# # ocr = PaddleOCR(use_angle_cls=True, lang="en")

# # # ✅ Open Video File
# # video_path = r"C:\Users\moham\OneDrive\Desktop\Batch-C5\PlateVision-X-Helmet-Numberplate-Speed-Detection-main\videos\test4.mp4"
# # cap = cv2.VideoCapture(video_path)

# # if not cap.isOpened():
# #     print("Error: Could not open video file.")
# #     exit()

# # # ✅ Get Video Properties
# # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # fps = int(cap.get(cv2.CAP_PROP_FPS))  # ✅ Correct FPS Handling

# # print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps}")

# # # ✅ Video Writer
# # out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

# # # ✅ Speed Limit Configuration
# # SPEED_LIMIT = 40  # Set realistic speed limit in km/h

# # def send_email_alert(plate_number, speed, image_path):
# #     try:
# #         subject = "🚨 Speed Violation Alert!"
# #         body = f"Alert! Vehicle {plate_number} exceeded {speed} km/h!"
# #         recipient_list = ["mohammadfarjana51@gmail.com"]  

# #         email = EmailMessage(subject, body, settings.DEFAULT_FROM_EMAIL, recipient_list)
# #         email.attach_file(image_path)  # Attach high-quality vehicle image
# #         email.send()

# #         print(f"✅ Email sent with image for vehicle {plate_number} exceeding {speed} km/h")
    
# #     except Exception as e:
# #         print(f"❌ Error sending email: {str(e)}")

# # # ✅ Vehicle Speed Tracking
# # vehicle_speeds = {}
# # speed_history = {}

# # # ✅ Real-world reference values
# # REAL_WORLD_DISTANCE_METERS = 10  # Approximate distance between two reference points on the road
# # PIXEL_DISTANCE_REFERENCE = 200  # Manually measured pixel distance for that real-world distance
# # meters_per_pixel = REAL_WORLD_DISTANCE_METERS / PIXEL_DISTANCE_REFERENCE  # ✅ Calculate only once

# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     original_frame = frame.copy()
# #     results = model(frame)
# #     detections = []

# #     for result in results:
# #         boxes = result.boxes.xyxy.cpu().numpy()
# #         confs = result.boxes.conf.cpu().numpy()
# #         clss = result.boxes.cls.cpu().numpy()

# #         for i, box in enumerate(boxes):
# #             x1, y1, x2, y2 = map(int, box)
# #             class_id = int(clss[i])
# #             conf = confs[i]

# #             if class_id in vehicle_classes and conf > 0.5:
# #                 detections.append(([x1, y1, x2, y2], conf, class_id))

# #     tracks = tracker.update_tracks(detections, frame=frame)

# #     for track in tracks:
# #         if not track.is_confirmed():
# #             continue

# #         track_id = track.track_id
# #         x1, y1, x2, y2 = map(int, track.to_tlbr())
# #         color = (0, 255, 0)  
# #         thickness = 4

# #         # ✅ Speed Calculation
# #         if track_id not in vehicle_speeds:
# #             vehicle_speeds[track_id] = (cap.get(cv2.CAP_PROP_POS_FRAMES), x1)
# #             speed_history[track_id] = deque(maxlen=5)
# #         else:
# #             prev_frame, prev_x1 = vehicle_speeds[track_id]
# #             current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
# #             frame_diff = current_frame - prev_frame  # ✅ Frame-based tracking
            
# #             if frame_diff > 0:
# #                 frame_distance = abs(x1 - prev_x1)  
# #                 real_distance = frame_distance * meters_per_pixel
# #                 speed = (real_distance / (frame_diff * (1 / fps))) * 3.6  # ✅ Corrected Speed Formula
                
# #                 speed_history[track_id].append(speed)
# #                 smooth_speed = int(np.mean(speed_history[track_id]))

# #                 # ✅ Detect Speed Violation
# #                 if smooth_speed > SPEED_LIMIT:
# #                     color = (0, 0, 255)  
# #                     plate_number = "UNKNOWN"

# #                     # ✅ License Plate Extraction
# #                     plate_roi = frame[y1:y2, x1:x2]

# #                     if plate_roi.shape[0] > 10 and plate_roi.shape[1] > 10:
# #                         plate_roi_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
# #                         plate_text = ocr.ocr(plate_roi_gray, cls=True)
                        
# #                         if plate_text and plate_text[0]:
# #                             plate_number = plate_text[0][0][1][0]
# #                         else:
# #                             plate_number = "UNKNOWN"

# #                     # ✅ Save High-Resolution Image
# #                     vehicle_img_path = f"overspeed_vehicle_{track_id}.jpg"
# #                     cv2.imwrite(vehicle_img_path, original_frame[y1:y2, x1:x2])  

# #                     # ✅ Send Email Alert with Image
# #                     send_email_alert(plate_number, smooth_speed, vehicle_img_path)

# #                     # ✅ Display Alert Message
# #                     cv2.putText(frame, "⚠ OVERSPEEDING!", (x1, y1 - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
# #                     cv2.putText(frame, f"Plate: {plate_number}", (x1, y1 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

# #                 # ✅ Display Speed Information
# #                 cv2.putText(frame, f"Speed: {smooth_speed} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# #                 vehicle_speeds[track_id] = (current_frame, x1)

# #         # ✅ Draw Bounding Box
# #         cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
# #         cv2.putText(frame, f"ID {track_id}", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

# #     # ✅ Show & Save Output
# #     cv2.imshow("Traffic System", cv2.resize(frame, (1024, 576)))
# #     out.write(frame)

# #     if cv2.waitKey(1) & 0xFF == ord("q"):
# #         break

# # cap.release()
# # out.release()
# # cv2.destroyAllWindows()


# # it is used for the before 1 save strictly 


# import cv2
# import numpy as np
# import os
# import django
# import time
# import argparse
# from collections import deque
# from ultralytics import YOLO
# from paddleocr import PaddleOCR
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from django.core.mail import EmailMessage
# from django.conf import settings
# from django.core.files.base import ContentFile
# from io import BytesIO
# from django.utils.timezone import now

# # ✅ Load Django settings
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'platevision.settings')
# django.setup()
# from app.models import SpeedViolation
# from datetime import datetime
# # ✅ Parse command-line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--video", required=True, help="Path to the video file")
# args = parser.parse_args()

# # ✅ Load YOLOv8 Model for Vehicle Detection
# model = YOLO("yolov8n.pt")
# vehicle_classes = [2, 3, 5, 7]  

# # ✅ Load DeepSORT Tracker
# tracker = DeepSort(max_age=30)

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
# fps = int(cap.get(cv2.CAP_PROP_FPS))  # ✅ Correct FPS Handling

# print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps}")

# # ✅ Video Writer
# out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

# # ✅ Speed Limit Configuration
# SPEED_LIMIT = 80  # Set realistic speed limit in km/h
# alerted_vehicles = {}  # ✅ Track vehicles that were alerted
# ALERT_COOLDOWN = 10  # ✅ Prevent multiple alerts within 10 seconds

# def save_violation(plate_number, speed, image_bytes):
#     """ ✅ Save speed violation details to the database """
#     try:
#         violation = SpeedViolation(plate_number=plate_number, speed=speed)
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

# # ✅ Vehicle Speed Tracking
# vehicle_speeds = {}
# speed_history = {}

# # ✅ Real-world reference values
# REAL_WORLD_DISTANCE_METERS = 10  # Approximate distance between two reference points on the road
# PIXEL_DISTANCE_REFERENCE = 200  # Manually measured pixel distance for that real-world distance
# meters_per_pixel = REAL_WORLD_DISTANCE_METERS / PIXEL_DISTANCE_REFERENCE  # ✅ Calculate only once

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     original_frame = frame.copy()
#     results = model(frame)
#     detections = []

#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy()
#         confs = result.boxes.conf.cpu().numpy()
#         clss = result.boxes.cls.cpu().numpy()

#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = map(int, box)
#             class_id = int(clss[i])
#             conf = confs[i]

#             if class_id in vehicle_classes and conf > 0.5:
#                 detections.append(([x1, y1, x2, y2], conf, class_id))

#     tracks = tracker.update_tracks(detections, frame=frame)

#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         track_id = track.track_id
#         x1, y1, x2, y2 = map(int, track.to_tlbr())
#         color = (0, 255, 0)  
#         thickness = 4

#         # ✅ Speed Calculation
#         if track_id not in vehicle_speeds:
#             vehicle_speeds[track_id] = (cap.get(cv2.CAP_PROP_POS_FRAMES), x1)
#             speed_history[track_id] = deque(maxlen=5)
#         else:
#             prev_frame, prev_x1 = vehicle_speeds[track_id]
#             current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
#             frame_diff = current_frame - prev_frame  # ✅ Frame-based tracking
            
#             if frame_diff > 0:
#                 frame_distance = abs(x1 - prev_x1)  
#                 real_distance = frame_distance * meters_per_pixel
#                 speed = (real_distance / (frame_diff * (1 / fps))) * 3.6  # ✅ Corrected Speed Formula
                
#                 speed_history[track_id].append(speed)
#                 smooth_speed = int(np.mean(speed_history[track_id]))

#                 # ✅ Detect Speed Violation
                  
#                 if smooth_speed > SPEED_LIMIT and (track_id not in alerted_vehicles or time.time() - alerted_vehicles[track_id] > ALERT_COOLDOWN):
#                     color = (0, 0, 255)  
#                     alerted_vehicles[track_id] = time.time()

#                     # ✅ Extract License Plate using PaddleOCR
#                     vehicle_img = original_frame[y1:y2, x1:x2]
#                     if vehicle_img.size == 0:
#                         print("❌ ERROR: Empty or invalid image provided to OCR!")
#                         continue

#                     ocr_result = ocr.ocr(vehicle_img, cls=True)
#                     plate_number = "UNKNOWN"
#                     if ocr_result and ocr_result[0] and len(ocr_result[0]) > 0:
#                         plate_number = ocr_result[0][0][1][0]  

#                     # ✅ Save Violation Image
#                     image_path = f"media/violations/overspeed_vehicle_{plate_number}.jpg"
#                     cv2.imwrite(image_path, vehicle_img)

#                     # ✅ Save to Database
#                     save_violation(plate_number, smooth_speed, image_path)

#                     # ✅ Send Email Alert
#                     send_email_alert(plate_number, smooth_speed, image_path)

#                 vehicle_speeds[track_id] = (current_frame, x1)

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
#         cv2.putText(frame, f"ID {track_id}", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

#     cv2.imshow("Traffic System", cv2.resize(frame, (1024, 576)))
#     out.write(frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# ####3###33333############################################################################################################

# speed is controlled in this code


# import cv2
# import numpy as np
# import os
# import django
# import time
# import argparse
# from collections import deque
# from ultralytics import YOLO
# from paddleocr import PaddleOCR
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from django.core.mail import EmailMessage
# from django.conf import settings
# from django.core.files.base import ContentFile
# from django.utils.timezone import now

# # ✅ Load Django settings
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'platevision.settings')
# django.setup()
# from app.models import SpeedViolation
# from datetime import datetime

# # ✅ Parse command-line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--video", required=True, help="Path to the video file")
# args = parser.parse_args()

# # ✅ Load YOLOv8 Model for Vehicle Detection
# model = YOLO("yolov8n.pt")
# vehicle_classes = [2, 3, 5, 7]  

# # ✅ Load DeepSORT Tracker
# tracker = DeepSort(max_age=30)

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
# out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

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

# def save_violation(plate_number, speed, image_path):
#     """ ✅ Save speed violation details to the database """
#     try:
#         with open(image_path, "rb") as image_file:
#             image_bytes = image_file.read()
        
#         violation = SpeedViolation(plate_number=plate_number, speed=speed)
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

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     original_frame = frame.copy()
#     results = model(frame)
#     detections = []

#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy()
#         confs = result.boxes.conf.cpu().numpy()
#         clss = result.boxes.cls.cpu().numpy()

#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = map(int, box)
#             class_id = int(clss[i])
#             conf = confs[i]

#             if class_id in vehicle_classes and conf > 0.5:
#                 detections.append(([x1, y1, x2, y2], conf, class_id))

#     tracks = tracker.update_tracks(detections, frame=frame)

#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         track_id = track.track_id
#         x1, y1, x2, y2 = map(int, track.to_tlbr())
#         color = (0, 255, 0)
#         thickness = 4

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

#                     # ✅ Extract License Plate using PaddleOCR
#                     vehicle_img = original_frame[y1:y2, x1:x2]
#                     if vehicle_img.size == 0:
#                         print("❌ ERROR: Empty or invalid image provided to OCR!")
#                         continue

#                     ocr_result = ocr.ocr(vehicle_img, cls=True)
#                     plate_number = "UNKNOWN"
#                     if ocr_result and ocr_result[0] and len(ocr_result[0]) > 0:
#                         plate_number = ocr_result[0][0][1][0]  

#                     # ✅ Save Violation Image
#                     image_path = f"media/violations/overspeed_vehicle_{plate_number}.jpg"
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

#######################################################################################################

# trying to workwith integration of speed and helmate


# import cv2
# import numpy as np
# import os
# import django
# import time
# import argparse
# from collections import deque
# from ultralytics import YOLO
# from paddleocr import PaddleOCR
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from django.core.mail import EmailMessage
# from django.conf import settings
# from django.core.files.base import ContentFile
# from django.utils.timezone import now

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
# vehicle_model = YOLO("yolov8n.pt")
# helmet_model = YOLO("best.pt")
# vehicle_classes = [2, 3, 5, 7]  

# # ✅ Load DeepSORT Tracker
# tracker = DeepSort(max_age=30)

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
# out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

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

# def save_violation(plate_number, speed, image_path):
#     """ ✅ Save speed violation details to the database """
#     try:
#         with open(image_path, "rb") as image_file:
#             image_bytes = image_file.read()
        
#         violation = SpeedViolation(plate_number=plate_number, speed=speed)
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
#         if track.det_class == 3:  # Motorcycle
#             helmet_detected = any(hx1 < x1 < hx2 and hy1 < y1 < hy2 for hx1, hy1, hx2, hy2 in helmet_detections)
#             if not helmet_detected and track_id not in helmet_violations:
#                 helmet_violations[track_id] = time.time()
#                 vehicle_img = original_frame[y1:y2, x1:x2]
#                 image_path = f"media/violations/helmet_violation_{track_id}.jpg"
#                 cv2.imwrite(image_path, vehicle_img)
#                 plate_number = "UNKNOWN"
#                 ocr_result = ocr.ocr(vehicle_img, cls=True)
#                 if ocr_result and ocr_result[0]:
#                     plate_number = ocr_result[0][0][1][0]
#                 save_helmet_violation(plate_number, image_path)
#                 send_helmet_email_alert(plate_number, image_path)
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

#                     # ✅ Extract License Plate using PaddleOCR
#                     vehicle_img = original_frame[y1:y2, x1:x2]
#                     if vehicle_img.size == 0:
#                         print("❌ ERROR: Empty or invalid image provided to OCR!")
#                         continue

#                     ocr_result = ocr.ocr(vehicle_img, cls=True)
#                     plate_number = "UNKNOWN"
#                     if ocr_result and ocr_result[0] and len(ocr_result[0]) > 0:
#                         plate_number = ocr_result[0][0][1][0]  

#                     # ✅ Save Violation Image
#                     image_path = f"media/violations/overspeed_vehicle_{plate_number}.jpg"
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

###################################################################################

#integration above code with lcr



