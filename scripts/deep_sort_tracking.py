import sys
import os
import numpy as np
sys.path.append('./Real-time-object-detection/deep_sort')
sys.path.append('./Real-time-object-detection/pytorch-YOLOv4')
import cv2
import torch
from deep_sort.deep_sort import DeepSort
from yolov4_detection import load_model, detect

# Initialize YOLOv4 and DeepSORT
yolo_model = load_model()
deepsort = DeepSort("./Real-time-object-detection/deep_sort/ckpt/ckpt.t7")

# Load class names
class_names = open("./Real-time-object-detection/data/coco.names").read().strip().split("\n")
print("Class names loaded:", class_names)

# Load video
video_path = './Real-time-object-detection/Videos/sample_video_1.mp4'  # Update this with your actual video path

# Verify that the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file does not exist at {video_path}")
    sys.exit()
else:
    print(f"Video file found at {video_path}")

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    sys.exit()
else:
    print(f"Video opened successfully at {video_path}")

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = './Real-time-object-detection/results/output_video.avi'
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Create a directory to save the annotation files
output_annotation_dir = './Real-time-object-detection/results/annotations'
if not os.path.exists(output_annotation_dir):
    os.makedirs(output_annotation_dir)

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv4
    boxes = detect(yolo_model, frame)
    
    bbox_xywh = []
    confs = []
    labels = []
    for box in boxes[0]:
        x1 = box[0] * frame.shape[1]
        y1 = box[1] * frame.shape[0]
        x2 = box[2] * frame.shape[1]
        y2 = box[3] * frame.shape[0]
        conf = box[4]
        label = int(box[6])
        
        bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        confs.append(conf)
        labels.append(label)

    bbox_xywh = torch.Tensor(bbox_xywh)
    confs = torch.Tensor(confs)
    labels = torch.Tensor(labels)

    # Ensure DeepSORT input format is correct
    if bbox_xywh.numel() > 0:
        outputs, mask_outputs = deepsort.update(bbox_xywh, confs, labels, frame)
    else:
        outputs = []

    # Prepare to write the annotation file for the current frame
    annotation_file_path = os.path.join(output_annotation_dir, f"{frame_id:06d}.txt")
    with open(annotation_file_path, 'w') as f:
        for output in outputs:
            if len(output) == 6:
                bbox = output[:4]
                track_id = int(output[4])
                class_id = int(output[5])
                
                # Ensure class_id is within the range of class_names
                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = 'Unknown'
                
                # Convert bbox format from xyxy to the format used in KITTI annotations (xmin, ymin, xmax, ymax)
                bbox_left = int(bbox[0])
                bbox_top = int(bbox[1])
                bbox_right = int(bbox[2])
                bbox_bottom = int(bbox[3])
                
                # Write the annotation line
                f.write(f"{class_name} 0 0 0 {bbox_left} {bbox_top} {bbox_right} {bbox_bottom} 0 0 0 0 0 0 0 0\n")

                # Draw bounding box and label on frame
                label = f"ID {track_id}: {class_name}"
                color = (255, 0, 0)
                cv2.rectangle(frame, (bbox_left, bbox_top), (bbox_right, bbox_bottom), color, 2)
                cv2.putText(frame, label, (bbox_left, bbox_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Print each detected object with its tracking ID and class name
                print(f"Tracking ID: {track_id}, Class ID: {class_id}, Label: {label}")

    # Write the frame to the output video
    out.write(frame)
    frame_id += 1

cap.release()
out.release()
print(f"Annotations saved in {output_annotation_dir}")
print(f"Output video saved at {output_video_path}")
