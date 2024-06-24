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
output_path = './Real-time-object-detection/results/output_video.avi'
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv4
    boxes = detect(yolo_model, frame)
    
    # Debugging: print detection boxes
    print("Boxes:", boxes)
    
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
        
        # Debugging: Print each detected box and its class ID
        print(f"Detected box: {box}, Class ID: {label}, Class name: {class_names[label] if 0 <= label < len(class_names) else 'Unknown'}")

        bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        confs.append(conf)
        labels.append(label)

    bbox_xywh = torch.Tensor(bbox_xywh)
    confs = torch.Tensor(confs)
    labels = torch.Tensor(labels)

    # Debugging: print DeepSORT inputs
    print("bbox_xywh:", bbox_xywh)
    print("confs:", confs)
    print("labels:", labels)

    # Ensure DeepSORT input format is correct
    if bbox_xywh.numel() > 0:
        outputs = deepsort.update(bbox_xywh, confs, labels, frame)
    else:
        outputs = []

    # Debugging: print the outputs
    print("Outputs:", outputs)

    # Check if outputs is not empty and has the expected structure
    if len(outputs) == 0:
        print("No tracks found.")
    else:
        # Flatten the list of outputs if necessary
        if isinstance(outputs[0], np.ndarray):
            outputs = [output for sublist in outputs for output in sublist]
        else:
            outputs = outputs

        # Use a set to keep track of unique track IDs
        unique_track_ids = set()

        for output in outputs:
            if len(output) == 6:
                bbox = output[:4]
                track_id = int(output[4])
                class_id = int(output[5])
                color = (255, 0, 0)
                
                # Ensure class_id is within the range of class_names
                if 0 <= class_id < len(class_names):
                    label = f"ID {track_id}: {class_names[class_id]}"
                else:
                    label = f"ID {track_id}: Unknown"

                # Debugging: Print the label for each tracked object and check for duplicate track IDs
                print(f"Tracking ID: {track_id}, Class ID: {class_id}, Label: {label}")
                if track_id in unique_track_ids:
                    print(f"Warning: Duplicate track ID {track_id} found!")
                else:
                    unique_track_ids.add(track_id)

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                print("Unexpected output format:", output)

    # Write the frame to the output video
    out.write(frame)

cap.release()
out.release()

print(f"Output video saved at {output_path}")
