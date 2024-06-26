import sys
import os
import numpy as np
sys.path.append('./Real-time-object-detection/deep_sort')
sys.path.append('./Real-time-object-detection/pytorch-YOLOv4')
import cv2
import torch
from deep_sort.deep_sort import DeepSort
from yolov4_detection import load_model, detect

def process_video(video_path, output_path):
    # Initialize YOLOv4 and DeepSORT
    yolo_model = load_model()
    deepsort = DeepSort("./Real-time-object-detection/deep_sort/ckpt/ckpt.t7")

    # Load class names
    class_names = open("./Real-time-object-detection/data/coco.names").read().strip().split("\n")

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

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

        if bbox_xywh.numel() > 0:
            outputs, _ = deepsort.update(bbox_xywh, confs, labels, frame)
            for output in outputs:
                predictions.append((frame_count, class_names[int(output[5])], [output[0], output[1], output[2], output[3]]))

        # Draw and write the frame to the output video
        for output in outputs:
            if len(output) == 6:
                bbox = output[:4]
                track_id = int(output[4])  # Tracking ID assigned by DeepSORT
                class_id = int(output[5])  # Class ID from detection
                color = (255, 0, 0)

                # Ensure class_id is within the range of class_names
                if 0 <= class_id < len(class_names):
                    label = f"{class_names[class_id]}"
                else:
                    label = f"ID {track_id}: Unknown"

                # Print the frame number, class name, and tracking ID
                print(f"Frame {frame_count} -  Class ID: {class_id}, Label: {label}")

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                print(f"Frame {frame_count} - Unexpected output format: {output}")


        out.write(frame)

    cap.release()
    out.release()
    
    return predictions
