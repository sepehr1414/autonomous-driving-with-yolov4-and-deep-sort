import sys
import os
sys.path.append('./Real-time-object-detection/pytorch-YOLOv4')
import torch
import cv2
import numpy as np
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect
from tool.darknet2pytorch import Darknet
from PIL import Image

def load_model():
    model = Darknet("./Real-time-object-detection/pytorch-YOLOv4/cfg/yolov4.cfg")
    model.load_state_dict(torch.load("./Real-time-object-detection/models/yolov4.pth"))
    model.eval()
    return model

def detect(model, img):
    sized = cv2.resize(img, (model.width, model.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(model, sized, 0.4, 0.6, use_cuda=False)
    return boxes

def draw_boxes(img, boxes, class_names):
    width = img.shape[1]
    height = img.shape[0]
    for box in boxes[0]:
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        cls_conf = box[4]
        cls_id = box[6]
        color = (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls_id]}: {cls_conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

if __name__ == "__main__":
    model = load_model()
    class_names = load_class_names("./Real-time-object-detection/data/coco.names")

    img = cv2.imread("./Real-time-object-detection/pytorch-YOLOv4/data/dog.jpg")
    boxes = detect(model, img)
    img = draw_boxes(img, boxes, class_names)
    
    # Save the image to a file and display it
    output_image_path = "./Real-time-object-detection/results/detection_output.jpg"
    cv2.imwrite(output_image_path, img)
    img = Image.open(output_image_path)
    img.show()
