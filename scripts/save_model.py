import sys
import os
sys.path.append('./Real-time-object-detection/pytorch-YOLOv4')
from tool.darknet2pytorch import Darknet
import torch

def save_model(weights_path, output_path, img_size):
    model = Darknet('./Real-time-object-detection/pytorch-YOLOv4/cfg/yolov4.cfg')
    model.load_weights(weights_path)
    model.eval()
    torch.save(model.state_dict(), output_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <weights_path> <output_path> <img_size>")
        sys.exit(1)
    
    weights_path = sys.argv[1]
    output_path = sys.argv[2]
    img_size = int(sys.argv[3])
    
    save_model(weights_path, output_path, img_size)
