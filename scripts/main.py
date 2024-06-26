import sys
import os

# Add project directory to system path
sys.path.append('./Real-time-object-detection/scripts')

from deep_sort_tracking import process_video

video_path = './Real-time-object-detection/Videos/sample_video_2.mp4'
output_path = './Real-time-object-detection/results/output_video.avi'
label_dir = './Real-time-object-detection/kitti_labels/'
results_dir = './Real-time-object-detection/results/'

# Process video and get predictions
predictions = process_video(video_path, output_path)
