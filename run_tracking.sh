#!/bin/bash

# Run the Python script with arguments
python3 camera_detection/tracking_persons.py \
  --model_path '/hdd1/Checkpoints/TrackWiseAI/person_detection_v1/yolov8m/train/weights/best.pt' \
  --video_path '/home/mgrau/personal/repos/TrackWiseAI/test.mp4' \
  --output_path 'output.mp4' \
  --conf 0.35 \