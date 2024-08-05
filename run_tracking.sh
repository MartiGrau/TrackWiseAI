#!/bin/bash

# Run the Python script with arguments
python3 your_script.py \
  --model_path './training/yolov8n/train/weights/best.pt' \
  --video_path './Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4' \
  --output_path 'output.avi' \
  --conf 0.25
