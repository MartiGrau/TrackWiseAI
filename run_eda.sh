#!/bin/bash

# Run the Python script with arguments
python3 camera_detection/exploratory_data_analysis.py \
    --classes person \
    --max_samples 10000 \
    --export_dir /hdd1/Datasets/Public/Raw/COCO_dataset
