#!/bin/bash

# Run the Python script with arguments
python3 your_script.py \
  --classes person \
  --max_samples 10000 \
  --export_dir ./yolov5-coco-datasets \
  --models yolov8n yolov8s yolov8m \
  --data_yaml ./yolov5-coco-datasets/dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --device 0 \
  --batch 8 \
  --project ./training \
  --datatest \
    https://cdn.antaranews.com/cache/1200x800/2023/10/13/Pengendara-Sepeda-Motor-Trotoar-060323-aaa-5.jpg \
    https://img.harianjogja.com/posts/2022/11/14/1117643/jalur-pedestrian-malioboro.jpg \
    https://assets.bwbx.io/images/users/iqjWHBFdfxIU/i.KTm08H6tuM/v1/1200x810.jpg \
    https://static.promediateknologi.id/crop/0x0:0x0/0x0/webp/photo/radarjogja/2023/01/web-JOG-Pedestrian-Harus-Sesuai-Fungsinya-FAT-010122.jpg
