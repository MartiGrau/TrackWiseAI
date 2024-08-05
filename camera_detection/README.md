## Features
- Conducted EDA on the COCO 2017 dataset, focusing on the 'person' class.
- Trained multiple YOLOv8 models (yolov8n, yolov8s, yolov8m) using the filtered COCO 2017 - dataset.
- Implemented a video tracking system utilizing the trained YOLOv8 models to detect and monitor individuals in video clips.
- Saved output videos with annotated bounding boxes and unique IDs for each detected person.
- Provided a summary report detailing the total number of people detected in the video.

## Prerequisites
>- Python 3.x
>- Ultralytics YOLO library
>- FiftyOne library
>- OpenCV library
>- Matplotlib library

## Installation
Clone the repository:

```bash
git clone https://github.com/MartiGrau/TrackWiseAI.git
```
Navigate to the project directory:

```bash
cd person-tracking
```
Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### Tracking People in Video
Ensure you have the YOLOv8 model weights and the video file you want to process.
Run the tracking script with the appropriate arguments. Example:
```bash
python3 your_script.py --model_path './training/yolov8n/train/weights/best.pt' --video_path './path_to_video/video.mp4' --output_path 'output.avi' --conf 0.25
```

### Example .sh Script
Create a shell script to run the tracking script easily:

```bash
#!/bin/bash

# Run the Python script with arguments
python3 your_script.py \
  --model_path './training/yolov8n/train/weights/best.pt' \
  --video_path './Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4' \
  --output_path 'output.avi' \
  --conf 0.25
```

## Contributing
Feel free to fork the project and submit pull requests. For significant changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
Thanks to the developers of FiftyOne and Ultralytics YOLO for providing excellent tools for dataset management and object detection.