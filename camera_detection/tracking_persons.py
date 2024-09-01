from collections import defaultdict
from ultralytics import YOLO
import numpy as np
import cv2
import argparse
import os
import subprocess
import tempfile


# Function to track objects in video using YOLO model
def track_objects(model_path, video_path, output_path, conf):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])

    # Ensure the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Read video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a temporary file for the initial output
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
        temp_output_path = temp_output.name

    # Create a video writer object to save the initial output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=conf)
            boxes = results[0].boxes.xywh.cpu()
            track_ids = (
                results[0].boxes.id.int().cpu().tolist()
                if results[0].boxes.id is not None
                else None
            )
            annotated_frame = results[0].plot()
            # plot the tracks
            if track_ids:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)
                    # draw the tracking lines
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=2,
                    )

            # Write the annotated frame to the output video
            vid_writer.write(annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()

    # Convert the video to H.264 codec using ffmpeg
    ffmpeg_command = [
        'ffmpeg',
        '-i', temp_output_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-y',  # Overwrite output file if it exists
        output_path
    ]
    subprocess.run(ffmpeg_command, check=True)

    # Remove the temporary file
    os.remove(temp_output_path)

    print(f"Video processing complete. Output saved to {output_path}")

"""
def track_objects(model_path, video_path, output_path, conf):
    model = YOLO(model_path)
    results = model.track(video_path, persist=True, stream=True, conf=conf, task='detect')

    max_track_id = 0

    cap = cv2.VideoCapture(video_path)
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (int(cap.get(3)), int(cap.get(4))))

    for result in results:
        summary = result.summary()
        for s in summary:
            if 'track_id' in s and 'name' in s and s['track_id'] > max_track_id and s['name'] == 'person':
                max_track_id = s['track_id']
        tracked_frame = result.plot()
        output.write(tracked_frame)
        # Comment out cv2.imshow if running in a headless environment
        # cv2.imshow('frame', tracked_frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    output.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking video complete...")
    print(f"There are {max_track_id} people in the video")
"""

# Main function to parse arguments and run the script
def main():
    parser = argparse.ArgumentParser(description="Track objects in video using YOLO model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLO model weights file')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='Path to save the output video file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for tracking')

    args = parser.parse_args()

    track_objects(model_path=args.model_path, video_path=args.video_path, output_path=args.output_path, conf=args.conf)

if __name__ == "__main__":
    main()
