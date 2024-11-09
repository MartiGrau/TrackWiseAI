import cv2
import os
import tempfile
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
import subprocess

def video_conversion(temp_output_path, final_output_path):
    """ 
    Convert the video to H.264 codec using ffmpeg
    """
    ffmpeg_command = [
        'ffmpeg',
        '-i', temp_output_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-y',  # Overwrite output file if it exists
        final_output_path
    ]
    subprocess.run(ffmpeg_command, check=True)

    # Remove the temporary file
    os.remove(temp_output_path)



def track_persons(model, video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])

    frame_count = 0
    objects_per_frame = []
    total_objects = set()

    # Ensure the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, 0

    # Read video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
            frame_count += 1
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xywh.cpu()
            track_ids = (
                results[0].boxes.id.int().cpu().tolist()
                if results[0].boxes.id is not None
                else None
            )
            annotated_frame = results[0].plot()
            # plot the tracks
            if track_ids:
                objects_per_frame.append((frame_count, len(track_ids)))
                total_objects.update(track_ids)
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 300:  # retain 300 tracks for 300 frames (10 seconds at 30 fps)
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

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()

    # Create a temporary file for the final output
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as final_output:
        final_output_path = final_output.name
    
    # Call the video conversion function
    video_conversion(temp_output_path, final_output_path)

    print(f"Video processing complete. Output saved to {final_output_path}")
    return final_output_path, objects_per_frame, len(total_objects), fps

