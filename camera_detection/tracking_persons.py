from ultralytics import YOLO
import cv2
import argparse

# Function to track objects in video using YOLO model
def track_objects(model_path, video_path, output_path, conf):
    model = YOLO(model_path)
    results = model.track(video_path, persist=True, stream=True, conf=conf, task='detect')

    max_track_id = 0

    cap = cv2.VideoCapture(video_path)
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MPEG'), 25, (int(cap.get(3)), int(cap.get(4))))

    for result in results:
        summary = result.summary()
        for s in summary:
            if 'track_id' in s and 'name' in s and s['track_id'] > max_track_id and s['name'] == 'person':
                max_track_id = s['track_id']
        tracked_frame = result.plot()
        output.write(tracked_frame)
        cv2.imshow('frame', tracked_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    output.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking video complete...")
    print(f"There are {max_track_id} people in the video")

# Main function to parse arguments and run the script
def main():
    parser = argparse.ArgumentParser(description="Track objects in video using YOLO model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLO model weights file')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_path', type=str, default='output.avi', help='Path to save the output video file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for tracking')

    args = parser.parse_args()

    track_objects(model_path=args.model_path, video_path=args.video_path, output_path=args.output_path, conf=args.conf)

if __name__ == "__main__":
    main()
