import subprocess

def run_inference(zone_configuration_path, source_video_path, model_id="yolov8x-640", classes=0, confidence_threshold=0.3, iou_threshold=0.7):
    # TODO: Remove once this is.
    env_path = '/home/mgrau/personal/environments/supervision'
    script_path = '/home/mgrau/personal/repos/supervision/examples/time_in_zone/scripts/inference_file_example.py'
    # Define the command to activate the virtual environment and run the script
    command = (
        f"source {env_path}/bin/activate && python {script_path} "
        f"--zone_configuration_path \"{zone_configuration_path}\" "
        f"--source_video_path \"{source_video_path}\" "
        f"--model_id \"{model_id}\" "
        f"--classes {classes} "
        f"--confidence_threshold {confidence_threshold} "
        f"--iou_threshold {iou_threshold}"
    )
    subprocess.run(command, shell=True, capture_output=True, text=True)
    return True
