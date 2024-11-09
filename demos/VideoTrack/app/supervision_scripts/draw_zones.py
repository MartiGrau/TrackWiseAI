import os
import subprocess
import atexit

def draw_zones(source_path, zone_configuration_path):
    print(source_path)
    print(zone_configuration_path)
    
    # Start Xvfb in the background
    xvfb_process = subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'])
    
    # Ensure Xvfb is terminated when the script exits
    atexit.register(xvfb_process.terminate)
    
    # Set the DISPLAY environment variable
    os.environ["DISPLAY"] = ":99"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    
    # TODO: Remove once this is.
    env_path = '/home/mgrau/personal/environments/supervision'
    script_path = '/home/mgrau/personal/repos/supervision/examples/time_in_zone/scripts/draw_zones.py'
    # Define the command to activate the virtual environment and run the script
    command = f"bash -c 'source {env_path}/bin/activate && python {script_path} --source_path \"{source_path}\" --zone_configuration_path \"{zone_configuration_path}\"'"
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    process.check_returncode()  # This will raise an error if the command failed
    return True
