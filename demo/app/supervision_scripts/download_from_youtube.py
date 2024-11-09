import subprocess

def download(url, output_path, filename):
    # TODO: Remove once this is.
    env_path = '/home/mgrau/personal/environments/supervision'
    script_path = '/home/mgrau/personal/repos/supervision/examples/time_in_zone/scripts/download_from_youtube.py'
    # Define the command to activate the virtual environment and run the script
    print(output_path)
    print(filename)
    command = f"bash -c 'source {env_path}/bin/activate && python {script_path} --url \"{url}\" --output_path \"{output_path}\" --file_name \"{filename}\"'"
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    process.check_returncode()  # This will raise an error if the command failed
    return True
