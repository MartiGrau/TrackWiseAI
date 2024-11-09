import streamlit as st
import tempfile
from urllib.parse import urlparse, parse_qs
from ..supervision_scripts.download_from_youtube import download
from ..supervision_scripts.draw_zones import draw_zones
from ..supervision_scripts.inference_file import run_inference

def app():
    st.title('TrackWiseAI :camera:')

    # Upload Video from Youtube
    st.subheader("Upload Video from Youtube")
    st.text("Here are some example Youtube URLs you can use:\n"
            "'https://www.youtube.com/watch?v=-8zyEwAa50Q'")

    youtube_url = st.text_input("Enter the Youtube URL", value="https://www.youtube.com/watch?v=-8zyEwAa50Q")
    if youtube_url:
        st.video(youtube_url)
    
    # -- SAVE YOUTUBE VIDEO -- #  (Only once)
    if st.button("Save Video"):
        # Extract the video ID from the YouTube URL
        query = urlparse(youtube_url).query
        if 'video_id' not in st.session_state:
            st.session_state.video_id = parse_qs(query).get('v', [None])[0]
        video_id = st.session_state.video_id

        if video_id:
            # Create a temporary file with the video ID as the name
            out_path = "/home/mgrau/personal/repos/TrackWiseAI"
            name_file = "video.mp4"
            temp_output_path = "/home/mgrau/personal/repos/TrackWiseAI/video.mp4" # TODO: Modify it
            st.session_state.temp_output_path = temp_output_path

            # Call the download function with the temporary file path
            download(youtube_url, out_path, name_file)
            st.success("Video saved successfully")
        else:
            st.error("Invalid YouTube URL")

    # -- DRAW ZONES -- #
    st.subheader("Draw Zones")
    st.write("Draw zones on the video to track people in specific areas.")
    if st.button("Draw Zones"):
        zone_configuration_path = "/home/mgrau/personal/repos/TrackWiseAI/zone_configuration.json" # TODO: Modify it
        st.session_state.zone_configuration_path = zone_configuration_path
        # Call the draw_zones function
        draw_zones(st.session_state.temp_output_path, zone_configuration_path)
        st.success("Zones drawn successfully")
    
    # -- INFERENCE FILE -- #
    st.subheader("Inference File")
    st.write("Get the final results of the video with the zones drawn.")
    if st.button("Get Inference File"):
        # Display the final video with the zones drawn
        run_inference(zone_configuration_path=st.session_state.zone_configuration_path, source_video_path=st.session_state.temp_output_path)
        st.video(temp_output_path)

    
