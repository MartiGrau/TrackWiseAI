import streamlit as st
from ..tracking_algorithm import track_persons
import tempfile
import cv2
import time
import io
from ..analytics import people_detected_over_time
import json

def app(cfg, model):
    st.title('TrackWiseAI :camera:')
    st.write("TrackWiseAI is an advanced computer vision and AI-based analytics system designed to enhance retail store operations by providing comprehensive insights into customer behavior. \
    In this demo, we will be tracking people in a video using the YOLO model.")
    
    # -- Video Upload -- #
    st.subheader("Upload Video")
    
    # Option to select default video or upload a custom one
    video_option = st.radio("Choose video source:", ("Use default video", "Upload custom video"))
    
    if video_option == "Use default video":
        uploaded_file = "/home/mgrau/personal/repos/TrackWiseAI/test.mp4"
        #uploaded_file = open(uploaded_file, "rb")
    else:
        uploaded_file = st.file_uploader("Choose a file", type=["mp4", "avi", "mov", "mkv"])
    
    processed_video_bytes = None

    
    if uploaded_file is not None:
    
        # Display the uploaded video
        st.video(uploaded_file)
        
        # Process the video
        if st.button("Process Video"):
            if video_option == "Upload custom video":
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                    temp_video.write(uploaded_file.read())
                    temp_video_path = temp_video.name
            else:
                temp_video_path = uploaded_file
            
            # Process the video with YOLO
            with st.spinner("Processing video..."):
                processed_video_path, objects_per_frame, total_objects, fps = track_persons(model, temp_video_path)
            
        # Read the processed video into BytesIO
            if processed_video_path:
                with open(processed_video_path, 'rb') as video_file:
                    processed_video_bytes = io.BytesIO(video_file.read())

        if processed_video_bytes:
            st.success("Video processing complete")
            st.video(processed_video_bytes.getvalue(), format="video/mp4")

            # -- ANALYTICS -- #
            st.title("Analytics :bar_chart:")

            st.metric("Total Persons", total_objects)

            # People Detected Over Time
            st.subheader('People Detected Over Time')
            people_per_second = people_detected_over_time(objects_per_frame, fps)

            # Create a line chart showing people detected over time
            st.line_chart(
                data=people_per_second,
                x='second',
                y='people',
                x_label='Time (seconds)',
                y_label='Number of People',
                use_container_width=True
            )

            # Add a description for the chart
            st.caption("This line chart displays the average number of people detected per second throughout the video. This visualization helps identify patterns and fluctuations in crowd density over the course of the video.")


