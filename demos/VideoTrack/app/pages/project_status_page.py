import streamlit as st

def app():
    st.markdown("""
    <div style="background-color: #007BFF; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <h2 style="color: #fff; font-family: 'Arial', sans-serif; text-align: center;">TrackWiseAI</h2>
        <p style="color: #f0f2f6; font-family: 'Arial', sans-serif; font-size: 16px; line-height: 1.6; text-align: justify;">
            TrackWiseAI is an advanced computer vision and AI-based analytics system designed to enhance retail store operations by providing comprehensive insights into customer behavior. 
            Utilizing video cameras installed in stores, TrackWiseAI detects and tracks the number of people, their positions, movement patterns, and time spent in specific areas.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.title('Project Status')
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
        <h4 style="color: #333;">On this page, we will provide detailed information about the dataset used for training, the detection model and its capabilities, the development of the demo, and considerations regarding privacy concerns.</h4>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Information
    st.subheader("Research of Dataset")
    st.write("To train our person detection model, we used the following dataset: [Human Dataset](https://www.kaggle.com/datasets/fareselmenshawii/human-dataset/data).")
    st.image("https://storage.googleapis.com/kaggle-datasets-images/3643278/6329753/07ae1e8a7e70bcd6c2a821c855a260e9/dataset-cover.png?t=2023-09-02-15-05-24", use_column_width=True)
    st.write("Additionally, we found the following data for testing video with people:")
    st.write("- [YouTube Video](https://www.youtube.com/watch?v=KMJS66jBtVQ)")
    st.write("- [VIRAT Video Dataset](https://viratdata.org/)")
    st.write("- [MERL Shopping Dataset](https://www.merl.com/research/license/MERL_Shopping_Dataset)")

    # Model Information
    st.subheader("Detection Model")
    st.write("We used YOLOv8 from Ultralytics to train a model with the source dataset that we found. More information can be found here: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).")
    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png", use_column_width=True)
    st.subheader("Model Capabilities")
    st.write("The model is currently able to:")
    st.write("- Detect all the people")
    st.write("- Identify the people with an ID")
    st.write("- Track their movements in the space that the camera is recording")
    st.write("- Count the total number of people at any given time")

    # Demo Information
    st.subheader("Demo Development")
    st.write("The demo is developed with Streamlit. More information can be found here: [Streamlit](https://streamlit.io/).")

    # Privacy Concerns Information
    st.subheader("Privacy Concerns")
    st.write("For privacy concerns, we've been analyzing the Privacy AI 2026 Europe regulations that will affect software using AI to track people. We could address these concerns by pre-processing the data and removing the faces from the people.")