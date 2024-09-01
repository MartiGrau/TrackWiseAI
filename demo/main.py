import streamlit as st
from app import tracking_page, company_page
import yaml
from ultralytics import YOLO

def main():
    # -- Setup -- #
    st.set_page_config(layout="wide")
    
    # Load configuration from config.yaml
    cfg = load_config('configuration/config.yaml')
    model = setup_initialization(cfg)

    # Initialize session state for page if not already set
    if 'page' not in st.session_state:
        st.session_state.page = 'tracking'  # Default to tracking page

    # -- sidebar -- #
    st.sidebar.title('TrackWiseAI')
    st.sidebar.write("TrackWiseAI is an advanced computer vision and AI-based analytics system designed to enhance retail store operations by providing comprehensive insights into customer behavior. \
    In this demo, we will be tracking people in a video using the YOLO model.")

    # -- Navigation -- #
    if st.sidebar.button('Company Configuration'):
        st.session_state.page = 'company'
    elif st.sidebar.button('Tracking'):
        st.session_state.page = 'tracking'

    # Run app
    if st.session_state.page == 'company':
        company_page.app()
    elif st.session_state.page == 'tracking':
        tracking_page.app(cfg, model)

@st.cache_resource
def load_config(config_path):
    with open(config_path) as file:
        cfg = yaml.safe_load(file)
    return cfg

@st.cache_resource
def setup_initialization(cfg):
    """
    This is initialized only at the beginning
    """
    model = YOLO(cfg['model_path']).to(cfg['model_device'])
    return model

if __name__ == "__main__":
    main()
