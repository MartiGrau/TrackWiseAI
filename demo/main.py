import streamlit as st
from app.pages import tracking_page, company_page, project_status_page
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
        st.session_state.page = 'project_status'  # Default to tracking page

    # -- sidebar -- #
    st.sidebar.title('TrackWiseAI')
    st.sidebar.write("TrackWiseAI uses AI to analyze customer behavior in retail stores. This demo tracks people in videos using the YOLO model.")

    # -- Navigation -- #
    if st.sidebar.button('📊 Project Status'):
        st.session_state.page = 'project_status'
    if st.sidebar.button('🎥 Tracking'):
        st.session_state.page = 'tracking'
    if st.sidebar.button('⚙️ Company Configuration'):
        st.session_state.page = 'company'

    # Run app
    if st.session_state.page == 'company':
        company_page.app()
    elif st.session_state.page == 'tracking':
        tracking_page.app(cfg, model)
    elif st.session_state.page == 'project_status':
        project_status_page.app()

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
