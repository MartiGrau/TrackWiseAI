import streamlit as st

def how_it_works():
    # Section Title
    st.markdown("<h1 style='text-align: center; color: white;'>How It Works: Unleash the Benefits</h1>", unsafe_allow_html=True)

    # Columns for features
    col1, col2, col3 = st.columns(3)

    # First Column: Easy to implement
    with col1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        _, im_1, _ = st.columns(3)
        im_1.image("https://cdn-icons-png.flaticon.com/512/2849/2849020.png", width=70)
        st.markdown(
            """
            <h3 style='color: white;'>Effortless Integration and User-Friendly Experience:</h3>
            <p style='color: white;'>
                <strong>Connect Any Video Camera</strong><br>
                • Utilize any type of camera within your physical spaces.<br>
                • Repurpose existing CCTV cameras or add new ones effortlessly.<br>
                <strong>Plug & Play:</strong> Quick and remote functionality for seamless integration.
            </p>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Second Column: Digital twin
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        _, im_2, _ = st.columns(3)
        im_2.image("https://cdn-icons-png.freepik.com/256/7328/7328583.png?semt=ais_hybrid", width=70)
        st.markdown(
            """
            <h3 style='color: white;'>Create a Digital Twin: Act in Real-Time</h3>
            <p style='color: white;'>
                • Experience a virtual replica of your physical space, providing unparalleled insights.<br>
                • Harness real-time data with AI-powered video analytics, accessible from anywhere.<br>
                <strong>Boost Efficiency:</strong> Receive instant alerts and comprehensive reports to streamline your operations.
            </p>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Third Column: Virtual manager
    with col3:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        _, im_3, _ = st.columns(3)
        im_3.image("https://cdn-icons-png.flaticon.com/512/8618/8618942.png", width=70)
        st.markdown(
            """
            <h3 style='color: white;'>Your Virtual Manager at Work</h3>
            <p style='color: white;'>
                • AI-driven agent seamlessly integrated with your business data.<br>
                • Engages with you and your team, offering actionable insights and timely alerts.<br>
                <strong>Optimize Your Business:</strong> Enhance operations and boost conversion rates effortlessly.
            </p>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)