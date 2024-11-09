import streamlit as st

def sidebar():
   #st.sidebar.title('SmartRetail')
    st.sidebar.image("https://play-lh.googleusercontent.com/cgR6vHikzuHGS6LzTrvIsHM6flRvlsAkt1EH3IJFNGOCqBOee-GjZ3Kp1AzLQHVvKg=w600-h300-pc0xffffff-pd", width=300)
    st.sidebar.subheader("Data-Driven Insights for Your Store")
    st.sidebar.markdown(
        """
        <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
            <h3 style="color: #4b4b4b; text-align: center;">SmartRetail</h3>
            <p style="color: #6c757d;">
                SmartRetail is a powerful platform that transforms store data into actionable insights. 
                Track visitor traffic, analyze sales trends, and measure key metrics like conversion rates and average ticket size.
                With real-time dashboards and intuitive visualizations, SmartRetail helps you optimize operations, 
                enhance customer experiences, and drive profitability—all in one simple tool.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    ) 