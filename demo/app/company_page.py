import streamlit as st

def app():
    st.title('Company Configuration')
    st.write('Configure your company details here.')
    
    # Create two columns
    col1, col2 = st.columns(2)

    # Company Name Input in the first column
    with col1:
        company_name = st.text_input("Company Name", placeholder="Enter your company name")

    # Location Input in the second column
    with col2:
        location = st.text_input("Location", placeholder="Enter company location")

    # Add a divider for better visual separation
    st.divider()

    # Display current configuration
    st.subheader("Current Configuration")
    if company_name or location:
        st.write(f"Company Name: {company_name}")
        st.write(f"Location: {location}")
    else:
        st.info("No configuration set yet. Please enter your company details above.")

    if st.button('Save Configuration'):
        st.success('Configuration saved!')

        # Check if location is provided
        if location:
            # Geocode the location to get latitude and longitude
            # For simplicity, we'll use a placeholder lat/long
            # In a real application, you'd use a geocoding service here
            latitude, longitude = 0, 0  # Placeholder values
            
            # Display the map
            st.subheader("Company Location")
            st.map(data=None, latitude=latitude, longitude=longitude, zoom=10)
        else:
            st.info("Enter a location to see it on the map.")
