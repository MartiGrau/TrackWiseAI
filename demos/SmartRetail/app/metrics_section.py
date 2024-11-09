import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import pandas as pd
load_dotenv()

def metrics(df_entradas, df_salidas, df_compras):
    # Metrics analysis
    st.header("Metrics Analysis")
    df_traffic = pd.concat([
        df_entradas.assign(event="entrada"),
        df_salidas.assign(event="salida")
    ])
    df_traffic["datetime"] = pd.to_datetime(df_traffic["datetime"])
    df_compras["datetime"] = pd.to_datetime(df_compras["datetime"])

    # Date selection
    min_date = df_traffic["datetime"].min().date()
    max_date = df_traffic["datetime"].max().date()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Select start date", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("Select end date", max_date, min_value=min_date, max_value=max_date)

    # Filter data based on selected dates
    df_traffic_filtered = df_traffic[(df_traffic["datetime"].dt.date >= start_date) & (df_traffic["datetime"].dt.date <= end_date)]
    df_compras_filtered = df_compras[(df_compras["datetime"].dt.date >= start_date) & (df_compras["datetime"].dt.date <= end_date)]
    # ---------------------------- #
    # Key metrics
    total_visitors = len(df_traffic_filtered[df_traffic_filtered["event"] == "entrada"])
    total_sales = df_compras_filtered["amount"].sum()
    avg_ticket = df_compras_filtered["amount"].mean()

    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Visitors", total_visitors)
    with col2:
        st.metric("Total Sales (€)", f"€{total_sales:,.2f}")
    with col3:
        st.metric("Average Ticket (€)", f"€{avg_ticket:,.2f}")


    # Hourly traffic
    hourly_traffic = df_traffic_filtered.groupby(df_traffic_filtered["datetime"].dt.floor("h")).size().reset_index(name="traffic_count")
    hourly_sales = df_compras_filtered.groupby(df_compras_filtered["datetime"].dt.floor("h"))["amount"].sum().reset_index(name="sales_total")

    # Hourly traffic visualization
    st.subheader("Hourly Traffic")
    st.line_chart(data=hourly_traffic, x="datetime", y="traffic_count", x_label="Hour", y_label="Number of People", use_container_width=True)

    # Hourly sales visualization
    st.subheader("Hourly Sales")
    st.line_chart(data=hourly_sales, x="datetime", y="sales_total", x_label="Hour", y_label="Sales Amount (€)", color="#ffa500", use_container_width=True)

    # Chatbot Section
    # Initialize OpenAI client
    client = OpenAI(
        # Fetch API key from environment variables
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    st.subheader("Ask the Chatbot")

    # User input for chatbot using st.chat_input
    user_question = st.chat_input(placeholder="Ask a question about the traffic or sales data:")

    if user_question:
        # Prepare the context for the chatbot
        context = f"""
        You are analyzing retail data. The total number of visitors is {total_visitors}, 
        total sales amount to €{total_sales:,.2f}, and the average ticket size is €{avg_ticket:,.2f}.
        The data is filtered between {start_date} and {end_date}.
        """

        # Call OpenAI API to get the response
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_question},
                ],
                model="gpt-3.5-turbo",  # Replace with "gpt-4" if needed
            )

            # Access the response correctly
            response_message = chat_completion.choices[0].message.content.strip()

            # Display the response with enhanced design and robot icon
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; display: flex; align-items: center;">
                    <img src="https://img.icons8.com/ios-filled/50/000000/robot.png" alt="Robot Icon" style="width: 30px; height: 30px; margin-right: 10px;">
                    <div style="color: #333; font-size: 16px;">
                        <strong>Smart Retail:</strong> {response_message}
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")