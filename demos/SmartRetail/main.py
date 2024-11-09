import streamlit as st
import pandas as pd
from app.sidebar_section import sidebar
from app.how_it_work_section import how_it_works
from app.metrics_section import metrics

st.set_page_config(layout="wide")
BASE_PATH = "sample_data/"

def main():
    # Sidebar section
    sidebar()

    # How it works section  
    how_it_works()

    # Load data
    df_entradas, df_salidas, df_compras = load_csv_data()
    metrics(df_entradas, df_salidas, df_compras)



@st.cache_resource
def load_csv_data():
    # Cargar los datos
    df_entradas = pd.read_csv(f"{BASE_PATH}entradas_extended.csv")
    df_salidas = pd.read_csv(f"{BASE_PATH}salidas_extended.csv")
    df_compras = pd.read_csv(f"{BASE_PATH}compras_extended.csv")
    return df_entradas, df_salidas, df_compras

if __name__ == "__main__":
    main()



