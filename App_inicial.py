import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import streamlit as st
from fuzzywuzzy import process, fuzz
import matplotlib.pyplot as plt
import seaborn as sns

st.title("MVP - Plataforma de Business Intelligence para PYMEs")

# Subida del archivo
uploaded_file = st.file_uploader("Por favor cargar el archivo en formato .XLSX", type="xlsx")

if uploaded_file:
    st.session_state['uploaded_file'] = uploaded_file

# Botón para reiniciar la app
if st.button("Reiniciar la app"):
    for key in ['uploaded_file', 'selected_sheet', 'dataframe', 'sheet_loaded']:
        if key in st.session_state:
            del st.session_state[key]
    st.write("El archivo ha sido borrado. Por favor cargue un archivo nuevo.")
    st.rerun()

# Mostrar hojas disponibles si hay archivo cargado
if 'uploaded_file' in st.session_state:
    xls = pd.ExcelFile(st.session_state['uploaded_file'])
    sheet_names = xls.sheet_names
    selected_sheet = st.selectbox("Seleccionar hoja activa", sheet_names)

    # Cargar hoja activa solo si no se ha cargado antes o si cambia
    if st.button("Cargar hoja activa"):
        dataframe = pd.read_excel(st.session_state['uploaded_file'], sheet_name=selected_sheet)
        st.session_state['selected_sheet'] = selected_sheet
        st.session_state['dataframe'] = dataframe
        st.session_state['sheet_loaded'] = True

# Mostrar contenido si ya se cargó hoja activa
if st.session_state.get('sheet_loaded', False):
    df = st.session_state['dataframe']
    st.subheader(f"Hoja cargada: {st.session_state['selected_sheet']}")
    st.write("Vista previa de los datos:")
    st.write(df.head())

    # Análisis exploratorio
    st.subheader("Análisis Exploratorio")
    if st.checkbox("Mostrar estadísticas descriptivas"):
        st.write(df.describe())

    if st.checkbox("Mostrar tipos de datos"):
        st.write(df.dtypes)

    if st.checkbox("Mostrar columnas con valores nulos"):
        null_cols = df.isnull().sum()
        st.write(null_cols[null_cols > 0])

    # ——— Análisis evolución temporal ———
    st.subheader("Análisis de la evolución temporal de la variable objetivo")

    # Selección de variable objetivo y temporal
    target_variable = st.selectbox("Seleccionar variable objetivo", df.columns, key="tv")
    time_variable = st.selectbox("Seleccionar variable temporal", df.columns, key="tv2")

    # Selección de filtro categórico y dejar la opcion de todos

    # Seleccionar la variable categórica para filtrar
    categorical_variable = st.selectbox(
        "Seleccionar variable categórica para filtrar", 
        df.columns, key="cat_var"
    )

    # Obtener sus valores únicos y permitir multiselección (todos por defecto)
    unique_vals = df[categorical_variable].dropna().unique().tolist()
    selected_vals = st.multiselect(
        f"Seleccionar valor(es) de '{categorical_variable}'",
        options=unique_vals,
        default=unique_vals,
        key="cat_vals"
    )

    

    # 1) Convertir la columna temporal a datetime
    df[time_variable] = pd.to_datetime(df[time_variable], errors='coerce')

    # 2) Filtrar filas según la categoría seleccionada
    df_cat = df[df[categorical_variable].isin(selected_vals)]

    # 3) Seleccionar sólo las dos columnas relevantes y quitar NaN
    df_filtered = (
        df_cat[[time_variable, target_variable]]
        .dropna()
    )

    # 4) Agrupar y ordenar (suma o promedio según tu elección)
    agg_type = st.radio("Tipo de agregación", ["Promedio", "Suma"], index=1)
    if agg_type == "Promedio":
        time_series = df_filtered.groupby(time_variable)[target_variable].mean()
    else:
        time_series = df_filtered.groupby(time_variable)[target_variable].sum()
    time_series = time_series.sort_index()

    # 5) Graficar
    st.line_chart(time_series)

    # ——— Después de st.line_chart(time_series) ———

    # 6) Calcular variaciones sobre la serie filtrada
    variation = time_series.pct_change().fillna(0).replace([np.inf, -np.inf], 0)

    # 7) Identificar fechas de máximo y mínimo cambio porcentual
    if not variation.empty:
        max_gain_date = variation.idxmax()
        max_loss_date = variation.idxmin()

        st.write("📈 **Mayor ganancia en el filtro seleccionado:**")
        st.write(f"- Fecha: {max_gain_date.date()}")
        st.write(f"- Valor objetivo: {time_series.loc[max_gain_date]:.2f}")
        st.write(f"- Cambio porcentual: {variation.loc[max_gain_date]:.2%}")

        st.write("📉 **Mayor pérdida en el filtro seleccionado:**")
        st.write(f"- Fecha: {max_loss_date.date()}")
        st.write(f"- Valor objetivo: {time_series.loc[max_loss_date]:.2f}")
        st.write(f"- Cambio porcentual: {variation.loc[max_loss_date]:.2%}")
    else:
        st.warning("La serie temporal está vacía: revisa tus filtros o el DataFrame original.")

        
            # cd "C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Emprendimiento"
            #  streamlit run App_inicial.py