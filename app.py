import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predicción de Ruido LAEQ",
    page_icon="🔊",
    layout="centered"
)

st.title("🔊 Predicción del Nivel de Ruido LAEQ")

st.markdown("""
Esta aplicación utiliza modelos de Machine Learning para predecir
el nivel de ruido ambiental (LAEQ) en la Municipalidad Distrital de Ate.
""")

modelo_opcion = st.selectbox(
    "Seleccione el modelo:",
    ["Random Forest", "Linear Regression"]
)

if modelo_opcion == "Random Forest":
    modelo = joblib.load("modelos/random_forest_model.pkl")
else:
    modelo = joblib.load("modelos/linear_regression_model.pkl")

st.subheader("Ingrese los datos")

coordenada_x = st.number_input("COORDENADA_X")
coordenada_y = st.number_input("COORDENADA_Y")
eca_ruido = st.number_input("ECA_RUIDO")
laf_min = st.number_input("LAF_MIN")
laf_max = st.number_input("LAF_MAX")
l90 = st.number_input("L90")
provincia = st.number_input("PROVINCIA")
distrito = st.number_input("DISTRITO")
ubigeo = st.number_input("UBIGEO")
fecha_corte = st.number_input("FECHA_CORTE")

datos = pd.DataFrame({
    "COORDENADA_X": [coordenada_x],
    "COORDENADA_Y": [coordenada_y],
    "ECA_RUIDO": [eca_ruido],
    "LAF_MIN": [laf_min],
    "LAF_MAX": [laf_max],
    "L90": [l90],
    "PROVINCIA": [provincia],
    "DISTRITO": [distrito],
    "UBIGEO": [ubigeo],
    "FECHA_CORTE": [fecha_corte]
})

if st.button("Obtener predicción"):
    prediccion = modelo.predict(datos)

    st.success(
        f"🔊 El nivel de ruido estimado (LAEQ) es: {prediccion[0]:.2f} dB"
    )

st.markdown("---")

st.markdown("### 📘 Enlace al Google Colab")

st.markdown(
    "[Abrir cuaderno en Google Colab](https://colab.research.google.com/drive/1iEjieUyCk5I7OoD-914XN075BrQh35gy?usp=sharing)"
)

st.markdown("---")

st.markdown("Desarrollado por Sandro Alfaro")
