import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO


modelo = joblib.load("IMPLEMENTACION/modelo_xgboost.pkl")

#Creamos un diccionario para cambiar las predicciones a palabras
classes = {0:"Baja", 1:"Media", 2:"Alta"}

#Declaramos desde aqui las columnas necesarias para el analisis
columnas_necesarias = ["year", "selling_price", "transmission", "potencia_motor_hp", "nivel_seguridad", "score_calidad", "eficiencia_km_l"]

#Colocamos un titulo dentro de la pagina
st.title("Prediccion de Calidad de Autos")
st.write("Esta aplicación predice la calidad del auto en base a sus características.")

#La pagina cuenta con dos modos una para meter el excel y otra para ingresar manualmente los datos
modo = st.radio("Selecciona el modo de uso:", ["Ingreso manual", "Cargar archivo Excel"])


#---------------------------CREAMOS EL MODO 1 --------------------------------
if modo == "Ingreso manual":
    st.subheader("Ingreso manual de caracteristicas")

    #Ponemos todos los atributos necesarios para el analisis
    year = st.number_input("Año del auto", min_value=1900, max_value=2025, value=2015)
    selling_price = st.number_input("Precio de venta", min_value=10000, value=60000) #Tomando en cuenta que en el excel el valor maximo es de 8900000
    transmission = st.selectbox("Transmisión", options=[0, 1], format_func=lambda x: "Manual" if x == 0 else "Automática")
    potencia_motor_hp = st.number_input("Potencia del motor (HP)", min_value=50, value=300)#Tomando en cuenta que en el excel el valor maximo es de 400
    nivel_seguridad = st.number_input("Nivel de seguridad", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
    score_calidad = st.number_input("Score de calidad", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    eficiencia_km_l = st.number_input("Eficiencia (km/l)", min_value=5.0, max_value=25.0, value=12.0, step=0.1)

    #Colocamos boton para realizar la prediccion
    if st.button("Predecir calidad"):
        entrada = np.array([[year, selling_price, transmission, potencia_motor_hp, nivel_seguridad, score_calidad, eficiencia_km_l]])
        pred = modelo.predict(entrada)
        st.success(f"Calidad predicha: {classes[int(pred[0])]}")

else:
    st.subheader("Cargar archivo Excel para clasificación masiva")
    archivo = st.file_uploader("Selecciona un archivo .xlsx", type=["xlsx"])

    if archivo is not None:
        df = pd.read_excel(archivo)
        st.write("Vista previa del archivo cargado:")
        st.dataframe(df.head())

        #Asi podemos checar que todos las columnas en nuestro Df tenga las columnas necesarias para el analisis
        #Basicamente lo que se hace es predecir toda los datos y irlo escribiendo en el excel
        if all(col in df.columns for col in columnas_necesarias):
            entrada = df[columnas_necesarias]
            predicciones = modelo.predict(entrada)
            df["calidad_auto_predicha"] = [classes[int(x)] for x in predicciones]

            st.success("Predicciones realizadas exitosamente.")
            st.dataframe(df)

            #En esta parte creamos funciones para la generacion del excel
            @st.cache_data
            def convertir_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                return output.getvalue()

            datos_excel = convertir_excel(df)
            #Creamos nuestro boton para descargar el excel con su respectivo nombre
            st.download_button(
                label="Descargar Excel con predicciones",
                data=datos_excel,
                file_name="autos_clasificados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.error("El archivo debe contener las columnas necesarias:")
            st.code(", ".join(columnas_necesarias))
    else:
        st.warning("Agrega un Archivoo")
