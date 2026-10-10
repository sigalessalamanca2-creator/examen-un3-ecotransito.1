 #TAREA 1: INGESTA, DIAGNÓSTICO Y SUSTITUCIÓN DE CENTINELAS

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Sistema EcoTránsito CDMX",
    layout="wide"
)

st.title("Sistema EcoTránsito CDMX")
st.write("¡Mi dashboard está funcionando!")

 
# 1. Cargar o definir el dataset
data = {
    'ID_Registro': ['REG-001', 'REG-002', 'REG-003', 'REG-004', 'REG-005',
                    'REG-006', 'REG-007', 'REG-008', 'REG-009', 'REG-010', 'REG-010'],
    'Timestamp': ['2026-10-01 08:00:00', '2026-10-01 09:00:00', '2026-10-01 10:00:00',
                  '2026-10-01 11:00:00', '2026-10-01 12:00:00', '2026-10-01 13:00:00',
                  '2026-10-01 14:00:00', '2026-10-01 15:00:00', '2026-10-01 16:00:00',
                  '2026-10-01 17:00:00', '2026-10-01 17:00:00'],
    'Zona_Estacion': ['Centro_Historico', 'centro_historico', 'CENTRO_HISTORICO',
                      'Zona_Oriente', 'Zona_Oriente', 'zona_oriente', 'Zona_Sur',
                      'Zona_Sur', 'Zona_Sur', 'Zona_Sur', 'Zona_Sur'],
    'Concentracion_PM25': [35.4, 42.1, -999.0, 120.5, np.nan, 310.0, 18.2, 20.1, 19.5, 19.5, 19.5],
    'Nivel_NO2': [42.1, 48.5, 51.0, 89.2, 82.0, 195.0, 22.4, -999.0, 25.1, 25.1, 25.1],
    'Indice_Trafico_Vehicular': [85.0, 92.5, 88.0, 98.0, 91.2, 15.0, 45.0, 42.1, np.nan, 41.0, 41.0],
    'Estado_Sensor': ['Normal', 'Normal', 'Alerta', 'Alerta', 'Normal', 'Falla', 'Normal',
                      'Normal', 'Normal', 'Normal', 'Normal']
}

df_raw_data = pd.DataFrame(data)

# Diagnostico de duplicados con centinelas
print(df_raw_data.duplicated().sum())
print((df_raw_data[["Concentracion_PM25", "Nivel_NO2", "Indice_Trafico_Vehicular"]] == -999.0).sum())

# Sustituri -999.0 por np.nan y eliminar los duplos 
datos = df_raw_data.replace(-999.0, np.nan)
datos = datos.drop_duplicates()


print("Estos son" + str(datos))
print(datos.shape)

# T2

from sklearn.impute import KNNImputer
 
df_clean = datos.copy()

datos["Zona_Estacion"] = datos["Zona_Estacion"].str.lower()

print("2. Estos son" + str(datos))

df_clean["Timestamp"] = pd.to_datetime(df_clean["Timestamp"])
df_clean["Hora_Dia"] = df_clean["Timestamp"].dt.hour

Q1 = df_clean["Concentracion_PM25"].quantile(0.25)
Q2 = df_clean["Concentracion_PM25"].quantile(0.75)

IQR = Q2 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q2 + 1.5 * IQR

condicion = (

    (df_clean["Concentracion_PM25"] < limite_inferior) |
    (df_clean["Concentracion_PM25"] > limite_superior)

)

df_clean.loc[condicion, "Concentracion_PM25"] = np.nan

print ("3.Estos son" + str(df_clean))


from sklearn.impute import KNNImputer

columnas_numericas = df_clean.select_dtypes(include=['number']).columns

imputer = KNNImputer(n_neighbors=2)

datos_imputados = imputer.fit_transform(df_clean[columnas_numericas])

df_clean[columnas_numericas] = datos_imputados

print("4. DataFrame Final con Imputación k-NN lista:")
print(df_clean)

# Estadísticas descriptivas de PM2.5
st.subheader("Estadísticas de PM2.5")
st.write(df_clean["Concentracion_PM25"].describe())

columna = df_clean["Concentracion_PM25"]

st.write("Media:", columna.mean())
st.write("Mediana:", columna.median())
st.write("Desv. estándar:", columna.std())
st.write("Skewness:", columna.skew())
st.write("Kurtosis:", columna.kurt())

# Matriz de correlación de Spearman
st.subheader("Correlación de Spearman")
st.dataframe(df_clean.corr(method="spearman", numeric_only=True))

# Matriz de correlación de Pearson
st.subheader("Correlación de Pearson")
st.dataframe(df_clean.corr(method="pearson", numeric_only=True))
df = df_clean.copy()

st.sidebar.title("Filtros")

zonas = df["Zona_Estacion"].unique()

zonas_seleccionadas = st.sidebar.multiselect(
    "Selecciona las zonas:",
    options=zonas,
    default=zonas
)

df_filtrado = df[df["Zona_Estacion"].isin(zonas_seleccionadas)]

df_filtrado = df[
    df["Zona_Estacion"].isin(zonas_seleccionadas)
]

st.subheader("3. Métricas principales (KPIs)")

total_registros = len(df_filtrado)

pm25_promedio = df_filtrado["Concentracion_PM25"].mean()

trafico_promedio = df_filtrado["Indice_Trafico_Vehicular"].mean()

sensores_falla = (
    df_filtrado["Estado_Sensor"]
    .str.strip()
    .str.lower()
    .eq("falla")
    .sum()
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total de registros",
        value=total_registros
    )

with col2:
    st.metric(
        label="PM2.5 promedio",
        value=(
            f"{pm25_promedio:.1f} µg/m³"
            if pd.notna(pm25_promedio)
            else "Sin datos"
        )
    )

with col3:
    st.metric(
        label="Tráfico promedio",
        value=(
            f"{trafico_promedio:.1f}"
            if pd.notna(trafico_promedio)
            else "Sin datos"
        )
    )

with col4:
    st.metric(
        label="Sensores en falla",
        value=int(sensores_falla)
    )

st.subheader("4. Visualización de datos")

tab1, tab2, tab3 = st.tabs([
    "Evolución de PM2.5",
    "Comparación por zonas",
    "Correlaciones"
])

with tab1:
    st.write("Concentración de PM2.5 a lo largo del tiempo")

    df_grafica = df_filtrado.dropna(
        subset=["Timestamp", "Concentracion_PM25"]
    ).copy()

    if not df_grafica.empty:
        df_grafica["Timestamp"] = pd.to_datetime(
            df_grafica["Timestamp"],
            errors="coerce"
        )
        df_grafica = df_grafica.dropna(subset=["Timestamp"])

        fig1 = px.line(
            df_grafica,
            x="Timestamp",
            y="Concentracion_PM25",
            color="Zona_Estacion",
            markers=True,
            title="Evolución temporal de PM2.5",
            labels={
                "Timestamp": "Fecha y hora",
                "Concentracion_PM25": "PM2.5 (µg/m³)",
                "Zona_Estacion": "Zona"
            }
        )

        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No hay datos disponibles para esta gráfica.")

with tab2:
    st.write("Promedio de PM2.5 por zona")

    df_zonas = (
        df_filtrado
        .groupby("Zona_Estacion", as_index=False)
        ["Concentracion_PM25"]
        .mean()
        .dropna()
    )

    if not df_zonas.empty:
        fig2 = px.bar(
            df_zonas,
            x="Zona_Estacion",
            y="Concentracion_PM25",
            color="Zona_Estacion",
            title="Concentración media de PM2.5 por zona",
            labels={
                "Zona_Estacion": "Zona",
                "Concentracion_PM25": "PM2.5 promedio (µg/m³)"
            }
        )

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No hay datos disponibles para comparar las zonas.")

with tab3:
    st.write("Relaciones entre las variables numéricas")

    columnas_numericas = df_filtrado.select_dtypes(
        include="number"
    )

    if (
        not columnas_numericas.empty
        and len(columnas_numericas.columns) >= 2
    ):
        correlacion = columnas_numericas.corr(
            method="spearman"
        )

        fig3, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            correlacion,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            vmin=-1,
            vmax=1,
            ax=ax
        )

        ax.set_title("Mapa de calor de correlaciones")
        plt.tight_layout()

        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info(
            "Se necesitan al menos dos columnas numéricas "
            "para calcular las correlaciones."
        )



#LO HICEEEEEEE GAAAAA  :b