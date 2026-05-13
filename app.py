"""
================================================================
PREDICTOR DE NOTAS - Aplicación Streamlit
================================================================
Carga los modelos pre-entrenados desde 'modelos_rendimiento.pkl'
y permite al usuario predecir su nota a partir de sus hábitos.

Modelos: Random Forest (regresión + clasificación multiclase).

Para ejecutar en local:
    streamlit run app.py
================================================================
"""

import streamlit as st
import pandas as pd
import joblib

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Predictor de notas",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CARGA DE LOS MODELOS (cacheado)
# ============================================================
@st.cache_resource
def cargar_cerebro():
    """Carga el diccionario con los dos modelos, promedios y categorías."""
    return joblib.load('modelos_rendimiento.pkl')

cerebro = cargar_cerebro()
modelo_nota = cerebro['modelo_nota']
modelo_clase = cerebro['modelo_multiclase']
promedios_top = cerebro['promedios_alto_rendimiento']
categorias = cerebro['categorias']
ORDEN_RANGOS = cerebro['orden_rangos']

COLOR_RANGOS = {
    "Muy bajo": "#d62728",
    "Bajo": "#ff7f0e",
    "Aprobado": "#bcbd22",
    "Buena nota": "#2ca02c",
    "Muy buena nota": "#1f77b4",
}

# ============================================================
# CABECERA
# ============================================================
st.title("🎓 Predictor de notas del examen")
st.markdown("""
Esta aplicación predice **la nota que sacarías** y **el rango cualitativo**
en el que caerías (de *Muy bajo* a *Muy buena nota*) a partir de tus
hábitos de estudio, descanso y bienestar.

Ajusta los valores en el panel lateral y observa cómo cambian las predicciones.
""")

# ============================================================
# PANEL LATERAL — ENTRADAS DEL USUARIO
# ============================================================
st.sidebar.header("📋 Tus hábitos")

st.sidebar.markdown("**Datos personales**")
age = st.sidebar.slider("Edad", 16, 35, 20)
gender = st.sidebar.selectbox("Género", categorias['gender'])
academic_level = st.sidebar.selectbox("Nivel académico", categorias['academic_level'])
part_time_job = st.sidebar.radio("¿Tienes trabajo a tiempo parcial?", ["No", "Sí"])
part_time_job_num = 1 if part_time_job == "Sí" else 0

st.sidebar.markdown("---")
st.sidebar.markdown("**Estudio**")
study_hours = st.sidebar.slider("Horas de estudio al día", 0.0, 12.0, 4.0, 0.5)
self_study_hours = st.sidebar.slider("Horas de estudio autónomo", 0.0, 8.0, 2.0, 0.5)
online_classes_hours = st.sidebar.slider("Horas de clases online", 0.0, 8.0, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Descanso y bienestar**")
sleep_hours = st.sidebar.slider("Horas de sueño", 3.0, 12.0, 7.0, 0.5)
mental_health_score = st.sidebar.slider("Salud mental (1=mal, 10=muy bien)", 1, 10, 6)
exercise_minutes = st.sidebar.slider("Minutos de ejercicio al día", 0, 180, 30, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Pantallas**")
social_media_hours = st.sidebar.slider("Horas de redes sociales", 0.0, 10.0, 2.5, 0.5)
screen_time_hours = st.sidebar.slider("Horas de pantalla totales", 0.0, 16.0, 6.0, 0.5)
gaming_hours = st.sidebar.slider("Horas de videojuegos", 0.0, 10.0, 1.0, 0.5)

# ============================================================
# CONSTRUCCIÓN DEL DATAFRAME DE ENTRADA
# ============================================================
entrada = pd.DataFrame([{
    'gender': gender,
    'academic_level': academic_level,
    'age': age,
    'study_hours': study_hours,
    'sleep_hours': sleep_hours,
    'online_classes_hours': online_classes_hours,
    'social_media_hours': social_media_hours,
    'screen_time_hours': screen_time_hours,
    'gaming_hours': gaming_hours,
    'exercise_minutes': exercise_minutes,
    'mental_health_score': mental_health_score,
    'part_time_job': part_time_job_num,
    'self_study_hours': self_study_hours,
}])

# ============================================================
# PREDICCIONES
# ============================================================
nota_pred = float(modelo_nota.predict(entrada)[0])
nota_pred = max(0, min(100, nota_pred))

clase_idx = int(modelo_clase.predict(entrada)[0])
rango_pred = ORDEN_RANGOS[clase_idx]

probas = modelo_clase.predict_proba(entrada)[0]

# ============================================================
# RESULTADO PRINCIPAL
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Nota estimada")
    st.markdown(
        f"<h1 style='text-align:center; color:{COLOR_RANGOS[rango_pred]};'>"
        f"{nota_pred:.1f} / 100</h1>",
        unsafe_allow_html=True,
    )
    st.progress(nota_pred / 100)

with col2:
    st.subheader("🎯 Rango cualitativo")
    st.markdown(
        f"<h1 style='text-align:center; color:{COLOR_RANGOS[rango_pred]};'>"
        f"{rango_pred}</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Categoría predicha por el modelo de clasificación. "
        "Puede no coincidir exactamente con el rango de la nota numérica "
        "porque son dos modelos distintos."
    )

st.markdown("---")

# ============================================================
# DISTRIBUCIÓN DE PROBABILIDADES POR RANGO
# ============================================================
st.subheader("Probabilidad de caer en cada rango")
df_probas = pd.DataFrame({
    "Rango": ORDEN_RANGOS,
    "Probabilidad": probas,
})
st.bar_chart(df_probas.set_index("Rango"))

# ============================================================
# COMPARATIVA CON ESTUDIANTES DE ALTO RENDIMIENTO
# ============================================================
st.markdown("---")
st.subheader("📈 ¿Cómo te comparas con los estudiantes que sacan ≥80?")
st.caption(
    "El modelo guarda los promedios de los estudiantes que sacaron 80 o más. "
    "Aquí ves tus valores frente a los suyos."
)

comparativa = pd.DataFrame({
    "Tú": [
        age, study_hours, sleep_hours, online_classes_hours, social_media_hours,
        screen_time_hours, gaming_hours, exercise_minutes, mental_health_score,
        part_time_job_num, self_study_hours,
    ],
    "Top estudiantes (≥80)": [
        promedios_top['age'], promedios_top['study_hours'],
        promedios_top['sleep_hours'], promedios_top['online_classes_hours'],
        promedios_top['social_media_hours'], promedios_top['screen_time_hours'],
        promedios_top['gaming_hours'], promedios_top['exercise_minutes'],
        promedios_top['mental_health_score'], promedios_top['part_time_job'],
        promedios_top['self_study_hours'],
    ],
}, index=[
    "Edad", "Horas estudio", "Horas sueño", "Clases online", "Redes sociales",
    "Pantalla total", "Videojuegos", "Ejercicio (min)", "Salud mental",
    "Trabajo parcial", "Estudio autónomo",
])

st.bar_chart(comparativa)

# ============================================================
# INFORMACIÓN SOBRE EL MODELO
# ============================================================
with st.expander("ℹ️ Sobre el modelo"):
    st.markdown("""
    - **Modelo de regresión**: Random Forest Regressor (30 árboles, max_depth=15)
      → predice la nota numérica entre 0 y 100.
    - **Modelo de clasificación**: Random Forest Classifier (30 árboles, max_depth=15)
      → predice el rango cualitativo entre 5 categorías.
    - **Preprocesado**:
        - Variables numéricas estandarizadas (`StandardScaler`).
        - Variables categóricas codificadas con One-Hot (`OneHotEncoder`).
    - Entrenado sobre el dataset `estudiantes_data_final_proyecto.csv`
      excluyendo casos atípicos (nota <20 con study_hours ≥2).

    ⚠️ **Esta app es una herramienta exploratoria, no un sistema de evaluación real.**
    La predicción es una estimación estadística basada en patrones del dataset,
    no una sentencia sobre lo que sacarás en tu examen.
    """)
