# 🎓 Predictor de notas

Aplicación web interactiva que predice la nota de examen y el rango cualitativo
de un estudiante a partir de sus hábitos de estudio, descanso y bienestar.

Desarrollada como parte del proyecto de Análisis Exploratorio de Datos.

## 🧠 Modelos

- **Random Forest Regressor** (30 árboles, max_depth=15): predice la nota numérica (0–100).
- **Random Forest Classifier** (30 árboles, max_depth=15): predice el rango cualitativo
  entre 5 categorías (*Muy bajo*, *Bajo*, *Aprobado*, *Buena nota*, *Muy buena nota*).

Ambos modelos comparten el mismo preprocesado (`StandardScaler` para numéricas,
`OneHotEncoder` para categóricas) y se entrenaron con el dataset
`estudiantes_data_final_proyecto.csv`, excluyendo casos atípicos
(nota < 20 con study_hours ≥ 2).

## 📁 Estructura del repositorio

```
.
├── app.py                       # Aplicación Streamlit
├── modelos_rendimiento.pkl      # Modelos pre-entrenados (~3 MB)
├── requirements.txt             # Dependencias Python
└── README.md
```

## ▶️ Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`.

## ☁️ Despliegue online

Desplegada en Streamlit Community Cloud (gratuito).

## ⚠️ Aviso

Esta herramienta es **exploratoria**, no un sistema de evaluación real.
Las predicciones son estimaciones estadísticas basadas en patrones del dataset.
