# 🍎 Valoración Nutricional Antropométrica Multi-Modelo con IA

Esta aplicación interactiva, desarrollada con **Streamlit**, provee una herramienta integral para la **valoración nutricional antropométrica** utilizando la potencia de la Inteligencia Artificial. Combina y compara las predicciones de **tres modelos de Machine Learning** avanzados, ofreciendo un análisis detallado y reportes profesionales.

## 🌟 Características Destacadas

* **Análisis Multi-Modelo**: Integra y compara predicciones de **XGBoost**, **Regresión Lineal** y **Random Forest** para una valoración más robusta.
* **Valoraciones Clave**: Predice y evalúa la **Valoración Talla-Edad** y la **Valoración IMC-Talla** de manera precisa.
* **Métricas de Rendimiento Avanzadas**: Muestra métricas clave como **Precisión, Exactitud, Recall, F1-Score, AUC-ROC, Coeficiente de Matthews (MCC)** y **Nivel de Confianza** para cada modelo, todas con un rendimiento superior al 80%.
* **Visualizaciones Interactivas**: Genera gráficos de barras de **probabilidades por categoría** (Talla-Edad e IMC-Talla) para cada modelo, facilitando la comprensión de las predicciones.
* **Reportes PDF Profesionales**: Crea **reportes PDF multi-modelo detallados**, incluyendo los datos del paciente, resultados de cada modelo, métricas, y un resumen con una recomendación automática del mejor modelo.
* **Integración de Gráficos en PDF**: Si están disponibles, los reportes PDF pueden incluir matrices de confusión, curvas Precision-Recall, curvas ROC y gráficos de calibración para un análisis más profundo.
* **Interfaz Intuitiva y Multilingüe**: Diseñada con Streamlit para una experiencia de usuario amigable y disponible en **español e inglés**.

## 🤖 Modelos de Machine Learning Implementados

Esta aplicación aprovecha las fortalezas de los siguientes algoritmos:

### 1. **XGBoost (Extreme Gradient Boosting)**
Un potente algoritmo de ensamble conocido por su alto rendimiento y eficiencia en el manejo de datos estructurados y valores faltantes. Ideal para tareas de clasificación complejas.

### 2. **Regresión Lineal (Linear Regression)**
Un modelo fundamental e interpretable que establece relaciones lineales entre las variables. Sirve como una base sólida para comparar el rendimiento con modelos más complejos.

### 3. **Random Forest (Bosques Aleatorios)**
Un algoritmo de ensamble basado en árboles de decisión, robusto contra el sobreajuste y excelente para la generalización, proporcionando predicciones fiables.

## 📂 Estructura de Carpetas (Para Reportes y Gráficos)

Para que los reportes PDF incluyan correctamente las imágenes y los modelos sean cargados, se espera la siguiente estructura en tu proyecto:

.
├── models/                     # Contiene modelo XGBoost y encoders
│   ├── modelo_valoracion_talla.joblib
│   ├── modelo_valoracion_imc.joblib
│   ├── le_sexo.joblib
│   ├── le_talla.joblib
│   └── le_imc.joblib
├── models1/                    # Contiene modelo Regresión Lineal y encoders
│   ├── modelo_valoracion_talla.joblib
│   ├── modelo_valoracion_imc.joblib
│   ├── le_sexo.joblib
│   ├── le_talla.joblib
│   └── le_imc.joblib
├── models2/                    # Contiene modelo Random Forest y encoders
│   ├── modelo_valoracion_talla.joblib
│   ├── modelo_valoracion_imc.joblib
│   ├── le_sexo.joblib
│   ├── le_talla.joblib
│   └── le_imc.joblib
├── confusion_matrices/         # Opcional: Para matrices de confusión en PDF
│   ├── confusion_matrix_MODELO_talla_edad.png
│   └── confusion_matrix_MODELO_imc_talla.png
└── Graficas/                   # Opcional: Para curvas de rendimiento en PDF
├── XGBoost/
│   ├── talla_edad_precision_recall_xgboost.png
│   ├── talla_edad_roc_xgboost.png
│   └── talla_edad_calibration_xgboost.png
├── Regresion_Lineal/
│   ├── imc_talla_precision_recall_linear.png
│   ├── imc_talla_roc_linear.png
│   └── imc_talla_calibration_linear.png
└── Random_Forest/
├── talla_edad_precision_recall_random_forest.png
├── talla_edad_roc_random_forest.png
└── talla_edad_calibration_random_forest.png

**Nota**: Reemplaza `MODELO` en los nombres de archivo de las matrices de confusión con el nombre real del modelo (ej. `xgboost`, `regresion_lineal`, `random_forest`). Los nombres de las gráficas de rendimiento también deben coincidir con los esperados por el script.

## 🚀 Instalación y Ejecución

Para poner en marcha esta aplicación en tu entorno local, sigue estos sencillos pasos:

1.  **Clona el repositorio:**

    ```bash
    git clone <URL_DE_TU_REPOSITORIO>
    cd <nombre_de_tu_repositorio>
    ```

2.  **Crea un entorno virtual (recomendado):**

    ```bash
    python -m venv venv
    # En Windows
    .\venv\Scripts\activate
    # En macOS/Linux
    source venv/bin/activate
    ```

3.  **Instala las dependencias:**
    Asegúrate de tener un archivo `requirements.txt` en la raíz de tu proyecto con todas las bibliotecas listadas.

    ```bash
    pip install -r requirements.txt
    ```

    Las dependencias clave incluyen: `streamlit`, `numpy`, `pandas`, `joblib`, `fpdf`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `scipy`.

4.  **Coloca los modelos y encoders:**
    Asegúrate de que tus archivos `.joblib` de modelos y codificadores estén ubicados en las carpetas `models/`, `models1/` y `models2/` como se especifica en la estructura anterior.

5.  **Crea las carpetas para las gráficas (opcional):**
    Si deseas que los reportes PDF incluyan las matrices de confusión y las curvas de rendimiento, crea las carpetas `confusion_matrices/` y `Graficas/` con sus respectivas subcarpetas y coloca tus archivos PNG generados previamente siguiendo la convención de nombres.

6.  **Ejecuta la aplicación Streamlit:**

    ```bash
    streamlit run app.py
    ```
7.  **Accede a la Aplicación Web:**
    También puedes disfrutar de la aplicación directamente en línea a través del siguiente enlace: [Valoración Nutricional Antropométrica Multi-Modelo](https://valoracionnutricionalantropometrica.streamlit.app/)

---
