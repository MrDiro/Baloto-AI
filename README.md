# 🎰 Baloto AI - Lottery Prediction System

`Baloto AI` es un sistema de predicción de números de lotería basado en redes neuronales. Usa modelos LSTM con múltiples salidas para intentar predecir combinaciones ganadoras de un archivo histórico de resultados.

⚠️ Este proyecto **no garantiza resultados reales ni premios**, pero es una excelente herramienta de aprendizaje sobre redes neuronales, modelos multi-salida y entrenamiento de modelos con Keras/TensorFlow.

---

## 🚀 Getting Started

### 📁 Requisitos

* Python 3.8+
* TensorFlow 2.x
* NumPy
* [art](https://pypi.org/project/art/) (para generar banners ASCII)

### 📦 Instalación

```bash
pip install -r requirements.txt
```

### 📄 Formato del archivo `data.txt`

Debes tener un archivo llamado `data.txt` en el mismo directorio, con este formato:

```
5,12,18,23,35,4
8,10,14,21,39,12
...
```

* Las primeras 5 columnas son los números principales.
* La sexta columna es el número **bonus**.

---

## ⚙️ Ejecución

```bash
python baloto_ai.py
```

El script:

1. Limpia la consola.
2. Muestra una pantalla de bienvenida.
3. Carga y valida los datos.
4. Construye y entrena un modelo LSTM multi-output.
5. Muestra una barra de progreso animada por época.
6. Genera una predicción de una nueva combinación de números.

---

## 📌 Disclaimer

Este proyecto es **puramente educativo** y no tiene ninguna capacidad mágica para adivinar resultados de lotería reales.

* No está afiliado a ningún sistema de lotería oficial.
* No garantiza ningún tipo de premio.
* Usar esto para apuestas reales es bajo tu **propio riesgo** (y probablemente mala idea).

---

## 🤖 Créditos

Desarrollado por Dairo Carrasquilla.
Inspirado en la idea absurda pero divertida de predecir el azar con redes neuronales.
