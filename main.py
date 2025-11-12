# ======================================================
# main.py — Flask + Firebase + Modelo ML (Render)
# ======================================================

from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np
import os
import json

# ======================================================
# 🔹 CONFIGURACIÓN FLASK
# ======================================================
app = Flask(__name__)

# ======================================================
# 🔹 CONFIGURACIÓN FIREBASE
# ======================================================
# En Render no puedes subir archivos JSON, por eso lo cargamos desde variable de entorno
firebase_creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
database_url = os.getenv("DATABASE_URL")

if not firebase_creds_json or not database_url:
    raise Exception("❌ Faltan las variables de entorno de Firebase")

# Convertir JSON string a dict temporal
cred_dict = json.loads(firebase_creds_json)
cred = credentials.Certificate(cred_dict)

firebase_admin.initialize_app(cred, {
    'databaseURL': database_url
})

# ======================================================
# 🔹 CARGAR MODELO
# ======================================================
MODEL_PATH = "modelo_calidad_aire.pkl"

try:
    modelo = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print("❌ Error al cargar el modelo:", e)
    modelo = None

# ======================================================
# 🔹 FUNCIÓN PARA CLASIFICAR UNA LECTURA
# ======================================================
def predecir_calidad(lectura):
    """
    Recibe un diccionario con los datos de sensores
    y devuelve la predicción de calidad del aire.
    """
    try:
        entrada = np.array([[ 
            lectura.get("gas", 0),
            lectura.get("humedad", 0),
            lectura.get("luz", 0),
            lectura.get("polvo", 0),
            lectura.get("temperatura", 0)
        ]])
        prediccion = modelo.predict(entrada)[0]
        return prediccion
    except Exception as e:
        print("❌ Error al predecir:", e)
        return "Error"

# ======================================================
# 🔹 ENDPOINT PRINCIPAL — para probar el servidor
# ======================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "Servidor Flask + Firebase + ML funcionando correctamente 🚀"
    })

# ======================================================
# 🔹 ENDPOINT PARA HACER PREDICCIONES MANUALES
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Permite enviar una lectura manual vía POST (JSON) para clasificarla.
    Ejemplo JSON:
    {
        "gas": 800,
        "humedad": 45,
        "luz": 2,
        "polvo": 300,
        "temperatura": 28
    }
    """
    if modelo is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    data = request.get_json()
    resultado = predecir_calidad(data)
    return jsonify({"prediccion": resultado})

# ======================================================
# 🔹 PROCESO AUTOMÁTICO — Leer de Firebase y subir predicciones
# ======================================================
@app.route("/procesar", methods=["GET"])
def procesar_lecturas():

