import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db

# ------------------------------
# 🔧 CONFIGURACIÓN INICIAL
# ------------------------------

# Crear la app Flask
app = Flask(__name__)

# ------------------------------
# 🔐 CREDENCIALES FIREBASE (seguras para Render)
# ------------------------------

project_id = os.getenv("FIREBASE_PROJECT_ID")
client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
private_key = os.getenv("FIREBASE_PRIVATE_KEY")
database_url = os.getenv("DATABASE_URL")

if not all([project_id, client_email, private_key, database_url]):
    raise ValueError("❌ Faltan variables de entorno de Firebase")

# Render elimina saltos de línea, los restauramos
private_key = private_key.replace("\\n", "\n")

cred_dict = {
    "type": "service_account",
    "project_id": project_id,
    "private_key_id": "manual-config",
    "private_key": private_key,
    "client_email": client_email,
    "client_id": "manual",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email.replace('@', '%40')}"
}

cred = credentials.Certificate(cred_dict)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {"databaseURL": database_url})

# ------------------------------
# 🤖 CARGAR EL MODELO
# ------------------------------
try:
    model = joblib.load("modelo_calidad_aire.pkl")
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print("⚠️ Error al cargar el modelo:", e)
    model = None

# ------------------------------
# 🏠 RUTA PRINCIPAL
# ------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "API de Calidad del Aire activa 🌎",
        "endpoints": {
            "/predict": "POST → predicción con datos de sensores",
            "/sync-firebase": "GET → clasifica los últimos datos de Firebase"
        }
    })

# ------------------------------
# 🔍 PREDICCIÓN MANUAL (POST)
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model:
            return jsonify({"error": "Modelo no cargado"}), 500

        data = request.get_json()
        required = ["gas", "humedad", "luz", "polvo", "temperatura"]

        if not all(k in data for k in required):
            return jsonify({"error": f"Faltan campos. Requeridos: {required}"}), 400

        features = np.array([[
            data["gas"],
            data["humedad"],
            data["luz"],
            data["polvo"],
            data["temperatura"]
        ]])

        prediction = model.predict(features)[0]

        return jsonify({
            "input": data,
            "prediccion": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------
# ☁️ SINCRONIZAR CON FIREBASE
# ------------------------------
@app.route("/sync-firebase", methods=["GET"])
def sync_firebase():
    try:
        if not model:
            return jsonify({"error": "Modelo no cargado"}), 500

        ref = db.reference("/lecturas")
        data = ref.get()

        if not data:
            return jsonify({"mensaje": "No hay lecturas disponibles en Firebase"})

        last_key = list(data.keys())[-1]
        lectura = data[last_key]

        # Validar campos esperados
        required = ["gas", "humedad", "luz", "polvo", "temperatura"]
        if not all(k in lectura for k in required):
            return jsonify({"error": "Lectura incompleta"}), 400

        features = np.array([[
            lectura["gas"],
            lectura["humedad"],
            lectura["luz"],
            lectura["polvo"],
            lectura["temperatura"]
        ]])

        prediccion = model.predict(features)[0]

        # Guardar resultado en /predicciones
        db.reference("/predicciones").push({
            "lectura_id": last_key,
            "prediccion": prediccion
        })

        return jsonify({
            "mensaje": "✅ Predicción guardada en Firebase",
            "lectura_id": last_key,
            "prediccion": prediccion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------
# 🚀 ARRANQUE DE LA APP (Render)
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)