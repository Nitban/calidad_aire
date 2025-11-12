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

# Leer las variables de entorno
cred_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
database_url = os.getenv("DATABASE_URL")

if not cred_json:
    raise ValueError("❌ No se encontró la variable GOOGLE_APPLICATION_CREDENTIALS_JSON")

if not database_url:
    raise ValueError("❌ No se encontró la variable DATABASE_URL")

# Convertir el JSON del service account
cred_dict = json.loads(cred_json)
cred = credentials.Certificate(cred_dict)

# Inicializar Firebase
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
            "mensaje": "Predicción guardada en Firebase",
            "prediccion": prediccion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------
# 🚀 ARRANQUE DE LA APP (Render)
# ------------------------------
# Render usa gunicorn, así que no necesitamos app.run()
# Solo aseguramos que app exista para gunicorn
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
