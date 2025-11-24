import os
import json
import joblib
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db

# ------------------------------
# 🔧 CONFIGURACIÓN INICIAL
# ------------------------------

app = Flask(__name__)

# ------------------------------
# 🔐 LEER JSON COMPLETO DESDE VARIABLE
# ------------------------------

cred_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
database_url = os.getenv("DATABASE_URL")

if not cred_json:
    raise ValueError("❌ No se encontró GOOGLE_APPLICATION_CREDENTIALS_JSON en Render")

if not database_url:
    raise ValueError("❌ No se encontró DATABASE_URL en Render")

# Restaurar saltos de línea solo en private_key
cred_dict = json.loads(cred_json)
cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

# Inicializar Firebase Admin
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
            "/sync-firebase": "GET → clasifica y guarda la última lectura de Firebase"
        }
    })

# ------------------------------
# 🔍 PREDICCIÓN MANUAL
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model:
            return jsonify({"error": "Modelo no cargado"}), 500

        data = request.get_json()
        required = ["gas", "humedad", "luz", "polvo", "temperatura"]

        if not all(k in data for k in required):
            return jsonify({"error": f"Faltan campos: {required}"}), 400

        features = np.array([[data[k] for k in required]])
        prediction = model.predict(features)[0]

        # Crear resultado con timestamp
        result = {
            "input": data,
            "prediccion": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------
# ☁️ SINCRONIZAR ÚLTIMA LECTURA DE FIREBASE
# ------------------------------
@app.route("/sync-firebase", methods=["GET"])
def sync_firebase():
    try:
        if not model:
            return jsonify({"error": "Modelo no cargado"}), 500

        ref_lecturas = db.reference("/lecturas")
        ref_preds = db.reference("/predicciones")

        lecturas = ref_lecturas.get() or {}
        predicciones = ref_preds.get() or {}

        # Identificar lecturas ya procesadas
        procesadas = {p["lectura_id"] for p in predicciones.values() if "lectura_id" in p}

        # Fecha de hoy (UTC)
        hoy = datetime.utcnow().strftime("%Y-%m-%d")

        generadas = []

        for key, lectura in lecturas.items():

            # Debe tener fecha
            if "fecha" not in lectura:
                continue

            if not lectura["fecha"].startswith(hoy):
                continue  # No es de hoy

            if key in procesadas:
                continue  # Ya clasificada

            required = ["gas", "humedad", "luz", "polvo", "temperatura"]
            if not all(k in lectura for k in required):
                continue

            # Realizar predicción
            features = np.array([[lectura[k] for k in required]])
            prediccion = model.predict(features)[0]

            registro = {
                "lectura_id": key,
                "fecha": lectura["fecha"],
                "datos": lectura,
                "prediccion": prediccion,
                "timestamp": datetime.utcnow().isoformat()
            }

            ref_preds.push(registro)
            generadas.append(registro)

        return jsonify({
            "mensaje": f"Predicciones generadas: {len(generadas)}",
            "registros": generadas
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ------------------------------
# 🚀 SERVIDOR (Render)
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
