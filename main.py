from flask import Flask, request, jsonify
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load("modelo_calidad_aire.pkl")

# --- Conectar con Firebase ---
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://calidad-aire-6685f-default-rtdb.firebaseio.com/"
})

@app.route('/')
def home():
    return "API de Calidad del Aire funcionando ✅"

# Endpoint para clasificar lecturas enviadas
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    # Realizar predicción
    pred = modelo.predict(df)[0]

    # Guardar predicción en Firebase (opcional)
    ref = db.reference("/predicciones")
    ref.push({
        "datos": data,
        "resultado": pred
    })

    return jsonify({
        "prediccion": pred,
        "mensaje": "Predicción guardada en Firebase"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
