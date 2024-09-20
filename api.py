import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carregar o modelo salvo
LSTM_model = load_model("LSTM_model.keras")

# Carregar o CountVectorizer salvo
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)


def predict(name):
    # Vetorizar o nome usando o CountVectorizer carregado
    name_vectorized = cv.transform([name]).toarray()
    # Fazer a previsão com o modelo LSTM
    prediction = LSTM_model.predict(name_vectorized)
    # Interpretar a previsão
    if prediction >= 0.5:
        out = 'Masculino'
    else:
        out = 'Feminino'
    return out


@app.route('/predict', methods=['POST'])
def predict_gender():
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({"error": "Nome é obrigatório"}), 400

    gender = predict(name)
    return jsonify({"name": name, "gender": gender})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
