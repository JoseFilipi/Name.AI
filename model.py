import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import h5py
import pickle
import tempfile

# Carregar o modelo salvo
LSTM_model = load_model("LSTM_model.keras")

# Carregar o CountVectorizer salvo
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

# Criar um arquivo temporário para o modelo Keras
with tempfile.NamedTemporaryFile(delete=False) as tmp_model_file:
    tmp_model_path = tmp_model_file.name
    LSTM_model.save(tmp_model_path)

# Salvar ambos em um arquivo HDF5
with h5py.File('combined_model.h5', 'w') as f:
    # Salvar o modelo Keras como um blob
    with open(tmp_model_path, 'rb') as tmp_model_file:
        model_data = tmp_model_file.read()
        f.create_dataset('model', data=np.void(model_data))
    
    # Salvar o CountVectorizer como um dataset de string
    vectorizer_data = pickle.dumps(cv)
    f.create_dataset('count_vectorizer', data=np.void(vectorizer_data))

# Remover o arquivo temporário
import os
os.remove(tmp_model_path)
