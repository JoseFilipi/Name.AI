import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer


# Carregar os dados
data = pd.read_csv('name.csv')  # Substitua 'dataset.csv' pelo caminho do seu arquivo CSV

# Limpar os dados
data.dropna(subset=['Name'], inplace=True)  # Remover entradas onde 'Name' é NaN
data['Name'] = data['Name'].astype(str)  # Garantir que todos os nomes sejam strings

# Pré-processamento dos dados
names = data['Name'].values
genders = data['Gender'].values

# Tokenização dos nomes
tokenizer = Tokenizer(char_level=True)  # Tokenizar a nível de caracteres
tokenizer.fit_on_texts(names)
sequences = tokenizer.texts_to_sequences(names)
maxlen = max(len(seq) for seq in sequences)  # Encontrar o comprimento máximo dos nomes
X = pad_sequences(sequences, maxlen=maxlen)

# Converter os rótulos para categórico
y = to_categorical(genders)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir o modelo LSTM
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=maxlen))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Salvar o modelo
model.save('gender_prediction_model.h5')
