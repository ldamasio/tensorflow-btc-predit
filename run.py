import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Função para carregar os dados
def load_data(file_path, sequence_length=10, split=0.8):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])  # Convert the date column to datetime

    # Assuming you want to use the 'Close' column as your price data
    data['Price'] = pd.to_numeric(data['Close'])  # Convert the 'Close' column to numeric and rename it to 'Price'

    # Normalization of data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data[['Price']])

    sequence_data = []
    for i in range(len(data_normalized) - sequence_length):
        sequence_data.append(data_normalized[i:i+sequence_length+1])

    sequence_data = np.array(sequence_data)
    row = round(split * sequence_data.shape[0])
    train = sequence_data[:int(row), :]

    # Shuffle the training data
    np.random.shuffle(train)

    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_test = sequence_data[int(row):, :-1]
    y_test = sequence_data[int(row):, -1]

    return x_train, y_train, x_test, y_test, scaler

# Carregar os dados históricos do Bitcoin
file_path = "BTC_price.csv"
sequence_length = 10
x_train, y_train, x_test, y_test, scaler = load_data(file_path, sequence_length)

# Construir o modelo da rede neural
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    tf.keras.layers.LSTM(units=50),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Testar o modelo
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotar os resultados
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Preço Real do BTC')
plt.plot(predictions, label='Previsões do Modelo')
plt.xlabel('Tempo')
plt.ylabel('Preço do BTC')
plt.legend()
plt.show()
