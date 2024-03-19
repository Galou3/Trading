import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yfinance as yf


dataframe = pd.read_csv(r"C:\Users\GaelT\PycharmProjects\TRADING\Data\BTC-Daily.csv", index_col="date", parse_dates=True)

features = dataframe[['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

split_index = int(len(scaled_features) * 0.8) #POURCENTAGE DE TAILLE D ENTRAINEMENT

# Division en données d'entraînement et de test
train_values = scaled_features[:split_index]

test_values = scaled_features[split_index:]
print(test_values)
def create_window(dataset, start_index, end_index, history_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset)
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(dataset[indices])
        labels.append(dataset[i][0])
    return np.array(data), np.array(labels)

# Création des fenêtres de données
history_size = 7
train_features, train_labels = create_window(train_values, 0, None, history_size)
test_features, test_labels = create_window(test_values, 0, None, history_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=(history_size, 6)),  # 6 pour le nombre de features
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(128).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(128).repeat()

history = model.fit(
    train_dataset,
    epochs=165,
    steps_per_epoch=20,
    validation_data=test_dataset,
    validation_steps=3
)
def plot_history(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, len(loss), 0, max(max(loss), max(val_loss))])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


# Test sur les données d'entraînement (exemple)
predictions = model.predict(train_features)
temp_array = np.zeros((predictions.shape[0], 6))
temp_array[:, 0] = predictions.ravel()
real_predictions = scaler.inverse_transform(temp_array)[:, 0]

print(real_predictions)

plt.figure(figsize=(15,5))
plt.plot(real_predictions[:300], 'r', label='Predictions')
plt.plot(scaler.inverse_transform(train_values[history_size:300+history_size])[:,0], 'b', label='Real Data')
plt.legend()
plt.show()