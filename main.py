import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Flatten, LeakyReLU, LSTM, Input, Conv1D
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import Minute

file_path = r"C:\Users\GaelT\PycharmProjects\TRADING\Data\BTC-Hourly.csv"

data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']])

close_prices = data['close']

latent_dim = 100
sequence_length = 50
#features = 6
epochs = 165
batch_size = 128
minutes_par_jour = 60


def build_generator():
    model = Sequential()
    model.add(GRU(256, input_shape=(None, latent_dim), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(1024))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1, activation='tanh'))
    return model


def build_discriminator():
    model = Sequential()
    model.add(Conv1D(128, kernel_size=3, strides=2, input_shape=(sequence_length, features), padding="same"))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv1D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv1D(32, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(220))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(220))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train_gan(generator, discriminator, gan, data, epochs, batch_size):
    for epoch in range(epochs):
        # ----- Entraînement du discriminateur -----
        # Sélection aléatoire d'observations réelles
        real_data = np.random.choice(data, batch_size)
        real_labels = np.ones((batch_size, 1))

        # Générer des données fausses
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        fake_data = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))

        # Entraîner le discriminateur
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ----- Entraînement du générateur -----
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Enregistrement des progrès
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}')

# PARTIE LSTM
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



##PARTIE GAN
#
# generator = build_generator()
# discriminator = build_discriminator()
#
# discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.00016), metrics=['accuracy'])
# discriminator.trainable = False
# gan_input = Input(shape=(latent_dim,))
# fake_data = generator(gan_input)
# gan_output = discriminator(fake_data)
# gan = Model(gan_input, gan_output)
#
#
# train_gan(generator, discriminator, gan, data, epochs, batch_size)
#
# predictions = data['close']*1.02
# plt.figure(figsize=(14, 7))
# plt.plot(data['date'], data['close'], label='Valeurs Réelles', color='blue')
# plt.plot(data['date'], predictions, label='Prédictions du Modèle', color='red', linestyle='--')
#
## Personnalisation du graphique
# plt.title('Comparaison des Valeurs Réelles et des Prédictions du Modèle')
# plt.xlabel('Date')
# plt.ylabel('Prix de Clôture')
# plt.legend()
# plt.show()
#