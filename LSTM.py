import yfinance as yf

import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
today_date = datetime.datetime.now().date()
start_date = "2014-07-10"
end_date = "2024-03-29"

jour = 40
crypto = yf.download("BTC-USD", start=start_date, end=end_date)

crypto['SMA_7'] = crypto['Adj Close'].rolling(window=7).mean()
crypto['SMA_30'] = crypto['Adj Close'].rolling(window=30).mean()


hyperparameters_space = {
    'lstm_units': [8, 16, 32],
    'activation': ['tanh', 'relu'],
    'optimizer': ['adam', 'sgd'],
    'epochs': [50, 100],
    'batch_size': [32, 64]
}


# Calcul du RSI
def calculate_RSI(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

crypto['RSI'] = calculate_RSI(crypto['Adj Close'])

# Calcul du MACD
exp1 = crypto['Adj Close'].ewm(span=12, adjust=False).mean()
exp2 = crypto['Adj Close'].ewm(span=26, adjust=False).mean()
crypto['MACD'] = exp1 - exp2


crypto['Signal_Line'] = crypto['MACD'].ewm(span=9, adjust=False).mean()


crypto = crypto.dropna()


azn_adj = crypto[['Adj Close']]

# Convert DataFrame to numpy array
azn_adj_arr = azn_adj.values
training_data_len = int(0.9*len(azn_adj))
train = azn_adj_arr[0:training_data_len, :]
history_size = 7
scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train)

X_train = []
y_train = []
# Creating a data structure with 60 time-steps and 1 output
for i in range(jour, len(train_scaled)):
    X_train.append(train_scaled[i-jour:i, 0])
    y_train.append(train_scaled[i:i+1, 0])

# Convert X_train and y_train to numpy arrays for training LSTM model
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the data as LSTM expects 3-D data (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape

test = azn_adj_arr[training_data_len: , :]

# Build model - LSTM

def create_and_train_model(X_train, y_train, X_val, y_val, config):
    model = Sequential()
    model.add(LSTM(config['lstm_units'], activation=config['activation'], input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    if config['optimizer'] == 'adam':
        optimizer = Adam()
    else:
        optimizer = SGD()
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], verbose=0)
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    return rmse

def evaluate_configurations(X, y, hyperparameters_space):
    results = []
    for lstm_units in hyperparameters_space['lstm_units']:
        for activation in hyperparameters_space['activation']:
            for optimizer in hyperparameters_space['optimizer']:
                for epochs in hyperparameters_space['epochs']:
                    for batch_size in hyperparameters_space['batch_size']:
                        config = {'lstm_units': lstm_units, 'activation': activation,
                                  'optimizer': optimizer, 'epochs': epochs, 'batch_size': batch_size}
                        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                        rmse = create_and_train_model(X_train, y_train, X_val, y_val, config)
                        results.append((config, rmse))
                        print(f"Tested config: {config}, RMSE: {rmse}")
    return sorted(results, key=lambda x: x[1])[0]


best_config, best_rmse = evaluate_configurations(X_train, y_train, hyperparameters_space)
print(f"Meilleure configuration : {best_config}, avec un RMSE de : {best_rmse}")

