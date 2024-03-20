import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import seaborn as sns
import tensorflow as tf
import plotly.graph_objects as go
import math
import datetime
import keras
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
from datetime import date, timedelta
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

today_date = datetime.datetime.now().date()
start_date = "2014-07-10"
end_date = today_date.strftime("%Y-%m-%d")

jour = 30
crypto = yf.download("BTC-USD", start=start_date, end=end_date)

crypto['SMA_7'] = crypto['Adj Close'].rolling(window=7).mean()
crypto['SMA_30'] = crypto['Adj Close'].rolling(window=30).mean()

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
training_data_len = int(0.8*len(azn_adj))
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

# Build model - LSTM with 50 neurons and 4 hidden layers

model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(8, input_shape = (X_train.shape[1], 1), activation='tanh'))

model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs = 200, batch_size = 128)

inputs = azn_adj_arr[len(azn_adj_arr) - len(test) - 100:]
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
y_test = azn_adj_arr[training_data_len:, :]

for i in range(100,inputs.shape[0]):
    X_test.append(inputs[i-jour:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions - y_test)**2)

train = azn_adj[:training_data_len]
test = azn_adj[training_data_len:]
test['Predictions'] = predictions

last_60_days = azn_adj_arr[-jour:]
last_60_days_scaled = scaler.transform(last_60_days)

future_predictions = []
jour_futur = 200
for i in range(jour_futur):
    X_last_60 = np.reshape(last_60_days_scaled, (1, jour, 1))
    next_day_prediction_scaled = model.predict(X_last_60)
    future_predictions.append(scaler.inverse_transform(next_day_prediction_scaled))
    last_60_days_scaled = np.append(last_60_days_scaled, next_day_prediction_scaled)[1:].reshape(-1, 1)

last_date = pd.to_datetime(end_date)
future_dates = [last_date + pd.Timedelta(days=x+1) for x in range(jour_futur)]

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predictions'])


fig = go.Figure()

fig.add_trace(go.Scatter(x=train.index, y=train['Adj Close'], mode='lines', name='Training Data'))

fig.add_trace(go.Scatter(x=test.index, y=test['Adj Close'], mode='lines', name='Actual Values'))

fig.add_trace(go.Scatter(x=test.index, y=test['Predictions'], mode='lines', name='Predicted Values'))

# Ajout des prédictions futures
fig.add_trace(go.Scatter(x=future_predictions_df.index, y=future_predictions_df['Predictions'], mode='lines', name='Future Predictions'))


fig.update_layout(title='BTC Close Price Prediction with LSTM',
                  xaxis_title='Date',
                  yaxis_title='Price in USD',
                  template='plotly_dark')

fig.show()