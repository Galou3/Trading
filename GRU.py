import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

today_date = datetime.datetime.now().date()
start_date = "2014-07-10"
end_date = today_date.strftime("%Y-%m-%d")

jour = 30
dataraw = yf.download("BTC-USD", start=start_date, end=end_date)
dataset = pd.DataFrame(dataraw['Close'])
print('Count row of data: ', len(dataset))

dataset_norm = dataset.copy()
dataset[['Close']]
scaler = MinMaxScaler()
dataset_norm['Close'] = scaler.fit_transform(dataset[['Close']])
dataset_norm

totaldata = dataset.values
totaldatatrain = int(len(totaldata)*0.7)
totaldataval = int(len(totaldata)*0.1)
totaldatatest = int(len(totaldata)*0.2)

# Store data into each partition
training_set = dataset_norm[0:totaldatatrain]
val_set = dataset_norm[totaldatatrain:totaldatatrain+totaldataval]
test_set = dataset_norm[totaldatatrain+totaldataval:]

# Initiation value of lag
lag = 2

# Sliding windows function
def create_sliding_windows(data, len_data, lag):
    x = []
    y = []
    for i in range(lag, len_data):
        x.append(data[i-lag:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Formating data into array for create sliding windows
array_training_set = np.array(training_set)
array_val_set = np.array(val_set)
array_test_set = np.array(test_set)

# Create sliding windows into training data
x_train, y_train = create_sliding_windows(array_training_set, len(array_training_set), lag)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# Create sliding windows into validation data
x_val, y_val = create_sliding_windows(array_val_set, len(array_val_set), lag)
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
# Create sliding windows into test data
x_test, y_test = create_sliding_windows(array_test_set, len(array_test_set), lag)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Hyperparameters
learning_rate = 0.0001
hidden_unit = 64
batch_size = 256
epoch = 100

# Architecture Gated Recurrent Unit
regressorGRU = Sequential()

# First GRU layer with dropout
regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Second GRU layer with dropout
regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Third GRU layer with dropout
regressorGRU.add(GRU(units=hidden_unit, return_sequences=False, activation='tanh'))
regressorGRU.add(Dropout(0.2))

# Output layer
regressorGRU.add(Dense(units=1))

# Compiling the Gated Recurrent Unit
regressorGRU.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

# Fitting ke data training dan data validation
pred = regressorGRU.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epoch)

# Tabel value of training loss & validation loss
learningrate_parameter = learning_rate
train_loss=pred.history['loss'][-1]
validation_loss=pred.history['val_loss'][-1]
learningrate_parameter=pd.DataFrame(data=[[learningrate_parameter, train_loss, validation_loss]],
                                    columns=['Learning Rate', 'Training Loss', 'Validation Loss'])
learningrate_parameter.set_index('Learning Rate')

# Implementation model into data test
y_pred_test = regressorGRU.predict(x_test)

# Invert normalization min-max
y_pred_invert_norm = scaler.inverse_transform(y_pred_test)

datacompare = pd.DataFrame()
datatest=np.array(dataset['Close'][totaldatatrain+totaldataval+lag:])
datapred= y_pred_invert_norm

datacompare['Data Test'] = datatest
datacompare['Prediction Results'] = datapred

# Calculatre value of Root Mean Square Error
def rmse(datatest, datapred):
    return np.round(np.sqrt(np.mean((datapred - datatest) ** 2)), 4)

print('Result Root Mean Square Error Prediction Model :', rmse(datatest, datapred))

def mape(datatest, datapred):
    return np.round(np.mean(np.abs((datatest - datapred) / datatest) * 100), 4)

print('Result Mean Absolute Percentage Error Prediction Model : ', mape(datatest, datapred), '%')

# Create graph data test and prediction result with Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(x=pd.date_range(start=start_date, periods=len(training_set), freq='D'), y=training_set['Close'], mode='lines', name='Training Data'))

fig.add_trace(go.Scatter(x=pd.date_range(start=start_date, periods=len(training_set) + len(test_set), freq='D')[len(training_set):], y=datatest, mode='lines', name='Actual Values'))

fig.add_trace(go.Scatter(x=pd.date_range(start=start_date, periods=len(training_set) + len(test_set), freq='D')[len(training_set):], y=datapred.flatten(), mode='lines', name='Predicted Values'))

fig.update_layout(title='BTC Close Price Prediction with GRU',
                  xaxis_title='Date',
                  yaxis_title='Price in USD',
                  template='plotly_dark')

fig.show()
