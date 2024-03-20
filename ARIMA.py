import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Télécharger les données historiques du Bitcoin
start_date = "2014-07-10"
end_date = datetime.datetime.now().date().strftime("%Y-%m-%d")  # Date d'aujourd'hui
crypto = yf.download("BTC-USD", start=start_date, end=end_date)

# Nous allons travailler avec le prix de clôture ajusté
crypto_close = crypto['Adj Close']

# Vérifier la stationnarité avec le test d'Augmented Dickey-Fuller
adf_test = adfuller(crypto_close)

# Afficher le résultat du test ADF
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")

# S'il est nécessaire de différencier, nous pouvons utiliser d=1 ou plus selon la nécessité

# Sélection des paramètres p, d, q à l'aide des plots ACF et PACF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
plot_acf(crypto_close, ax=ax1)
plot_pacf(crypto_close, ax=ax2)

# Ces plots aideront à choisir les paramètres p et q pour le modèle ARIMA

# Pour l'exemple, supposons que nous avons décidé d'utiliser p=1, d=1, q=1 après analyse
# Modèle ARIMA
model = sm.tsa.arima.ARIMA(crypto_close, order=(1, 1, 1))
model_fit = model.fit()
steps = 15
# Prédiction
prediction = model_fit.forecast(steps=steps)

# Obtenir les dates futures pour l'index de prédiction
# Assurez-vous que `crypto_close.index[-1]` est la dernière date dans vos données et que la fréquence est correcte ('D' pour quotidien)
future_dates = pd.date_range(start=crypto_close.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')


predicted_series = pd.Series(prediction, index=future_dates)


plt.figure(figsize=(14,7))
plt.plot(crypto_close, label='Historique')

plt.plot(predicted_series, color='red', label='Prédiction')

plt.title('Prédiction du prix du Bitcoin avec ARIMA')
plt.xlabel('Date')
plt.ylabel('Prix du Bitcoin (USD)')
plt.legend()
plt.show()

