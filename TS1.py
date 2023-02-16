#this will have code for timeseries forecast
import pandas as pd
import statsmodels.api as sm

# Load the time series data
data = pd.read_csv('data.csv', index_col='date', parse_dates=True)

# Create a SARIMA model
model = sm.tsa.SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# Fit the model to the data
results = model.fit()

# Make predictions with the model
forecast = results.predict(start='2023-01-01', end='2023-12-31')

# Print the forecast
print(forecast)
