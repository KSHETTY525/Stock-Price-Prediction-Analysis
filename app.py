import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

plt.style.use('fivethirtyeight')

# Load the model
model = load_model('stock_dl_model.h5')

# Streamlit app
st.title('Stock Market Predictor')

# Input stock symbol
stock = st.text_input('Enter Stock Symbol', 'POWERGRID.NS')

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2024, 11, 1)

# Fetch data
df = yf.download(stock, start=start, end=end)

# Display data
st.subheader('Raw Stock Data')
st.write(df)

# Moving Averages and line plots
st.subheader(f'{stock} Closing Price Over Time')
fig_close = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{stock} Closing Price Over Time')
plt.legend()
st.pyplot(fig_close)

st.subheader(f'{stock} Opening Price Over Time')
fig_open = plt.figure(figsize=(12,6))
plt.plot(df['Open'], label='Opening Price', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{stock} Opening Price Over Time')
plt.legend()
st.pyplot(fig_open)

st.subheader(f'{stock} High Price Over Time')
fig_high = plt.figure(figsize=(12,6))
plt.plot(df['High'], label='High Price', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{stock} High Price Over Time')
plt.legend()
st.pyplot(fig_high)

st.subheader(f'{stock} Volume Over Time')
fig_volume = plt.figure(figsize=(12,6))
plt.plot(df['Volume'], label='Volume', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title(f'{stock} Volume Over Time')
plt.legend()
st.pyplot(fig_volume)

# MA and Predictions
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

st.subheader('MA100 vs MA200')
fig_ma = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, label='MA100')
plt.plot(ma200, label='MA200')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{stock} MA100 vs MA200')
plt.legend()
st.pyplot(fig_ma)

# Prediction Section
st.subheader('Stock Price Prediction')

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.80):])

scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test_full = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_full)

x_test = []
y_test = []

for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)

scale_factor = 1/scaler.scale_[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

fig_pred = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Original vs Predicted Price')
plt.legend()
st.pyplot(fig_pred)