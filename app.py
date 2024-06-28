import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Define the start and end dates
start = '2010-01-01'
end = '2019-12-31'

# Streamlit app
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetch stock data using yfinance
df = yf.download(user_input, start=start, end=end)

# Describe the data
st.subheader('Data From 2010 - 2019')
st.write(df.describe())

# Display the data
st.write(df)

# Visualizations
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], 'b')
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(df['Close'], 'b')
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df['Close'], 'b')
st.pyplot(fig)

# Splitting data into training and testing
train_size = int(len(df) * 0.70)
data_training = pd.DataFrame(df['Close'][:train_size])
data_testing = pd.DataFrame(df['Close'][train_size:])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Load model
model = load_model('my_model.keras')

# Testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions 
y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final graph
st.subheader('Predictions VS Originals')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
