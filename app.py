import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Stock Price Prediction with LSTM", layout="wide")
st.title("📈 Stock Price Prediction using LSTM")

# Load hardcoded CSV file
file_path = "cleaned_air_quality.csv"  # Change this to your desired file name
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# Preview dataset
st.subheader("📊 Preview of Dataset")
st.dataframe(df.head())

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Sequence length input
seq_length = st.slider("Select sequence length", 5, 50, 10)
train_size = int(len(scaled_data) * 0.8)

# Sequence creation
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Model builder
def build_model(seq_length, n_features):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(n_features)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Split and create sequences
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Model training
st.info("🧠 Training the LSTM model... This may take a few minutes.")
model = build_model(seq_length, df.shape[1])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
st.success("✅ Model training complete!")

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Plot results
st.subheader("📈 Predicted vs Actual Prices")
fig, axs = plt.subplots(nrows=(len(df.columns)+1)//2, ncols=2, figsize=(16, 10))
axs = axs.flatten()

for i, stock in enumerate(df.columns):
    axs[i].plot(y_test[:, i], label='Actual')
    axs[i].plot(predictions[:, i], label='Predicted')
    axs[i].set_title(f'{stock} Prediction')
    axs[i].legend()

plt.tight_layout()
st.pyplot(fig)
