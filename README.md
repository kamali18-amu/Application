# CRACKING-THE-MARKET-CODE-WITHAI-DRIVEN-STOCK-PRICE-PREDICTION-USING-TIME-SERIES-ANALYSIS


# 📈 Stock Price Prediction Using LSTM 🚀

Welcome to the **Stock Price Prediction** project! This repository implements a powerful Long Short-Term Memory (LSTM) neural network to forecast stock prices using time series analysis. Dive into the world of AI-driven financial forecasting with us! 🌟

## 🎯 Project Overview

This project, part of *Cracking the Market Code with AI-Driven Stock Price Prediction*, aims to predict stock prices by leveraging historical data. Using Python and TensorFlow, we train an LSTM model to capture temporal patterns and forecast closing prices with high accuracy. 📊

## 📂 Table of Contents

- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Code Structure](#-code-structure)
- [Model Details](#-model-details)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## 📊 Dataset

The dataset is sourced from Kaggle and includes historical stock data in CSV format. Key columns include:

- 📅 **Date**: The trading date.
- 💰 **Open**: Opening price.
- 📈 **High**: Highest price of the day.
- 📉 **Low**: Lowest price of the day.
- 🎯 **Close**: Closing price (target for prediction).
- 📦 **Volume**: Trading volume.

**Note**: Download the dataset from [Kaggle](https://www.kaggle.com/datasets/your-dataset-link) and save it as `stock_data.csv` in the project directory. 📥

## 🔧 Installation

Get started in a few simple steps! Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
```

### 📋 Requirements

Install the following libraries listed in `requirements.txt`:

```
pandas
numpy
scikit-learn
tensorflow
matplotlib
```

Run:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

## 🚀 Usage

1. Place `stock_data.csv` in the project root.
2. Run the main script to train and predict:

   ```bash
   python lstm_stock_prediction.py
   ```

3. The script will:
   - 🧹 Preprocess the data.
   - 🧠 Train the LSTM model.
   - 🔮 Generate predictions.
   - 📊 Save plots as `stock_predictions.png`.

## 🗂 Code Structure

- `lstm_stock_prediction.py`: Main script for model training and visualization. 🧑‍💻
- `stock_data.csv`: Input dataset (to be added). 📄
- `stock_predictions.png`: Output plot of predictions. 📉
- `requirements.txt`: Dependency list. 📋
- `README.md`: You're reading it! 📖

## 🧠 Model Details

The LSTM model is designed for robust time series forecasting:

- **Preprocessing**:
  - 🧼 Handle missing values with forward fill and interpolation.
  - 📏 Scale data using `MinMaxScaler`.
  - 📅 Create sequences of 10 time steps.

- **Architecture**:
  - 🏗 Two LSTM layers (100 and 50 units) with 20% dropout.
  - 🔗 Dense layer for multi-feature output.
  - ⚙️ Adam optimizer with MSE loss.

- **Training**:
  - 📊 80% training, 20% testing split.
  - ⏳ 50 epochs with 10% validation and early stopping.

- **Evaluation**:
  - 📈 MAE and RMSE metrics.
  - 🖼 Visualizations of actual vs. predicted prices.

## 📉 Results

The model predicts stock prices with visualizations saved in `stock_predictions.png`. Key insights:

- ✅ Excels in stable market conditions.
- ⚠️ May struggle with high volatility.
- 📏 MAE and RMSE provide accuracy metrics (check console output).

![Stock Price Predictions](stock_predictions.png)

## 🤝 Contributing

We love contributions! To join the project:

1. 🍴 Fork the repository.
2. 🌱 Create a branch (`git checkout -b feature/your-feature`).
3. 💾 Commit changes (`git commit -m 'Add your feature'`).
4. 🚀 Push to the branch (`git push origin feature/your-feature`).
5. 📬 Open a pull request.

Follow PEP 8 and add clear comments. 🙌

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. 📝

---

**Author**: Kamali S.D  
**Github Profile Link**:https://github.com/kamali18-amu/CRACKING-THE-MARKET-CODE-WITHAI-DRIVEN-STOCK-PRICE-PREDICTION
**Institution**: Sri Ramanujar Engineering College  
**Department**: Computer Science and Engineering  


Happy forecasting! 🚀📈

