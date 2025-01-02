import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Fetch SPY data from 1993-01-29 to today
data = yf.download("SPY", start="1993-01-29")

# Display the first few rows of the data to verify
print(data.head())

# Display the shape of the dataset
print(f"Data Shape: {data.shape}")

# Plot SPY Closing Price
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label="SPY Closing Price", color='blue')
plt.title("SPY Closing Price (1993 - Present)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.show()

# Display summary statistics
print(data.describe())

# Add Simple Moving Averages (SMA)
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# Add Exponential Moving Averages (EMA)
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

# Add Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


data['RSI'] = calculate_rsi(data)

# Display the first few rows with new features
print(data.head(15))

# Drop rows with missing values
data = data.dropna()

# Verify data shape after dropping NaN rows
print(f"Data Shape after dropping NaNs: {data.shape}")

# Plot Closing Price with SMA and EMA
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='SPY Closing Price', color='blue', alpha=0.5)
plt.plot(data['SMA_10'], label='SMA 10', color='green')
plt.plot(data['SMA_50'], label='SMA 50', color='red')
plt.plot(data['EMA_10'], label='EMA 10', linestyle='--', color='purple')
plt.title('SPY Closing Price with SMA and EMA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()

# Plot RSI
plt.figure(figsize=(14, 7))
plt.plot(data['RSI'], label='RSI', color='orange')
plt.axhline(70, color='red', linestyle='--', label='Overbought')
plt.axhline(30, color='green', linestyle='--', label='Oversold')
plt.title('RSI (Relative Strength Index)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.grid()
plt.show()

# Define features (technical indicators) and target (next day's closing price)
features = data[['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI']]  # These are the technical indicators
target = data['Close'].shift(-1)  # Predicting the next day's closing price

# Drop the last row where the target is NaN
features = features[:-1]
target = target[:-1]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Check the size of the training and test sets
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)

print(f"RMSE: {rmse}")

feature_importances = model.feature_importances_
# Ensure feature names are in the correct format
feature_names = list(features.columns)

# Plot Predicted vs Actual values
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, predictions, label='Predicted', color='orange')
plt.title('Actual vs Predicted SPY Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()


# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train.values.ravel())


# Get the best parameters from grid search
print(f"Best Parameters: {grid_search.best_params_}")

# Train the model with best parameters
best_model = grid_search.best_estimator_

# Evaluate the model with test set
predictions = best_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")


def predict_next_day(features, model):
    # Ensure the features are in the correct format (a 2D array with shape [1, n_features])
    prediction = model.predict(features)
    return prediction[0]


# Example: Predict the next day's closing price using the most recent technical indicators
recent_data = data.iloc[-1:][['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI']]
next_day_prediction = predict_next_day(recent_data, best_model)

print(f"Predicted Next Day's SPY Closing Price: ${next_day_prediction:.2f}")


# Create a function for making ongoing predictions (e.g., daily)
def rolling_prediction(data, model, window_size=30):
    predictions = []
    for i in range(window_size, len(data)):
        features = data.iloc[i - window_size:i][['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI']].iloc[-1:]
        prediction = predict_next_day(features, model)
        predictions.append(prediction)
    return predictions


# Make predictions for the next 30 days based on the most recent data
future_predictions = rolling_prediction(data, best_model, window_size=30)

# Plot the predicted future prices
plt.figure(figsize=(14, 7))
plt.plot(data.index[-30:], future_predictions, label="Predicted Future Prices", color='red')
plt.title("Predicted Future SPY Prices for the Next 30 Days")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.show()

# Save the trained model
joblib.dump(best_model, 'spy_price_predictor_model.pkl')

# Load the model
loaded_model = joblib.load('spy_price_predictor_model.pkl')

# Make predictions with the loaded model
predictions = loaded_model.predict(X_test)
print(f"Predictions with Loaded Model: {predictions[:5]}")



model = Sequential()

# Adding LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# Adding another LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Adding output layer
model.add(Dense(units=1))  # Predict the next day's price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model = Sequential()

# Adding GRU layer
model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# Adding another GRU layer
model.add(GRU(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Adding output layer
model.add(Dense(units=1))  # Predict the next day's price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

y_pred = model.predict(X_test)
# Calculate RMSE or MAE
