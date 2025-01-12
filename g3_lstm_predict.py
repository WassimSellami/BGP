import pandas as pd
from constants import Constants
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import os

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
INPUT_FILE = os.path.join(TEST_DATA_DIR, f'test_data_{Constants.TIME_WINDOW}.csv')
FEATURE_COLUMNS = [Constants.FEATURE_NB_A, Constants.FEATURE_NB_W, Constants.FEATURE_NB_A_W]

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_sequences(data, sequence_length=Constants.SEQUENCE_LENGTH):
    """
    Create sequences for LSTM prediction.
    """
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
    return np.array(X)

model = load_model(f'models/g3_lstm_{Constants.TIME_WINDOW}.h5', compile = False)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
with open(f'scalers/g3_scaler_{Constants.TIME_WINDOW}.pkl', 'rb') as f:
    scaler = pickle.load(f)

test_df = pd.read_csv(INPUT_FILE)

scaled_data = scaler.transform(test_df[FEATURE_COLUMNS])

X_test = create_sequences(scaled_data)
y_test = scaled_data[Constants.SEQUENCE_LENGTH:]

predictions = model.predict(X_test)

predictions_actual = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
mae = mean_absolute_error(y_test_actual, predictions_actual)
mape = mean_absolute_percentage_error(y_test_actual, predictions_actual)
r2 = r2_score(y_test_actual, predictions_actual)

print("\nModel Performance Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R^2 Score: {r2:.2f}")

plt.figure(figsize=(15, 8))

NUM_PLOTS = Constants.PLOT_WINDOW

time_steps = range(NUM_PLOTS)
actuals = y_test_actual[:NUM_PLOTS, 0]
preds = predictions_actual[:NUM_PLOTS, 0]

plt.plot(time_steps, actuals, label='Actual', color='#1f77b4', linewidth=2.5)
plt.plot(time_steps, preds, label='Predicted', color='#ff7f0e', linestyle='--', linewidth=2.5)

plt.xlabel('Time Steps', fontsize=14, fontweight='bold')
plt.ylabel('Values (Actual and Predicted)', fontsize=14, fontweight='bold')
plt.title(f'Actual vs Predicted Values (First {NUM_PLOTS} Time Steps)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

os.makedirs('prediction_results', exist_ok=True)
plt.savefig(f'prediction_results/g3_test_predictions_{Constants.TIME_WINDOW}.png', dpi=300)
plt.close()

