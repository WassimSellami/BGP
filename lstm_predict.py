import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
from constants import Constants
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import os

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_data1.csv')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def window_data(X, Y, window=7):
    '''
    Creates sliding windows for sequential prediction
    '''
    x = []
    y = []
    for i in range(window-1, len(X)):
        x.append(X[i-window+1:i+1])
        y.append(Y[i])
    return np.array(x), np.array(y)

os.makedirs('prediction_results', exist_ok=True)

def create_time_features(df, target=None):
    """
    Creates time series features from datetime index
    """
    
    X = df
    if target:
        y = df[target]
        X = X.drop([target], axis=1)
        return X, y
    return X

# Load the saved model and scaler
model = load_model('model/lstm_model_1.h5')
with open('scaler/scaler_1.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load test data
test_df = pd.read_csv(INPUT_FILE)

# Prepare features
X_test_df, y_test = create_time_features(test_df, target=Constants.FEATURE_NB_A_W)

# Scale the features
X_test_scaled = scaler.transform(X_test_df)

# Create windowed data for prediction
X_test_w, y_test_w = window_data(X_test_scaled, y_test, window=Constants.SEQUENCE_LENGTH)

# Make predictions
predictions = model.predict(X_test_w)

# Create comparison plot for first 5 time steps
plt.figure(figsize=(15, 8))

# Get first 5 time steps
time_steps = range(Constants.PLOT_WINDOW)
actuals = y_test_w[:Constants.PLOT_WINDOW].flatten()  # Flatten the array
preds = predictions[:Constants.PLOT_WINDOW].flatten()  # Flatten the predictions

# Plot with bars side by side
x = np.arange(len(time_steps))
width = 0.35

# Create bar plots
plt.bar(x - width/2, actuals, width, label='Actual', color='black')
plt.bar(x + width/2, preds, width, label='Predicted', color='#FF1493')

# Add value labels on top of each bar
for i in range(len(time_steps)):
    plt.text(x[i] - width/2, actuals[i], f'{actuals[i]:.0f}', ha='center', va='bottom')
    plt.text(x[i] + width/2, preds[i], f'{preds[i]:.0f}', ha='center', va='bottom')

plt.xlabel('Time Steps', fontsize=23, fontweight="bold")
plt.ylabel('Number of Announcements & W', fontsize=27, fontweight="bold")
plt.title('Actual vs Predicted Values - First  5Time Steps', fontsize=20, fontweight="bold")
plt.xticks(x, [f'Step {i+1}' for i in range(len(time_steps))], fontsize=15, fontweight="bold")
plt.yticks(fontsize=15, fontweight="bold")
plt.legend(fontsize=20, loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('prediction_results/test.png')
plt.close()

print("RMSE : ", np.sqrt(mean_squared_error(y_test_w, predictions)), end=",     ")
print("MAE: ", mean_absolute_error(y_test_w, predictions), end=",     ")
print("MAPE : ", mean_absolute_percentage_error(y_test_w, predictions), end=",     ")
print("r2 : ", r2_score(y_test_w, predictions), end=",     ")
