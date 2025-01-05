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
INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_data_30.csv')

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
    df_1 = pd.DataFrame(df)
    
    X = df_1
    if target:
        y = df[target]
        X = X.drop([target], axis=1)
        return X, y
    return X

# Load the saved model and scaler
model = load_model('model/lstm_model_30.h5')
with open('scaler/scaler_30.pkl', 'rb') as f:
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

# Calculate metrics
print("\nModel Performance Metrics:")
print("RMSE : ", np.sqrt(mean_squared_error(y_test_w, predictions)), end=",     ")
print("MAE: ", mean_absolute_error(y_test_w, predictions), end=",     ")
print("MAPE : ", mean_absolute_percentage_error(y_test_w, predictions), end=",     ")
print("r2 : ", r2_score(y_test_w, predictions))

# Create comparison plot
plt.figure(figsize=(15, 8))

# Set the style parameters manually
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.facecolor'] = '#f0f0f0'

# Get first PLOT_WINDOW time steps
time_steps = range(Constants.PLOT_WINDOW)
actuals = y_test_w[:Constants.PLOT_WINDOW].flatten()
preds = predictions[:Constants.PLOT_WINDOW].flatten()

# Plot lines with enhanced styling
plt.plot(time_steps, actuals, color='#1f77b4', label='Actual', 
         linewidth=2.5, marker='o', markersize=8, markerfacecolor='white')
plt.plot(time_steps, preds, color='#ff7f0e', label='Predicted', 
         linewidth=2.5, marker='s', markersize=8, markerfacecolor='white', 
         linestyle='--')

# Add value labels with improved positioning
for i, (actual, pred) in enumerate(zip(actuals, preds)):
    plt.annotate(f'{actual:.0f}', (i, actual), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
    plt.annotate(f'{pred:.0f}', (i, pred), textcoords="offset points", 
                xytext=(0,-15), ha='center', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

# Customize axes and labels
plt.xlabel('Time Steps', fontsize=23, fontweight="bold", labelpad=15)
plt.ylabel('Number of Announcements & W', fontsize=27, fontweight="bold", labelpad=15)
plt.title('Actual vs Predicted Values', fontsize=20, fontweight="bold", pad=20)

# Customize ticks
plt.xticks(time_steps, [f'Step {i+1}' for i in range(len(time_steps))], 
          fontsize=12, fontweight="bold", rotation=45)
plt.yticks(fontsize=12, fontweight="bold")

# Enhance legend
plt.legend(fontsize=16, loc='upper left', frameon=True, 
          facecolor='white', edgecolor='gray')

# Add a light border around the plot
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with high quality
plt.savefig('prediction_results/test.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
