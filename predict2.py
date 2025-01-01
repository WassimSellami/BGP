import pandas as pd
from br import mean_absolute_percentage_error
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

# Create prediction_results directory if it doesn't exist
os.makedirs('prediction_results', exist_ok=True)

def create_time_features(df, target=None):
    """
    Creates time series features from datetime index
    """
    df_1 = pd.DataFrame(df, columns=[Constants.FEATURE_NB_A, Constants.FEATURE_NB_W, 
                                   Constants.FEATURE_NB_A_W, Constants.FEATURE_NB_A_MA, 
                                   Constants.FEATURE_NB_W_MA])
    
    X = df_1
    if target:
        y = df[target]
        X = X.drop([target], axis=1)
        return X, y
    return X

# Load the saved model and scaler
model = load_model('model/lstm_model.h5')
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load test data
test_df = pd.read_csv('test_data/test_data1.csv')

# Prepare features
X_test_df, y_test = create_time_features(test_df, target=Constants.FEATURE_NB_A_W)

# Scale the features
X_test_scaled = scaler.transform(X_test_df)

# Reshape for LSTM
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Make predictions
predictions = model.predict(X_test_scaled)

# Create comparison plot
plt.figure(figsize=(12, 8))
plt.plot(y_test.values, label='Original', linewidth=4.0, color='black')
plt.plot(predictions, color='#FF1493', label='LSTM', linewidth=4.0)
plt.xticks(fontsize=15, fontweight="bold")
plt.yticks(fontsize=15, fontweight="bold")
plt.xlabel('Time steps', fontsize=23, fontweight="bold")
plt.ylabel('Number of Announcements & W', fontsize=27, fontweight="bold")
plt.yscale("log")
plt.legend(fontsize=28, loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('prediction_results/test.png')
plt.close()


print("RMSE : ", np.sqrt(mean_squared_error(y_test, predictions)), end=",     ")
print("MAE: ", mean_absolute_error(y_test, predictions), end=",     ")
print("MAPE : ", mean_absolute_percentage_error(y_test, predictions), end=",     ")
print("r2 : ", r2_score(y_test, predictions), end=",     ")
