import pandas as pd
from br import mean_absolute_percentage_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from constants import Constants
import pickle
import os

# Assuming df is your original dataframe with time-series data
df = pd.read_csv('train_data/rrc12-ma-5-g3.csv')  # Load your data

# Split the data into training and testing sets
df_training, df_test = train_test_split(df, test_size=0.2, random_state=Constants.SEED)

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
        X = X.drop([target], axis=1)  # Drop target column from X
        return X, y

    return X

# Create time features for training
X_train_df, y_train = create_time_features(df_training, target=Constants.FEATURE_NB_A_W)
X_test_df, y_test = create_time_features(df_test, target=Constants.FEATURE_NB_A_W)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

# Reshape the data for LSTM input: [samples, time steps, features]
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the LSTM model
model = Sequential([
    LSTM(128, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), dropout=0.0),
    Dense(128),
    Dense(128),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, 
                   epochs=Constants.EPOCHS,
                   batch_size=Constants.BATCH_SIZE,
                   validation_data=(X_test_scaled, y_test),
                   validation_steps=10,
                   steps_per_epoch=Constants.EVALUATION_INTERVAL,
                   verbose=2)

# Predict using the LSTM model
yhat = model.predict(X_test_scaled)

# Evaluate the model
print("RMSE : ", np.sqrt(mean_squared_error(y_test, yhat)), end=",     ")
print("MAE: ", mean_absolute_error(y_test, yhat), end=",     ")
print("MAPE : ", mean_absolute_percentage_error(y_test, yhat), end=",     ")
print("r2 : ", r2_score(y_test, yhat), end=",     ")

# Save predictions in a dictionary
predictionsDict = {
    'Tensorflow simple LSTM': yhat.flatten()
}

# Visualize the predictions vs. actual values
plt.plot(y_test.values[4708:4908], label='Original', linewidth=4.0, color='black')
plt.plot(yhat[4708:4908], color='#FF1493', label='LSTM', linewidth=4.0)
plt.xticks(fontsize=15, fontweight="bold")
plt.yticks(fontsize=15, fontweight="bold")
plt.xlabel('Time steps', fontsize=23, fontweight="bold")
plt.ylabel('Number of Announcements & W', fontsize=27, fontweight="bold")
plt.yscale("log")  # Added log scale
plt.legend(fontsize=28, loc='upper left')
plt.tight_layout()
plt.show()

# Save model and scaler

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)

# Save the model
model.save('model/lstm_model.h5')

# Save the scaler
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
