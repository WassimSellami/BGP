import pandas as pd
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
import tensorflow as tf

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

WINDOW_LENGTH = Constants.SEQUENCE_LENGTH
BATCH_SIZE = Constants.BATCH_SIZE
BUFFER_SIZE = Constants.BUFFER_SIZE
TRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), 'train_data')
INPUT_FILE = os.path.join(TRAIN_DATA_DIR, 'rrc12-ma-1-g3.csv')

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

# Read data and take only the first MAX_TRAIN_ROWS
df = pd.read_csv(INPUT_FILE).head(Constants.MAX_TRAIN_ROWS)
print(f"Using {len(df)} rows for training and testing")

df_training, df_test = train_test_split(df, test_size=0.2, random_state=Constants.SEED)

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

X_train_df, y_train = create_time_features(df_training, target=Constants.FEATURE_NB_A_W)
X_test_df, y_test = create_time_features(df_test, target=Constants.FEATURE_NB_A_W)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

X_w = np.concatenate((X_train_scaled, X_test_scaled))
y_w = np.concatenate((y_train, y_test))
X_w, y_w = window_data(X_w, y_w, window=WINDOW_LENGTH)

X_train_w = X_w[:-len(X_test_scaled)]
y_train_w = y_w[:-len(X_test_scaled)]
X_test_w = X_w[-len(X_test_scaled):]
y_test_w = y_w[-len(X_test_scaled):]

train_data = tf.data.Dataset.from_tensor_slices((X_train_w, y_train_w))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((X_test_w, y_test_w))
val_data = val_data.batch(BATCH_SIZE).repeat()

model = Sequential([
    LSTM(128, input_shape=(WINDOW_LENGTH, X_train_scaled.shape[1]), dropout=0.0),
    Dense(128),
    Dense(128),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(train_data, 
                   epochs=Constants.EPOCHS,
                   batch_size=Constants.BATCH_SIZE,
                   validation_data=val_data,
                   validation_steps=10,
                   steps_per_epoch=Constants.EVALUATION_INTERVAL,
                   verbose=2)

yhat = model.predict(X_test_w)

print("\nModel Performance Metrics:")
print("RMSE : ", np.sqrt(mean_squared_error(y_test_w, yhat)), end=",     ")
print("MAE: ", mean_absolute_error(y_test_w, yhat), end=",     ")
print("MAPE : ", mean_absolute_percentage_error(y_test_w, yhat), end=",     ")
print("r2 : ", r2_score(y_test_w, yhat))

# Plot a subset of predictions
plot_start = len(y_test_w) // 2  # Start from middle of test set
plot_length = 200  # Plot 200 points

plt.figure(figsize=(12, 8))
plt.plot(y_test_w[plot_start:plot_start+plot_length], label='Original', linewidth=4.0, color='black')
plt.plot(yhat[plot_start:plot_start+plot_length], color='#FF1493', label='LSTM', linewidth=4.0)
plt.xticks(fontsize=15, fontweight="bold")
plt.yticks(fontsize=15, fontweight="bold")
plt.xlabel('Time steps', fontsize=23, fontweight="bold")
plt.ylabel('Number of Announcements & W', fontsize=27, fontweight="bold")
plt.yscale("log")
plt.legend(fontsize=28, loc='upper left')
plt.tight_layout()
plt.show()

os.makedirs('model', exist_ok=True)
os.makedirs('scaler', exist_ok=True)

model.save('model/lstm_model_1.h5')

with open('scaler/scaler_1.pkl', 'wb') as f:
    pickle.dump(scaler, f)
