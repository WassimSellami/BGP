import pybgpstream
import csv
import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
from constants import Constants
from bgp_features import BGPFeatures
import signal
import sys
import os
from collections import deque

# Create prediction_results directory if it doesn't exist
os.makedirs('prediction_results', exist_ok=True)

# Set up matplotlib style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#CCCCCC'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2

def signal_handler(sig, frame):
    print("\nSaving final plot...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'prediction_results/real_time_prediction_{timestamp}.png')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def calculate_moving_average(records, field, window_size):
    if not records:
        return 0
    values = [r[field] for r in records[-window_size:]]
    return round(sum(values) / len(values), 2)

# Load the LSTM model and scaler
model = load_model('model/lstm_model_1.h5')
with open('scaler/scaler_1.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize BGP stream
stream = pybgpstream.BGPStream(
    project="ris-live",
)

features = BGPFeatures()
last_save_time = time.time()
recent_records = []

# Initialize plot
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots(figsize=(12, 6))
actual_line, = ax.plot([], [], 'b-', label='Actual', linewidth=2)
pred_line, = ax.plot([], [], 'r--', label='Predicted', linewidth=2)
ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Announcements & W', fontsize=12, fontweight='bold')
ax.set_title('Real-time BGP Announcements vs Predictions', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.grid(True)

actual_values = []
predicted_values = []
time_steps = []

# Initialize sequence buffer for LSTM
sequence_buffer = deque(maxlen=Constants.SEQUENCE_LENGTH)

print(f"Collecting initial {Constants.SEQUENCE_LENGTH} data points...")
print("Press Ctrl+C to stop and save the plot...")

for elem in stream:
    if Constants.CHOSEN_COLLECTOR in elem.collector:
        features.classify_elem(elem.type)
        current_time = time.time()
        
        if current_time - last_save_time >= Constants.TIME_WINDOW:
            # Create current record
            current_record = {
                'nb_A': features.nb_A,
                'nb_W': features.nb_W,
                'nb_A_W': features.nb_A_W
            }
            
            recent_records.append(current_record)
            if len(recent_records) > Constants.MA_WINDOW:
                recent_records.pop(0)
            
            nb_A_ma = calculate_moving_average(recent_records, 'nb_A', Constants.MA_WINDOW)
            nb_W_ma = calculate_moving_average(recent_records, 'nb_W', Constants.MA_WINDOW)
            nb_A_W_ma = calculate_moving_average(recent_records, 'nb_A_W', Constants.MA_WINDOW)
            
            # Create feature vector for current state
            feature_vector = pd.DataFrame([[
                nb_A_ma,
                nb_W_ma,
            ]], columns=['nb_A', 'nb_W'])
            
            # Scale features
            X_scaled = scaler.transform(feature_vector)
            
            # Update sequence buffer with current features
            sequence_buffer.append(X_scaled[0])
            
            # If we have enough sequence data, make prediction for current value
            if len(sequence_buffer) == Constants.SEQUENCE_LENGTH:
                # Make prediction using current sequence
                X_sequence = np.array(list(sequence_buffer))
                X_sequence = X_sequence.reshape((1, Constants.SEQUENCE_LENGTH, X_scaled.shape[1]))
                prediction = float(model.predict(X_sequence, verbose=0)[0][0])
                
                # Add current actual value and its prediction
                actual_values.append(nb_A_W_ma)
                predicted_values.append(prediction)
                
                # Update time steps to show the latest window
                if len(actual_values) <= Constants.PLOT_WINDOW:
                    time_steps = list(range(len(actual_values)))
                else:
                    # Slide the window to show latest points
                    time_steps = list(range(len(actual_values) - Constants.PLOT_WINDOW, len(actual_values)))
                    actual_values = actual_values[-Constants.PLOT_WINDOW:]
                    predicted_values = predicted_values[-Constants.PLOT_WINDOW:]
                
                # Update plot
                actual_line.set_data(time_steps, actual_values)
                pred_line.set_data(time_steps, predicted_values)
                
                # Update axis limits to show moving window
                ax.set_xlim(time_steps[0], time_steps[-1] + 1)
                ax.relim()
                ax.autoscale_view(scaley=True)  # Only autoscale y-axis
                plt.draw()
                plt.pause(0.1)
                
                # Print current values
                print(f"\rStep {len(time_steps) + (len(actual_values) - Constants.PLOT_WINDOW if len(actual_values) > Constants.PLOT_WINDOW else 0)} - "
                      f"Actual: {nb_A_W_ma:.2f}, Prediction: {prediction:.2f}, "
                      f"Difference: {abs(nb_A_W_ma - prediction):.2f}", end="")
            else:
                print(f"\rCollecting data: {len(sequence_buffer)}/{Constants.SEQUENCE_LENGTH}", end="")
            
            features.reset()
            last_save_time = current_time
