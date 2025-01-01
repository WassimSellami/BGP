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
model = load_model('model/lstm_model.h5')
with open('scaler/scaler.pkl', 'rb') as f:
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
step = 0

print(f"Collecting initial {Constants.SEQUENCE_LENGTH} data points...")
print("Press Ctrl+C to stop and save the plot...")

for elem in stream:
    if Constants.CHOSEN_COLLECTOR in elem.collector:
        features.classify_elem(elem.type)
        current_time = time.time()
        
        if current_time - last_save_time >= Constants.TIME_WINDOW:
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
            
            # Create feature vector for prediction with exact column names
            feature_vector = pd.DataFrame([[
                current_record['nb_A'],
                current_record['nb_W'],
                nb_A_ma,
                nb_W_ma
            ]], columns=['nb_A', 'nb_W', 'nb_A_ma', 'nb_W_ma'])
            
            # Scale and reshape for LSTM
            X_scaled = scaler.transform(feature_vector)
            X_scaled = X_scaled.reshape((1, 1, X_scaled.shape[1]))
            
            # Make prediction
            prediction = model.predict(X_scaled, verbose=0)[0][0]
            
            # Update plot data
            actual_values.append(current_record['nb_A_W'])
            predicted_values.append(prediction)
            time_steps.append(step)
            step += 1
            
            # Update plot with sequence length window
            if len(time_steps) > Constants.SEQUENCE_LENGTH:
                time_steps = time_steps[-Constants.SEQUENCE_LENGTH:]
                actual_values = actual_values[-Constants.SEQUENCE_LENGTH:]
                predicted_values = predicted_values[-Constants.SEQUENCE_LENGTH:]
            
            if len(time_steps) >= Constants.SEQUENCE_LENGTH:
                actual_line.set_data(time_steps, actual_values)
                pred_line.set_data(time_steps, predicted_values)
                
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.1)
                
                # Print current values and difference
                difference = abs(actual_values[-1] - predicted_values[-1])
                print(f"\rActual: {actual_values[-1]:.2f}, Predicted: {predicted_values[-1]:.2f}, Difference: {difference:.2f}", end="")
            else:
                print(f"Collecting data: {len(time_steps)}/{Constants.SEQUENCE_LENGTH}", end="\r")
            
            features.reset()
            last_save_time = current_time
