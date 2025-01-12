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

os.makedirs('prediction_results', exist_ok=True)

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
    plt.savefig(f'prediction_results/g3_real_prediction_{Constants.TIME_WINDOW}_{timestamp}.png')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def calculate_moving_average(records, field, window_size):
    if not records:
        return 0
    values = [r[field] for r in records[-window_size:]]
    return round(sum(values) / len(values), 2)

model = load_model(f'models/g3_lstm_{Constants.TIME_WINDOW}.h5', compile=False)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
with open(f'scalers/g3_scaler_{Constants.TIME_WINDOW}.pkl', 'rb') as f:
    scaler = pickle.load(f)

stream = pybgpstream.BGPStream(
    project="ris-live",
)

features = BGPFeatures()
last_save_time = time.time()
recent_records = []

plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
actual_line_A, = ax.plot([], [], 'b-', label='Actual nb_A', linewidth=2)
pred_line_A, = ax.plot([], [], 'b--', label='Predicted nb_A', linewidth=2)

ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Announcements', fontsize=12, fontweight='bold')
ax.set_title('Real-time BGP Announcements vs Predictions', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.grid(True)

time_steps = []

actual_values_A = []
predicted_values_A = []

sequence_buffer = deque(maxlen=Constants.SEQUENCE_LENGTH)

print(f"Collecting initial {Constants.SEQUENCE_LENGTH} data points...")
print("Press Ctrl+C to stop and save the plot...")

for elem in stream:
    if Constants.CHOSEN_COLLECTOR in elem.collector:
        features.classify_elem(elem.type)
        current_time = time.time()

        if current_time - last_save_time >= Constants.TIME_WINDOW:
            current_record = {
                Constants.FEATURE_NB_A: features.nb_A,
                Constants.FEATURE_NB_W: features.nb_W,
                Constants.FEATURE_NB_A_W: features.nb_A_W
            }
            
            recent_records.append(current_record)
            if len(recent_records) > Constants.MA_WINDOW:
                recent_records.pop(0)

            nb_A_ma = calculate_moving_average(recent_records, Constants.FEATURE_NB_A, Constants.MA_WINDOW)
            nb_W_ma = calculate_moving_average(recent_records, Constants.FEATURE_NB_W, Constants.MA_WINDOW)
            nb_A_W_ma = calculate_moving_average(recent_records, Constants.FEATURE_NB_A_W, Constants.MA_WINDOW)

            feature_vector = pd.DataFrame([[
                nb_A_ma,
                nb_W_ma,
                nb_A_W_ma
            ]], columns=[Constants.FEATURE_NB_A, Constants.FEATURE_NB_W, Constants.FEATURE_NB_A_W])

            X_scaled = scaler.transform(feature_vector)

            if len(sequence_buffer) == Constants.SEQUENCE_LENGTH:
                X_sequence = np.array(list(sequence_buffer))
                X_sequence = X_sequence.reshape((1, Constants.SEQUENCE_LENGTH, X_scaled.shape[1]))
                print(f"X_sequence: {X_sequence}")

                prediction = model.predict(X_sequence, verbose=0)[0]
                prediction_reshaped = prediction.reshape(1, -1)
                prediction = scaler.inverse_transform(prediction_reshaped)
                print(f"Prediction: {prediction}")

                actual_values_A.append(nb_A_ma)
                predicted_values_A.append(prediction[0][0])  # Accessing the first element correctly

                if len(actual_values_A) <= Constants.PLOT_WINDOW:
                    time_steps = list(range(len(actual_values_A)))
                else:
                    time_steps = list(range(len(actual_values_A) - Constants.PLOT_WINDOW, len(actual_values_A)))
                    actual_values_A = actual_values_A[-Constants.PLOT_WINDOW:]
                    predicted_values_A = predicted_values_A[-Constants.PLOT_WINDOW:]

                actual_line_A.set_data(time_steps, actual_values_A)
                pred_line_A.set_data(time_steps, predicted_values_A)

                ax.set_xlim(time_steps[0], time_steps[-1] + 1)
                ax.relim()
                ax.autoscale_view(scaley=True)
                plt.draw()
                plt.pause(0.1)

                print(f"\rStep {len(time_steps)} - Actual nb_A: {nb_A_W_ma:.2f}, "
                      f"Predicted nb_A: {prediction[0][0]:.2f}, "
                      f"Difference: {abs(nb_A_W_ma - prediction[0][0]):.2f}", end="")

            sequence_buffer.append(X_scaled)

            features.reset()
            last_save_time = current_time
        else:
            print(f"\rCollecting data: {len(sequence_buffer)}/{Constants.SEQUENCE_LENGTH}", end="")
