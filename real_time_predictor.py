import pybgpstream
import csv
import time
import numpy as np
from bgp_features import BGPFeatures
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from collections import deque
from constants import (
    FEATURE_NB_A,
    FEATURE_NB_A_MA,
    FEATURE_NB_A_W,
    FEATURE_NB_W,
    FEATURE_NB_W_MA,
    TIME_WINDOW,
    MA_WINDOW,
    SEQUENCE_LENGTH,
    PLOT_WINDOW,
    CHOSEN_COLLECTOR,
    REAL_TIME_FEATURES_FILENAME,
    MODEL_PATH,
    SCALER_PATH
)

def calculate_moving_average(records, field, window_size):
    if not records:
        return 0
    values = [r[field] for r in records[-window_size:]]
    return round(sum(values) / len(values), 2)

def create_time_features(record):
    return [
        record[FEATURE_NB_A_W],
        record[FEATURE_NB_A_MA],
        record[FEATURE_NB_W_MA]
    ]

def window_data(X, window=SEQUENCE_LENGTH):
    x = []
    for i in range(window-1, len(X)):
        x.append(X[i-window+1:i+1])
    return np.array(x)

model = load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

stream = pybgpstream.BGPStream(
    project="ris-live",
)
features = BGPFeatures()

last_save_time = time.time()

with open(REAL_TIME_FEATURES_FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([FEATURE_NB_A, FEATURE_NB_W, FEATURE_NB_A_W, FEATURE_NB_A_MA, FEATURE_NB_W_MA])

recent_records = []
prediction_window = []

nb_A_history = deque(maxlen=PLOT_WINDOW)
nb_W_history = deque(maxlen=PLOT_WINDOW)
pred_A_history = deque(maxlen=PLOT_WINDOW)
pred_W_history = deque(maxlen=PLOT_WINDOW)

plt.ion() 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

for elem in stream:
    if CHOSEN_COLLECTOR in elem.collector:
        features.classify_elem(elem.type)
        current_time = time.time()
        
        if current_time - last_save_time >= TIME_WINDOW:
            current_record = {
                FEATURE_NB_A: features.nb_A,
                FEATURE_NB_W: features.nb_W,
                FEATURE_NB_A_W: features.nb_A_W
            }
            
            recent_records.append(current_record)
            if len(recent_records) > MA_WINDOW:
                recent_records.pop(0)
            
            nb_A_ma = calculate_moving_average(recent_records, FEATURE_NB_A, MA_WINDOW)
            nb_W_ma = calculate_moving_average(recent_records, FEATURE_NB_W, MA_WINDOW)
            
            current_record[FEATURE_NB_A_MA] = nb_A_ma
            current_record[FEATURE_NB_W_MA] = nb_W_ma
            
            feature_vector = create_time_features(current_record)
            
            scaled_features = scaler.transform([feature_vector])[0]
            prediction_window.append(scaled_features)
            
            if len(prediction_window) > SEQUENCE_LENGTH:
                prediction_window.pop(0)
            
            predictions = [0, 0]
            if len(prediction_window) == SEQUENCE_LENGTH:
                X = window_data(np.array(prediction_window))
                predictions = model.predict(X, verbose=0)[0]
                
                nb_A_history.append(current_record[FEATURE_NB_A])
                nb_W_history.append(current_record[FEATURE_NB_W])
                pred_A_history.append(predictions[0])
                pred_W_history.append(predictions[1])
                
                ax1.clear()
                ax2.clear()
                
                ax1.set_xlabel('Time Steps')
                ax1.set_ylabel('Number of Announcements')
                ax1.set_title('BGP Updates Real-time Prediction - Announcements')
                ax1.plot(list(nb_A_history), label='Actual nb_A', 
                        linewidth=1.0, color='blue', alpha=0.8)
                ax1.plot(list(pred_A_history), label='Predicted nb_A', 
                        linewidth=1.0, color='orange', alpha=0.8)
                ax1.legend()
                
                ax2.set_xlabel('Time Steps')
                ax2.set_ylabel('Number of Withdrawals')
                ax2.set_title('BGP Updates Real-time Prediction - Withdrawals')
                ax2.plot(list(nb_W_history), label='Actual nb_W', 
                        linewidth=1.0, color='blue', alpha=0.8)
                ax2.plot(list(pred_W_history), label='Predicted nb_W', 
                        linewidth=1.0, color='orange', alpha=0.8)
                ax2.legend()
                
                plt.tight_layout()
                plt.draw()
                plt.pause(0.1) 
            
            with open(REAL_TIME_FEATURES_FILENAME, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    current_record[FEATURE_NB_A],
                    current_record[FEATURE_NB_W],
                    current_record[FEATURE_NB_A_W],
                    nb_A_ma,
                    nb_W_ma
                ])
                writer.writerow([
                    round(predictions[0], 2),
                    round(predictions[1], 2)
                ])
            features.reset()
            last_save_time = current_time
            
