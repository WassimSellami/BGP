import pybgpstream
import csv
import time
import numpy as np
from bgp_features import BGPFeatures
from tensorflow.keras.models import load_model
import pickle
import os

TIME_WINDOW = 5
REAL_TIME_FEATURES_FILENAME = "output/real_time_collector_with_predictions.csv"
CHOSEN_COLLECTOR = "rrc12"
MA_WINDOW = 10
SEQUENCE_LENGTH = 24
MODEL_PATH = os.path.join('prof', 'model', 'lstm_model.h5')
SCALER_PATH = os.path.join('prof', 'scaler', 'scaler.pkl')

def calculate_moving_average(records, field, window_size):
    if not records:
        return 0
    values = [r[field] for r in records[-window_size:]]
    return round(sum(values) / len(values), 2)

def create_time_features(record):
    return [
        record['nb_A_W'],
        record['nb_A_ma'],
        record['nb_W_ma']
    ]

def window_data(X, window=24):
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
    writer.writerow(['nb_A', 'nb_W', 'nb_A_W', 'nb_A_ma', 'nb_W_ma'])

recent_records = []
prediction_window = []

for elem in stream:
    if CHOSEN_COLLECTOR in elem.collector:
        features.classify_elem(elem.type)
        current_time = time.time()
        
        if current_time - last_save_time >= TIME_WINDOW:
            current_record = {
                'nb_A': features.nb_A,
                'nb_W': features.nb_W,
                'nb_A_W': features.nb_A_W
            }
            
            recent_records.append(current_record)
            if len(recent_records) > MA_WINDOW:
                recent_records.pop(0)
            
            nb_A_ma = calculate_moving_average(recent_records, 'nb_A', MA_WINDOW)
            nb_W_ma = calculate_moving_average(recent_records, 'nb_W', MA_WINDOW)
            
            current_record['nb_A_ma'] = nb_A_ma
            current_record['nb_W_ma'] = nb_W_ma
            
            feature_vector = create_time_features(current_record)
            
            scaled_features = scaler.transform([feature_vector])[0]
            prediction_window.append(scaled_features)
            
            if len(prediction_window) > SEQUENCE_LENGTH:
                prediction_window.pop(0)
            
            predictions = [0, 0]
            if len(prediction_window) == SEQUENCE_LENGTH:
                X = window_data(np.array(prediction_window))
                predictions = model.predict(X, verbose=0)[0]
                print(f"\nPredictions for next interval:")
                print(f"Predicted Announcements (nb_A): {predictions[0]:.2f}")
                print(f"Predicted Withdrawals (nb_W): {predictions[1]:.2f}")
            
            with open(REAL_TIME_FEATURES_FILENAME, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    current_record['nb_A'],
                    current_record['nb_W'],
                    current_record['nb_A_W'],
                    nb_A_ma,
                    nb_W_ma
                ])
                writer.writerow([
                    round(predictions[0], 2),
                    round(predictions[1], 2)
                ])
            features.reset()
            last_save_time = current_time
            
