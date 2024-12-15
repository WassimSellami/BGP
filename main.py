import pybgpstream
import csv
import time
from bgp_features import BGPFeatures
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE
import pickle
from collections import deque

TIME_WINDOW = 5
REAL_TIME_FEATURES_FILENAME = "real_time_bgp_features.csv"
SEQUENCE_LENGTH = 10  # Must match the sequence length used during training

# Load the model and scaler
custom_objects = {
    'loss': MeanSquaredError(),
    'mse': MSE()
}
model = load_model('bgp_lstm_model.h5', custom_objects=custom_objects)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

features_filename = REAL_TIME_FEATURES_FILENAME

stream = pybgpstream.BGPStream(
    project="ris-live",
    # filter="collector rrc12",
    record_type="updates",
)

last_save_time = time.time()
features = BGPFeatures()

# Initialize a deque to store recent observations
recent_observations = deque(maxlen=SEQUENCE_LENGTH)

with open(features_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'nb_A', 'nb_W', 'nb_A_W', 'predicted_nb_A', 'predicted_nb_W'])

for elem in stream:
    timestamp = elem.time
    features.classify_elem(elem.type)

    current_time = time.time()
    if current_time - last_save_time >= TIME_WINDOW:
        current_features = [features.nb_A, features.nb_W, features.nb_A_W]
        
        # Scale current features
        scaled_features = scaler.transform([current_features])[0]
        recent_observations.append(scaled_features)
        
        # Make prediction if we have enough observations
        predictions = [0, 0]  # Default values
        if len(recent_observations) == SEQUENCE_LENGTH:
            X = np.array([list(recent_observations)])
            scaled_predictions = model.predict(X, verbose=0)[0]
            predictions = scaler.inverse_transform([scaled_predictions])[0]
        
        with open(features_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                time.ctime(current_time),
                features.nb_A,
                features.nb_W,
                features.nb_A_W,
                round(predictions[0], 2),  # predicted_nb_A
                round(predictions[1], 2)   # predicted_nb_W
            ])
        
        features.reset()
        last_save_time = current_time

    time.sleep(0.0001)
