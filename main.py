import pybgpstream
import csv
import time
from bgp_features import BGPFeatures
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE
import pickle
from collections import deque
from tensorflow.keras.optimizers import Adam

TIME_WINDOW = 5
REAL_TIME_FEATURES_FILENAME = "output/real_time_collector_with_predictions.csv"
SEQUENCE_LENGTH = 10
CHOSEN_COLLECTOR = "rrc12"

stream = pybgpstream.BGPStream(
    project="ris-live",
)
features = BGPFeatures()


model = load_model('bgp_lstm_model.h5', compile=False)  # Load without compilation
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=MeanSquaredError(),
    metrics=['mse']
)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


last_save_time = time.time()

recent_observations = deque(maxlen=SEQUENCE_LENGTH)

with open(REAL_TIME_FEATURES_FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'nb_A', 'nb_W', 'nb_A_W'])

for elem in stream:
    timestamp = elem.time
    if CHOSEN_COLLECTOR in elem.collector:
        print(elem)
        features.classify_elem(elem.type)
        current_time = time.time()
        if current_time - last_save_time >= TIME_WINDOW:
            current_features = [features.nb_A, features.nb_W, features.nb_A_W]
            
            scaled_features = scaler.transform([current_features])[0]
            recent_observations.append(scaled_features)
            
            predictions = [0, 0]
            if len(recent_observations) == SEQUENCE_LENGTH:
                X = np.array([list(recent_observations)])
                scaled_predictions = model.predict(X, verbose=0)[0]
                predictions = scaler.inverse_transform([scaled_predictions])[0]
            
            with open(REAL_TIME_FEATURES_FILENAME, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    time.ctime(current_time),
                    features.nb_A,
                    features.nb_W,
                    features.nb_A_W,
                ])
                writer.writerow([
                    'predictions: ',
                    round(predictions[0], 2),
                    round(predictions[1], 2),
                    predictions[0] + predictions[1] 
                ])

            features.reset()
            last_save_time = current_time

    time.sleep(0.0001)
