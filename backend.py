from flask import Flask, jsonify
import pybgpstream
import time
import os
import numpy as np
from bgp_features import BGPFeatures
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
from constants import Constants
from flask_cors import CORS
from collections import deque
import threading

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join('models', f'g3_lstm_{Constants.TIME_WINDOW}.h5')
SCALER_PATH = os.path.join('scalers', f'g3_scaler_{Constants.TIME_WINDOW}.pkl')

model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

recent_records = []
sequence_buffer = deque(maxlen=Constants.SEQUENCE_LENGTH)
current_actual_A = 0
current_prediction_A = 0
current_actual_W = 0
current_prediction_W = 0
current_actual_A_W = 0
current_prediction_A_W = 0

@app.route('/data')
def get_data():
    return jsonify({
        "actual_A": float(current_actual_A),
        "prediction_A": float(current_prediction_A),
        "actual_W": float(current_actual_W),
        "prediction_W": float(current_prediction_W),
        "actual_A_W": float(current_actual_A_W),
        "prediction_A_W": float(current_prediction_A_W),
    })

def calculate_moving_average(records, field, window_size):
    if not records:
        return 0
    values = [r[field] for r in records[-window_size:]]
    return round(sum(values) / len(values), 2)

def bgp_collector():
    global recent_records, sequence_buffer, current_actual_A, current_prediction_A, current_actual_W, current_prediction_W, current_actual_A_W, current_prediction_A_W
    
    stream = pybgpstream.BGPStream(project="ris-live")
    features = BGPFeatures()
    last_save_time = time.time()

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

                    prediction = model.predict(X_sequence, verbose=0)[0]
                    prediction_reshaped = prediction.reshape(1, -1)
                    prediction = scaler.inverse_transform(prediction_reshaped)

                    current_actual_A = nb_A_ma
                    current_prediction_A = prediction[0][0]

                    current_actual_W = nb_W_ma
                    current_prediction_W = prediction[0][1]

                    current_actual_A_W = nb_A_W_ma
                    current_prediction_A_W = prediction[0][2]

                sequence_buffer.append(X_scaled)
                features.reset()
                last_save_time = current_time

if __name__ == '__main__':
    collector_thread = threading.Thread(target=bgp_collector)
    collector_thread.daemon = True
    collector_thread.start()

    app.run(port=3000)
