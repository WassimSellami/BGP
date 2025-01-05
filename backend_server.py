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

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join('model', 'lstm_model.h5')
SCALER_PATH = os.path.join('scaler', 'scaler.pkl')

def calculate_moving_average(records, field, window_size):
    if not records:
        return 0
    values = [r[field] for r in records[-window_size:]]
    return round(sum(values) / len(values), 2)

# Load model and scaler
model = load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Global variables to store state
recent_records = []
sequence_buffer = deque(maxlen=Constants.SEQUENCE_LENGTH)
current_actual = 0
next_prediction = None

@app.route('/data')
def get_data():
    return jsonify({
        "actual": current_actual,
        "prediction": next_prediction
    })

def bgp_collector():
    global recent_records, sequence_buffer, current_actual, next_prediction
    
    stream = pybgpstream.BGPStream(
        project="ris-live",
    )
    features = BGPFeatures()
    last_save_time = time.time()

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
                
                # Create feature vector for prediction
                feature_vector = pd.DataFrame([[
                    current_record['nb_A'],
                    current_record['nb_W'],
                    nb_A_ma,
                    nb_W_ma
                ]], columns=['nb_A', 'nb_W', 'nb_A_ma', 'nb_W_ma'])
                
                # Scale features
                X_scaled = scaler.transform(feature_vector)
                
                # Add to sequence buffer
                sequence_buffer.append(X_scaled[0])
                
                # Update current actual value
                current_actual = current_record['nb_A_W']
                
                # Make prediction when we have enough sequence data
                if len(sequence_buffer) == Constants.SEQUENCE_LENGTH:
                    # Reshape sequence for LSTM [samples, time steps, features]
                    X_sequence = np.array(list(sequence_buffer))
                    X_sequence = X_sequence.reshape((1, Constants.SEQUENCE_LENGTH, X_scaled.shape[1]))
                    
                    # Make prediction for next time step
                    next_prediction = float(model.predict(X_sequence, verbose=0)[0][0])
                
                features.reset()
                last_save_time = current_time

if __name__ == '__main__':
    import threading
    collector_thread = threading.Thread(target=bgp_collector)
    collector_thread.daemon = True
    collector_thread.start()
    
    app.run(port=3000) 