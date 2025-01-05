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

MODEL_PATH = os.path.join('model', 'lstm_model_1.h5')
SCALER_PATH = os.path.join('scaler', 'scaler_1.pkl')

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
current_prediction = None

@app.route('/data')
def get_data():
    return jsonify({
        "actual": current_actual,
        "prediction": current_prediction
    })

def bgp_collector():
    global recent_records, sequence_buffer, current_actual, current_prediction
    
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
                    
                    # Update current actual value and its prediction
                    current_actual = nb_A_W_ma
                    current_prediction = prediction
                
                features.reset()
                last_save_time = current_time

if __name__ == '__main__':
    import threading
    collector_thread = threading.Thread(target=bgp_collector)
    collector_thread.daemon = True
    collector_thread.start()
    
    app.run(port=3000) 