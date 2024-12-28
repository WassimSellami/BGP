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

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join('model', 'lstm_model.h5')
SCALER_PATH = os.path.join('scaler', 'scaler.pkl')

def calculate_moving_average(records, field, window_size):
    if not records:
        return 0
    values = [r[field] for r in records[-window_size:]]
    return round(sum(values) / len(values), 2)

def create_time_features(record):
    df = pd.DataFrame([{
        Constants.FEATURE_NB_A: record[Constants.FEATURE_NB_A],
        Constants.FEATURE_NB_W: record[Constants.FEATURE_NB_W],
        Constants.FEATURE_NB_A_W: record[Constants.FEATURE_NB_A_W],
        Constants.FEATURE_NB_A_MA: record[Constants.FEATURE_NB_A_MA],
        Constants.FEATURE_NB_W_MA: record[Constants.FEATURE_NB_W_MA]
    }])
    return df

def window_data(X, window=Constants.SEQUENCE_LENGTH):
    x = []
    for i in range(window-1, len(X)):
        x.append(X[i-window+1:i+1])
    return np.array(x)

# Load model and scaler
model = load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Global variables to store state
recent_records = []
feature_sequences = []
current_predictions = {"nb_A": 0, "nb_W": 0}
current_values = {"nb_A": 0, "nb_W": 0}

@app.route('/data')
def get_data():
    return jsonify({
        "current": current_values,
        "predictions": current_predictions
    })

def bgp_collector():
    global recent_records, feature_sequences, current_predictions, current_values
    
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
                    Constants.FEATURE_NB_A: features.nb_A,
                    Constants.FEATURE_NB_W: features.nb_W,
                    Constants.FEATURE_NB_A_W: features.nb_A_W
                }
                
                recent_records.append(current_record)
                if len(recent_records) > Constants.MA_WINDOW:
                    recent_records.pop(0)
                
                nb_A_ma = calculate_moving_average(recent_records, Constants.FEATURE_NB_A, Constants.MA_WINDOW)
                nb_W_ma = calculate_moving_average(recent_records, Constants.FEATURE_NB_W, Constants.MA_WINDOW)
                
                current_record[Constants.FEATURE_NB_A_MA] = nb_A_ma
                current_record[Constants.FEATURE_NB_W_MA] = nb_W_ma
                
                features_df = create_time_features(current_record)
                scaled_features = scaler.transform(features_df)[0]
                feature_sequences.append(scaled_features)
                
                if len(feature_sequences) > Constants.SEQUENCE_LENGTH:
                    feature_sequences.pop(0)
                
                if len(feature_sequences) == Constants.SEQUENCE_LENGTH:
                    X = np.array([feature_sequences])
                    predictions = model.predict(X, verbose=0)[0]
                    current_predictions = {
                        "nb_A": float(predictions[0]),
                        "nb_W": float(predictions[1])
                    }
                
                current_values = {
                    "nb_A": features.nb_A,
                    "nb_W": features.nb_W
                }
                
                features.reset()
                last_save_time = current_time

if __name__ == '__main__':
    import threading
    collector_thread = threading.Thread(target=bgp_collector)
    collector_thread.daemon = True
    collector_thread.start()
    
    app.run(port=3000) 