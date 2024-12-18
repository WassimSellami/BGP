import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os
import time

WINDOW_LENGTH = 24
TIME_WINDOW = 1  # Matching collector's time window
MODEL_PATH = os.path.join('prof', 'model', 'lstm_model.h5')
SCALER_PATH = os.path.join('prof', 'scaler', 'scaler.pkl')
COLLECTOR_FILE = 'output/real_time_collector.csv'

def create_time_features(df, target=None):
    df_1 = pd.DataFrame(df, columns=['nb_A_W', 'nb_A_ma', 'nb_W_ma'])
    
    if target:
        if isinstance(target, list):
            y = df[target]
        else:
            y = df[target]
        return df_1, y
    return df_1

def window_data(X, window=24):
    x = []
    for i in range(window-1, len(X)):
        x.append(X[i-window+1:i+1])
    return np.array(x)

def make_real_time_prediction():
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    last_row_count = 0
    
    while True:
        try:
            df = pd.read_csv(COLLECTOR_FILE)
            
            if len(df) == last_row_count:
                time.sleep(TIME_WINDOW)
                continue
                
            last_row_count = len(df)
            
            if len(df) < WINDOW_LENGTH:
                print(f"Waiting for more data... Currently have {len(df)} samples")
                time.sleep(TIME_WINDOW)
                continue
            
            df_window = df.tail(WINDOW_LENGTH).copy()
            
            X_test_df = create_time_features(df_window)
            
            X_test = scaler.transform(X_test_df)
            
            X_test_w = window_data(X_test, window=WINDOW_LENGTH)
            
            if len(X_test_w) > 0:
                predictions = model.predict(X_test_w, verbose=0)
                predicted_nb_A = predictions[0, 0]
                predicted_nb_W = predictions[0, 1]
                
                current_values = df.iloc[-1]
                print("\nLatest Data Point:")
                print(f"Actual Announcements: {current_values['nb_A']:.2f}")
                print(f"Actual Withdrawals: {current_values['nb_W']:.2f}")
                print("\nPredictions for next time step:")
                print(f"Predicted Announcements: {predicted_nb_A:.2f}")
                print(f"Predicted Withdrawals: {predicted_nb_W:.2f}")
                print("-" * 50)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(TIME_WINDOW)
            continue
        
        time.sleep(TIME_WINDOW)

if __name__ == "__main__":
    print("Starting real-time predictions...")
    make_real_time_prediction() 