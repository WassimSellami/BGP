from constants import Constants
import os
import pandas as pd
import csv

TRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), 'train_data')
INPUT_FILE = os.path.join(TRAIN_DATA_DIR, 'rrc12-ma-5-prof.csv')
FILTERED_OUTPUT_FILE = os.path.join(TRAIN_DATA_DIR, 'rrc12-5.csv')
MA_OUTPUT_FILE = os.path.join(TRAIN_DATA_DIR, 'rrc12-ma-5-g3.csv')

def filter_features(input_file=INPUT_FILE):
    df = pd.read_csv(input_file)
    
    selected_features = [Constants.FEATURE_NB_A, Constants.FEATURE_NB_W, Constants.FEATURE_NB_A_W]
    df_processed = df[selected_features]
    
    df_processed = df_processed.round(2)
    
    df_processed.to_csv(FILTERED_OUTPUT_FILE, index=False)
    print(f"Filtered data saved to {FILTERED_OUTPUT_FILE}")
    print(f"Shape of filtered data: {df_processed.shape}")
    return df_processed

def calculate_moving_average(records, field, window_size):
    """
    Calculate moving average exactly like in real_time_predictor
    """
    if not records:
        return 0
    values = [r[field] for r in records[-window_size:]]
    return round(sum(values) / len(values), 2)

def read_and_process_file(window_size=Constants.MA_WINDOW):
    """
    Read CSV and process it with moving averages like in real-time predictor
    """
    df = pd.read_csv(INPUT_FILE)
    
    recent_records = []
    
    with open(MA_OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([Constants.FEATURE_NB_A, Constants.FEATURE_NB_W, Constants.FEATURE_NB_A_W, 
                        Constants.FEATURE_NB_A_MA, Constants.FEATURE_NB_W_MA])
        
        for idx, row in df.iterrows():
            current_record = {
                Constants.FEATURE_NB_A: row[Constants.FEATURE_NB_A],
                Constants.FEATURE_NB_W: row[Constants.FEATURE_NB_W],
                Constants.FEATURE_NB_A_W: row[Constants.FEATURE_NB_A_W]
            }
            
            recent_records.append(current_record)
            if len(recent_records) > window_size:
                recent_records.pop(0)
            
            nb_A_ma = calculate_moving_average(recent_records, Constants.FEATURE_NB_A, window_size)
            nb_W_ma = calculate_moving_average(recent_records, Constants.FEATURE_NB_W, window_size)
            
            writer.writerow([
                current_record[Constants.FEATURE_NB_A],
                current_record[Constants.FEATURE_NB_W],
                current_record[Constants.FEATURE_NB_A_W],
                nb_A_ma,
                nb_W_ma
            ])

if __name__ == "__main__":
    
    read_and_process_file()
    print(f"Data with moving averages saved to {MA_OUTPUT_FILE}")
