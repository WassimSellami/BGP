import pandas as pd
import numpy as np
import csv

INPUT_FILE = 'test_data/test_rrc12-1-prof.csv'
FILTERED_OUTPUT_FILE = 'test_data/test_rrc12-1.csv'
MA_OUTPUT_FILE = 'test_data/test_rrc12-ma-1-g3.csv'

def filter_features(input_file=INPUT_FILE):
    df = pd.read_csv(input_file)
    
    selected_features = ['nb_A', 'nb_W', 'nb_A_W']
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

def read_and_process_file(input_file, output_file, window_size=10):
    """
    Read CSV and process it with moving averages like in real-time predictor
    """
    # Read the input CSV
    df = pd.read_csv(input_file)
    
    # Process records one by one like in streaming
    recent_records = []
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['nb_A', 'nb_W', 'nb_A_W', 'nb_A_ma', 'nb_W_ma'])
        
        for idx, row in df.iterrows():
            current_record = {
                'nb_A': row['nb_A'],
                'nb_W': row['nb_W'],
                'nb_A_W': row['nb_A_W']
            }
            
            # Add to recent records and maintain window
            recent_records.append(current_record)
            if len(recent_records) > window_size:
                recent_records.pop(0)
            
            # Calculate moving averages
            nb_A_ma = calculate_moving_average(recent_records, 'nb_A', window_size)
            nb_W_ma = calculate_moving_average(recent_records, 'nb_W', window_size)
            
            # Write row with moving averages
            writer.writerow([
                current_record['nb_A'],
                current_record['nb_W'],
                current_record['nb_A_W'],
                nb_A_ma,
                nb_W_ma
            ])

if __name__ == "__main__":
    input_file = "train_data/rrc12-5.csv"
    output_file = "train_data/rrc12-ma-5-g3.csv"
    
    read_and_process_file(input_file, output_file)
    print(f"Data with moving averages saved to {output_file}")
