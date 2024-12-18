import pandas as pd
import numpy as np

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

def create_moving_average_features(df, window_size=10):
    """
    Calculate moving averages using available data up to window_size
    
    Args:
        df: pandas DataFrame with the filtered features
        window_size: maximum size of the moving average window
    """
    df_ma = df.copy()
    
    # For each row, use min(window_size, available_rows) for the moving average
    for i in range(len(df_ma)):
        start_idx = max(0, i - window_size + 1)
        window_data_A = df_ma['nb_A'].iloc[start_idx:i+1]
        window_data_W = df_ma['nb_W'].iloc[start_idx:i+1]
        
        df_ma.loc[df_ma.index[i], 'nb_A_ma'] = window_data_A.mean()
        df_ma.loc[df_ma.index[i], 'nb_W_ma'] = window_data_W.mean()
    
    df_ma = df_ma.round(2)
    
    if MA_OUTPUT_FILE:
        df_ma.to_csv(MA_OUTPUT_FILE, index=False)
        print(f"Data with moving average saved to {MA_OUTPUT_FILE}")
        print(f"Shape of data with moving average: {df_ma.shape}")
    
    return df_ma

if __name__ == "__main__":
    df_filtered = filter_features()
    df_with_ma = create_moving_average_features(df_filtered)
