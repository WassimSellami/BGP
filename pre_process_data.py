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
    Args:
        df: pandas DataFrame with the filtered features
        window_size: size of the moving average window
    """
    df_ma = df.copy()
    df_ma['nb_A_ma'] = df_ma['nb_A'].rolling(window=window_size).mean()
    df_ma['nb_W_ma'] = df_ma['nb_W'].rolling(window=window_size).mean()
    df_ma = df_ma.dropna()
    df_ma = df_ma.round(2)
    
    df_ma.to_csv(MA_OUTPUT_FILE, index=False)
    print(f"Data with moving average saved to {MA_OUTPUT_FILE}")
    print(f"Shape of data with moving average: {df_ma.shape}")
    return df_ma

if __name__ == "__main__":
    df_filtered = filter_features()
    df_with_ma = create_moving_average_features(df_filtered)
