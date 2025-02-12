from constants import Constants
import os
import pandas as pd

INPUT_FILE = os.path.join(os.path.dirname(__file__), f'generated_data/g3_rrc12_{Constants.TIME_WINDOW}_generated_300.csv')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), f'test_data/new_new_test_data_{Constants.TIME_WINDOW}.csv')

def calculate_moving_average(data, window_size=Constants.MA_WINDOW):
    """Calculate moving average for a pandas series"""
    return data.rolling(window=window_size, min_periods=1).mean().round(2)

def smooth_features(input_file=INPUT_FILE, window_size=Constants.MA_WINDOW):
    df = pd.read_csv(input_file)
    
    df[Constants.FEATURE_NB_A] = calculate_moving_average(df[Constants.FEATURE_NB_A], window_size)
    df[Constants.FEATURE_NB_W] = calculate_moving_average(df[Constants.FEATURE_NB_W], window_size)
    df[Constants.FEATURE_NB_A_W] = calculate_moving_average(df[Constants.FEATURE_NB_A_W], window_size)
    
    print(f"Shape of smoothed data: {df.shape}")
    return df

if __name__ == "__main__":
    df_smoothed = smooth_features()
    df_smoothed.to_csv(OUTPUT_FILE, index=False)
    print(f"Smoothed data saved to {OUTPUT_FILE}")
