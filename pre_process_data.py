from constants import Constants
import os
import pandas as pd

TRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), 'train_data')
INPUT_FILE = os.path.join(TRAIN_DATA_DIR, 'rrc12-ma-5-prof.csv')
OUTPUT_FILE = os.path.join(TRAIN_DATA_DIR, 'rrc12-ma-5-g3.csv')

def filter_features(input_file=INPUT_FILE):
    df = pd.read_csv(input_file)
    
    selected_features = [Constants.FEATURE_NB_A, Constants.FEATURE_NB_W, Constants.FEATURE_NB_A_W]
    df_processed = df[selected_features]
    
    df_processed = df_processed.round(2)
    
    print(f"Shape of filtered data: {df_processed.shape}")
    return df_processed

if __name__ == "__main__":
    df_filtered = filter_features()
    df_filtered.to_csv(OUTPUT_FILE, index=False)
    print(f"Filtered data saved to {OUTPUT_FILE}")
