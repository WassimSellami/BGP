import pybgpstream
import csv
import time
import pandas as pd
import numpy as np
from bgp_features import BGPFeatures
from pre_process_data import create_moving_average_features

TIME_WINDOW = 5
REAL_TIME_FEATURES_FILENAME = "output/real_time_collector.csv"
CHOSEN_COLLECTOR = "rrc12"
MA_WINDOW = 10

stream = pybgpstream.BGPStream(
    project="ris-live",
)
features = BGPFeatures()

last_save_time = time.time()

# with open(REAL_TIME_FEATURES_FILENAME, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['nb_A', 'nb_W', 'nb_A_W', 'nb_A_ma', 'nb_W_ma'])

recent_records = []

for elem in stream:
    # timestamp = elem.time
    if CHOSEN_COLLECTOR in elem.collector:
        print(elem)
        features.classify_elem(elem.type)
        current_time = time.time()
        
        if current_time - last_save_time >= TIME_WINDOW:
            current_record = {
                'nb_A': features.nb_A,
                'nb_W': features.nb_W,
                'nb_A_W': features.nb_A_W
            }
            
            recent_records.append(current_record)
            
            if len(recent_records) > MA_WINDOW:
                recent_records.pop(0)
            
            df = pd.DataFrame(recent_records)
            if not df.empty:
                df_ma = create_moving_average_features(df, window_size=MA_WINDOW)
                if not df_ma.empty:
                    latest_record = df_ma.iloc[-1]
                    
                    with open(REAL_TIME_FEATURES_FILENAME, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            latest_record['nb_A'],
                            latest_record['nb_W'],
                            latest_record['nb_A_W'],
                            latest_record['nb_A_ma'],
                            latest_record['nb_W_ma']
                        ])

            features.reset()
            last_save_time = current_time
            
    time.sleep(0.0001)
