import pybgpstream
import csv
import time
from bgp_features import BGPFeatures

TIME_WINDOW = 5
REAL_TIME_FEATURES_FILENAME = "output/real_time_collector.csv"
CHOSEN_COLLECTOR = "rrc12"
MA_WINDOW = 10

def calculate_moving_average(records, field, window_size):
    if not records:
        return 0
    values = [r[field] for r in records[-window_size:]]
    return round(sum(values) / len(values), 2)

stream = pybgpstream.BGPStream(
    project="ris-live",
)


features = BGPFeatures()

last_save_time = time.time()

with open(REAL_TIME_FEATURES_FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['nb_A', 'nb_W', 'nb_A_W', 'nb_A_ma', 'nb_W_ma'])

recent_records = []

for elem in stream:
    if CHOSEN_COLLECTOR in elem.collector:
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
            
            nb_A_ma = calculate_moving_average(recent_records, 'nb_A', MA_WINDOW)
            nb_W_ma = calculate_moving_average(recent_records, 'nb_W', MA_WINDOW)
            
            with open(REAL_TIME_FEATURES_FILENAME, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    current_record['nb_A'],
                    current_record['nb_W'],
                    current_record['nb_A_W'],
                    nb_A_ma,
                    nb_W_ma
                ])

            features.reset()
            last_save_time = current_time
            
