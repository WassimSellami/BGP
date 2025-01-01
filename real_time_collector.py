import pybgpstream
import csv
import time
from constants import Constants
from bgp_features import BGPFeatures


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

with open('output/real_time_collector.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([Constants.FEATURE_NB_A, Constants.FEATURE_NB_W, Constants.FEATURE_NB_A_W, Constants.FEATURE_NB_A_MA, Constants.FEATURE_NB_W_MA])

recent_records = []

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
            
            with open('output/real_time_collector.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    current_record[Constants.FEATURE_NB_A],
                    current_record[Constants.FEATURE_NB_W],
                    current_record[Constants.FEATURE_NB_A_W],
                    nb_A_ma,
                    nb_W_ma
                ])

            features.reset()
            last_save_time = current_time
            
