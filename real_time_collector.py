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
    writer.writerow([Constants.FEATURE_NB_A, Constants.FEATURE_NB_W, Constants.FEATURE_NB_A_W])

recent_records = []

for elem in stream:
    if Constants.CHOSEN_COLLECTOR in elem.collector:
        features.classify_elem(elem.type)
        current_time = time.time()
        
        if current_time - last_save_time >= Constants.TIME_WINDOW:
             with open('output/real_time_collector.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    features.nb_A,
                    features.nb_W,
                    features.nb_A_W])
                features.reset()
                last_save_time = current_time
            
