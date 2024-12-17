import pybgpstream
import csv
import time
from bgp_features import BGPFeatures

TIME_WINDOW = 5
REAL_TIME_FEATURES_FILENAME = "output/real_time_collector.csv"
CHOSEN_COLLECTOR = "rrc12"

stream = pybgpstream.BGPStream(
    project="ris-live",
)
features = BGPFeatures()

last_save_time = time.time()

with open(REAL_TIME_FEATURES_FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'nb_A', 'nb_W', 'nb_A_W'])

for elem in stream:
    timestamp = elem.time
    if CHOSEN_COLLECTOR in elem.collector:
        print(elem)
        features.classify_elem(elem.type)
        current_time = time.time()
        if current_time - last_save_time >= TIME_WINDOW:
            current_features = [features.nb_A, features.nb_W, features.nb_A_W]
            
            with open(REAL_TIME_FEATURES_FILENAME, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    time.ctime(current_time),
                    features.nb_A,
                    features.nb_W,
                    features.nb_A_W
                ])

            features.reset()
            last_save_time = current_time
    time.sleep(0.0001)
