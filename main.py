import pybgpstream
import csv
import time
from bgp_features import BGPFeatures
TIME_WINDOW = 5
REAL_TIME_FEATURES_FILENAME = "real_time_bgp_features.csv"

features_filename = REAL_TIME_FEATURES_FILENAME

stream = pybgpstream.BGPStream(
    project="ris-live",
    # filter="collector rrc12",
    record_type="updates",
)

last_save_time = time.time()
features = BGPFeatures()

with open(features_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'nb_A', 'nb_W', 'nb_A_W'])

for elem in stream:
    timestamp = elem.time
    features.classify_elem(elem.type)

    current_time = time.time()
    if current_time - last_save_time >= TIME_WINDOW:
        with open(features_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                time.ctime(current_time),
                features.nb_A,
                features.nb_W,
                features.announcements_and_withdrawals_count
            ])
        
        features.reset()

        last_save_time = current_time

    time.sleep(0.0001)
