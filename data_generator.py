import pybgpstream
import csv
from datetime import datetime, timedelta
from bgp_features import BGPFeatures
TIME_WINDOW = 30

TRAIN_FEATURES_FILENAME = "test_features.csv"
FROM_TIME = datetime.strptime("2017-07-09 00:10:00", "%Y-%m-%d %H:%M:%S")
ROW_COUNT = 200
TIME_OFFSET  = 7200
UNTIL_TIME = FROM_TIME + timedelta(seconds=ROW_COUNT * TIME_WINDOW)

from_time=FROM_TIME.strftime("%Y-%m-%d %H:%M:%S")

stream = pybgpstream.BGPStream(
    from_time=FROM_TIME.strftime("%Y-%m-%d %H:%M:%S"),
    until_time=UNTIL_TIME.strftime("%Y-%m-%d %H:%M:%S"),
    collectors=["rrc12"],
    record_type="updates",
)

features = BGPFeatures()
features_dict = {i: {'nb_A': 0, 'nb_W': 0, 'nb_A_W': 0} for i in range(ROW_COUNT)}

for elem in stream:
    timestamp = elem.time - TIME_OFFSET

    update_time = datetime.fromtimestamp(timestamp)
    time_diff = (update_time - FROM_TIME).total_seconds()
    window_index = int(time_diff // TIME_WINDOW)
    
    if 0 <= window_index < ROW_COUNT:
        if elem.type == 'A':
            features_dict[window_index]['nb_A'] += 1
        elif elem.type == 'W':
            features_dict[window_index]['nb_W'] += 1
            features_dict[window_index]['nb_A_W'] = features_dict[window_index]['nb_A'] + features_dict[window_index]['nb_W']

with open(TRAIN_FEATURES_FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'nb_A', 'nb_W', 'nb_A_W'])
    
    for i in range(ROW_COUNT):
        timestamp = FROM_TIME + timedelta(seconds=i * TIME_WINDOW)
        writer.writerow([
            timestamp,
            features_dict[i]['nb_A'],
            features_dict[i]['nb_W'],
            features_dict[i]['nb_A_W']
        ])

