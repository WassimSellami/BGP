import pybgpstream
import csv
from datetime import datetime, timedelta
from bgp_features import BGPFeatures
from constants import Constants

GENERATED_DATA_FILENAME = "generated_data/generated_data1.csv"
FROM_TIME = datetime.strptime("2023-06-09 00:10:00", "%Y-%m-%d %H:%M:%S")
ROW_COUNT = 180000
TIME_OFFSET  = 7200
UNTIL_TIME = FROM_TIME + timedelta(seconds=ROW_COUNT * Constants.TIME_WINDOW)

from_time=FROM_TIME.strftime("%Y-%m-%d %H:%M:%S")

stream = pybgpstream.BGPStream(
    from_time=FROM_TIME.strftime("%Y-%m-%d %H:%M:%S"),
    until_time=UNTIL_TIME.strftime("%Y-%m-%d %H:%M:%S"),
    collectors=["rrc12"]
    )

features = BGPFeatures()
features_dict = {i: {Constants.FEATURE_NB_A: 0, Constants.FEATURE_NB_W: 0, Constants.FEATURE_NB_A_W: 0} for i in range(ROW_COUNT)}

for elem in stream:
    timestamp = elem.time - TIME_OFFSET

    update_time = datetime.fromtimestamp(timestamp)
    time_diff = (update_time - FROM_TIME).total_seconds()
    window_index = int(time_diff // Constants.TIME_WINDOW)
    
    if 0 <= window_index < ROW_COUNT:
        if elem.type == 'A':
            features_dict[window_index][Constants.FEATURE_NB_A] += 1
        elif elem.type == 'W':
            features_dict[window_index][Constants.FEATURE_NB_W] += 1
            features_dict[window_index][Constants.FEATURE_NB_A_W] = features_dict[window_index][Constants.FEATURE_NB_A] + features_dict[window_index][Constants.FEATURE_NB_W]

with open(GENERATED_DATA_FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([Constants.FEATURE_NB_A, Constants.FEATURE_NB_W, Constants.FEATURE_NB_A_W])
    
    for i in range(ROW_COUNT):
        writer.writerow([
            features_dict[i][Constants.FEATURE_NB_A],
            features_dict[i][Constants.FEATURE_NB_W],
            features_dict[i][Constants.FEATURE_NB_A_W]
        ])

