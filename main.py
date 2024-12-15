import pybgpstream
import csv
import time
import os
from bgp_features import BGPFeatures
TIME_WINDOW = 5
REAL_TIME_FEATURES_FILENAME = "real_time_bgp_features.csv"

features_filename = REAL_TIME_FEATURES_FILENAME

stream = pybgpstream.BGPStream(
    project="routeviews-stream",
    filter="router amsix",
)

# stream = pybgpstream.BGPStream(
#     # Consider this time interval:
#     # Sat, 01 Aug 2015 7:50:00 GMT -  08:10:00 GMT
#     # from_time="2015-08-01 07:50:00", until_time="2015-08-01 08:10:00",
#     collectors=["rrc12"],
#     record_type="ribs",
# )

last_save_time = time.time()
features = BGPFeatures()


# current_filename = "current.csv"
# previous_filename = "previous.csv"



# with open(current_filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Timestamp', 'Prefix', 'Next-hop', 'AS Path', 'Other Fields'])

with open(features_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'nb_A', 'nb_W', 'nb_A_W'])

for elem in stream:
    timestamp = elem.time
    # readable_time = time.ctime(timestamp)
    # prefix = elem.fields.get('prefix', 'N/A')
    # next_hop = elem.fields.get('next-hop', 'N/A')
    # as_path = elem.fields.get('as-path', 'N/A')

    features.classify_elem(elem.type)

    # with open(current_filename, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([readable_time, prefix, next_hop, as_path])

    current_time = time.time()
    if current_time - last_save_time >= TIME_WINDOW:
        # if os.path.exists(previous_filename):
        #     os.remove(previous_filename)
            
        # os.rename(current_filename, previous_filename)
        
        # with open(current_filename, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['Timestamp', 'Prefix', 'Next-hop', 'AS Path', 'Other Fields'])

        with open(features_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                last_save_time,
                features.nb_A,
                features.nb_W,
                features.announcements_and_withdrawals_count
            ])
        
        features.reset()

        last_save_time = current_time

    time.sleep(0.0001)
