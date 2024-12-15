import pybgpstream
import csv
import time
import os

TIME_WINDOW = 10

stream = pybgpstream.BGPStream(
    project="routeviews-stream",
    filter="router amsix",
)

last_save_time = time.time()

current_filename = "current.csv"
previous_filename = "previous.csv"

with open(current_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Prefix', 'Next-hop', 'AS Path', 'Other Fields'])

for elem in stream:
    timestamp = elem.time
    readable_time = time.ctime(timestamp)
    prefix = elem.fields.get('prefix', 'N/A')
    next_hop = elem.fields.get('next-hop', 'N/A')
    as_path = elem.fields.get('as-path', 'N/A')

    with open(current_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([readable_time, prefix, next_hop, as_path])

    current_time = time.time()
    if current_time - last_save_time >= TIME_WINDOW:
        if os.path.exists(previous_filename):
            os.remove(previous_filename)
            
        os.rename(current_filename, previous_filename)
        
        with open(current_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Prefix', 'Next-hop', 'AS Path', 'Other Fields'])

        last_save_time = current_time

    time.sleep(0.0001)
