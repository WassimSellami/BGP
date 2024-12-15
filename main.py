import pybgpstream
import csv
import time
from collections import deque

# Set the time window duration
TIME_WINDOW = 10  # in seconds

# Initialize the BGPStream
stream = pybgpstream.BGPStream(
    project="routeviews-stream",  # Example project (modify as per your use case)
    filter="router amsix",        # Example filter, change as needed
)

# Track the time window
last_save_time = time.time()

# Start the first file creation
current_time = time.time()
time_str = time.strftime("%M_%S", time.localtime(current_time))
filename = f"bgp_data_{time_str}.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['Timestamp', 'Prefix', 'Next-hop', 'AS Path', 'Other Fields'])

# Iterate through the stream to capture live data
for elem in stream:
    # Extract timestamp
    timestamp = elem.time
    # Convert timestamp to a human-readable format
    readable_time = time.ctime(timestamp)
    
    # Extract other relevant fields (you can add or modify fields)
    prefix = elem.fields.get('prefix', 'N/A')
    next_hop = elem.fields.get('next-hop', 'N/A')
    as_path = elem.fields.get('as-path', 'N/A')

    # Open the CSV file in append mode to dynamically add records
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the new record to the CSV
        writer.writerow([readable_time, prefix, next_hop, as_path])

    # Check if 10 seconds have passed since the last save
    current_time = time.time()
    if current_time - last_save_time >= TIME_WINDOW:
        # Get current time for the filename (t1_t2 format)
        time_str = time.strftime("%M_%S", time.localtime(current_time))
        filename = f"bgp_data_{time_str}.csv"
        
        # Create a new CSV file with the updated timestamp and write the header row
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Prefix', 'Next-hop', 'AS Path', 'Other Fields'])

        # Update the last save time to the current time (this is the new window start)
        last_save_time = current_time

    # Sleep for a short period to avoid flooding the stream
    time.sleep(0.0001)  # Adjust based on your requirements
