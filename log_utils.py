# log_utils.py
import csv

def log_distance_angle(filename, time_ms, distance, angle):
    """
    Log the Distance and Angle data to a CSV file.

    Parameters:
    - filename: Name of the CSV file
    - time_ms: Time in milliseconds
    - distance: Distance value
    - angle: Angle value
    """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time_ms, distance, angle])
