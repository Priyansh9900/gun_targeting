# geometry_utils.py
import numpy as np
import math

def calculate_distance_and_angle(center_frame, red_dot):
    distance = np.linalg.norm(np.array(center_frame) - np.array(red_dot))
    angle = math.degrees(math.atan2(red_dot[1] - center_frame[1], red_dot[0] - center_frame[0]))

    # Add 180 degrees to the angle
    angle += 180.0
    angle %= 360.0  # Ensure the angle is in the range [0, 360)

    return distance, angle
