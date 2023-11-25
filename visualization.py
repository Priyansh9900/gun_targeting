import numpy as np
import cv2
import time


# Function to draw the path on the visualization window
def draw_path(path, background_image, clap_detected, clap_index, frame_rate, aim_start_time, start_d, start_a):
    path_visualization = np.copy(background_image)
    
    start_point_visualization = (
    500 + int(start_d * np.cos(np.radians(start_a))),
    500 + int(start_d * np.sin(np.radians(start_a)))
)
    pink_circle_center = (0, 0)

    for i in range(1, len(path)):
        # Draw a line between consecutive points
        pt1 = (path[i - 1][1] + start_point_visualization[0], path[i - 1][2] + start_point_visualization[1])
        pt2 = (path[i][1] + start_point_visualization[0], path[i][2] + start_point_visualization[1])

        # Avoid drawing lines that go beyond the frame boundaries
        if 0 <= pt1[0] < path_visualization.shape[1] and 0 <= pt1[1] < path_visualization.shape[0] \
                and 0 <= pt2[0] < path_visualization.shape[1] and 0 <= pt2[1] < path_visualization.shape[0]:
            # Determine the color based on time
            color = get_line_color(i, clap_index, frame_rate, clap_detected)
            if clap_detected and i >= clap_index:
                cv2.line(path_visualization, pt1, pt2, color, 1)
                if i == clap_index:
                    center_circle = (pt2[0], pt2[1])
                    radius_circle = 15
                    pink_color = (255, 0, 255, 100)  # Set alpha channel to 100 for partial transparency
                    overlay = np.copy(path_visualization)
                    cv2.circle(overlay, center_circle, radius_circle, pink_color, -1)
                    cv2.addWeighted(overlay, 1 - pink_color[3] / 255, path_visualization, pink_color[3] / 255, 0,
                                    path_visualization)
                    pink_circle_center = center_circle

                     # Start the aim timer when the line reaches the circle of radius 276
                    aim_start_time = i / frame_rate if aim_start_time is None else aim_start_time
            else:
                cv2.line(path_visualization, pt1, pt2, color, 1)

    return path_visualization, pink_circle_center


def update_pink_circle_coordinates(pink_circle_coordinates, center_circle):
    pink_circle_coordinates.append(center_circle)
    return pink_circle_coordinates


def log_pink_circle_coordinates(filename, coordinates, background_image, scores, aim_time, inside_ring, aim_trace_speed, aim_trace_speed2, inside_ring_avg):
    mean_x = int(np.mean([c[0] for c in coordinates]))
    mean_y = int(np.mean([c[1] for c in coordinates]))
    distances = [np.sqrt((c[0] - mean_x)**2 + (c[1] - mean_y)**2) for c in coordinates]
    
    with open(filename, 'w') as file:
        file.write("Index,X,Y,Direction,Score,Aim Time,10.0,S1,S2,DA,10a.0\n")
        for index, (coord, score, at, ten, SOne, STwo, distance, tenA) in enumerate(zip(coordinates, scores, aim_time, inside_ring, aim_trace_speed, aim_trace_speed2, distances, inside_ring_avg)):
            direction = calculate_direction(coord, (background_image.shape[1] // 2, background_image.shape[0] // 2))  # Assuming center is defined
            file.write(f"{index + 1},{coord[0]},{coord[1]},{direction},{score},{at},{ten},{SOne},{STwo},{distance},{tenA}\n")

def calculate_inside_ring_avg(coordinates, path):
    mean_x = int(np.mean([c[0] for c in coordinates]))
    mean_y = int(np.mean([c[1] for c in coordinates]))
    

    # Calculate Ratio of time inside ring from avg aim point
    # Go back 1 second and check if each dot in the path was inside the 10th ring
    current_time = time.time()
    start_time = current_time - 1.0
    prev_dot_time = start_time
    inside_ring2 = 0
    for dot_time, dot_distance, dot_angle in reversed(path):
        # Check if the dot was within 1 second before the clap
        if dot_time < start_time:
            break

        # Convert polar coordinates to Cartesian coordinates
        dot_x = dot_distance * np.cos(np.radians(dot_angle))
        dot_y = dot_distance * np.sin(np.radians(dot_angle))
        distance = np.sqrt((dot_x - mean_x)**2 + (dot_y - mean_y)**2)

        # Check if the dot was inside a radius of 6 from the background center
        if distance <= 6:
            inside_ring2 += abs(prev_dot_time - dot_time)
        
        # Update prev_dot_time for the next iteration
        prev_dot_time = dot_time
    return inside_ring2

def calculate_direction(coord, center):
    # Calculate the angle between the line connecting the center and the point
    # and the positive x-axis (in degrees)
    
    angle_rad = np.arctan2(coord[1] - center[1], coord[0] - center[0])
    angle_deg = np.degrees(angle_rad)

    # Convert angle to positive values
    if angle_deg < 0:
        angle_deg += 360

    # Define directional ranges
    direction_ranges = {
        (22.5, 67.5): 'North-East',
        (67.5, 112.5): 'North',
        (112.5, 157.5): 'North-West',
        (157.5, 202.5): 'West',
        (202.5, 247.5): 'South-West',
        (247.5, 292.5): 'South',
        (292.5, 337.5): 'South-East',
        (337.5, 22.5): 'East'
    }

    # Determine the direction
    for angle_range, direction in direction_ranges.items():
        if angle_range[0] <= angle_deg < angle_range[1]:
            return direction

    return 'Unknown'

def calculate_score(scores, pink_circle_center, background_image_shape):
    print("Calculate Score", pink_circle_center)
    center_x, center_y = background_image_shape[0] // 2, background_image_shape[1] // 2
    distance_from_center = np.sqrt((pink_circle_center[0] - center_x)**2 + (pink_circle_center[1] - center_y)**2)
    print(distance_from_center)

    # Define radius ranges
    radius_ranges = [
        (0, 6),
        (6, 36),
        (36, 66),
        (66, 96),
        (96, 126),
        (126, 156),
        (156, 186),
        (186, 216),
        (216, 246),
        (246, 276)
    ]

    # Determine the score based on the distance from the center
    for i, (min_radius, max_radius) in enumerate(radius_ranges):
        if min_radius <= distance_from_center < max_radius:
            scores.append(10 - i)  # Scores are assigned in descending order

    return 0

def calculate_aim_trace_speed(aim_trace_positions, frame_rate):
    distances = [np.sqrt((aim_trace_positions[i][0] - aim_trace_positions[i - 1][0]) ** 2 +
                         (aim_trace_positions[i][1] - aim_trace_positions[i - 1][1]) ** 2)
                 for i in range(1, len(aim_trace_positions))]
    total_distance = sum(distances)
    total_time = (len(aim_trace_positions) - 1) / frame_rate
    aim_trace_speed = total_distance / total_time if total_time > 0 else 0.0
    print("aim_trace_speed", aim_trace_speed)
    return aim_trace_speed

def save_series_shots(pink_circle_coordinates, shot_image, output_folder='Output'):
    # Calculate the mean of X and Y values
    mean_x = int(np.mean([coord[0] for coord in pink_circle_coordinates]))
    mean_y = int(np.mean([coord[1] for coord in pink_circle_coordinates]))

    # Draw a gray plus sign at the mean position
    plus_size = 20
    plus_color = (128, 128, 128)  # Gray color
    cv2.line(shot_image, (mean_x - plus_size, mean_y), (mean_x + plus_size, mean_y), plus_color, 2)
    cv2.line(shot_image, (mean_x, mean_y - plus_size), (mean_x, mean_y + plus_size), plus_color, 2)

    # Draw pink circles and indices
    for index, coord in enumerate(pink_circle_coordinates):
        radius_circle = 15
        pink_color = (255, 0, 255, 100)  # Set alpha channel to 100 for partial transparency
        overlay = np.copy(shot_image)

        # Draw the shot on the path_visualization
        cv2.circle(overlay, coord, radius_circle, pink_color, -1)
        cv2.addWeighted(overlay, 1 - pink_color[3] / 255, shot_image, pink_color[3] / 255, 0, shot_image)

        # Add the index in gray color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        font_color = (0, 0, 0)  # Black color
        index_text = str(index + 1)
        text_size = cv2.getTextSize(index_text, font, font_scale, font_thickness)[0]
        text_position = (coord[0] - text_size[0] // 2, coord[1] + text_size[1] // 2)
        cv2.putText(shot_image, index_text, text_position, font, font_scale, font_color, font_thickness)

    # Display the resulting image
    cv2.imshow('Series Shots', shot_image)
    cv2.waitKey(0)  # Wait until any key is pressed
    cv2.destroyAllWindows()

    # Save the shot image
    shot_filename = f'{output_folder}/series_shots.png'
    cv2.imwrite(shot_filename, shot_image)


def get_line_color(current_index, clap_index, frame_rate, clap_detected):
    time_before_clap_ms = (clap_index - current_index) * (1000 / frame_rate)

    if 0 <= time_before_clap_ms <= 300 and clap_detected:
        return (255, 0, 0)  # Blue
    elif 300 < time_before_clap_ms <= 1000 and clap_detected:
        return (0, 0, 0)  # Yellow
    elif clap_detected and time_before_clap_ms < 0:
        return (0, 255, 0)  # Green
    else:
        return (0, 0, 255)  # Red (default color)
