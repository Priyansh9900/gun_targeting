import cv2
import numpy as np
import math

def calculate_distance_and_angle(center_frame, red_dot):
    distance = np.linalg.norm(np.array(center_frame) - np.array(red_dot))
    angle = math.degrees(math.atan2(red_dot[1] - center_frame[1], red_dot[0] - center_frame[0]))
    return distance, angle

# Read the video
video = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video.read()

    cv2.imshow('Tracking', frame)
    # Check for space or enter key press
    if cv2.waitKey(1) & 0xFF in (ord(' '), ord('\n')):
        break

# Release the capture
cv2.destroyAllWindows()


# Select ROI
bbox = cv2.selectROI(frame)

# Load an image as the background for the visualization window
background_image = cv2.imread('bg.webp')  # Replace with your image file
if background_image is None:
    raise ValueError("Error: Background image not found!")

# Resize the image to match the size of the ROI
background_image = cv2.resize(background_image, (int(bbox[2]), int(bbox[3])))

# Create a new window for path visualization with the size of the ROI
path_visualization = np.copy(background_image)

# Initialize the tracker with the ROI
tracker = cv2.TrackerKCF_create()

ok = tracker.init(frame, bbox)

# Initialize the path with the starting position
start_point = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
path = [start_point]

while True:
    ok, frame = video.read()
    if not ok:
        break
    ok, bbox = tracker.update(frame)
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2, 1)

        # Calculate center coordinates
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw red dot at the center
        red_dot = (center_x, center_y)
        cv2.circle(frame, red_dot, 3, (0, 0, 255), -1)

        # Calculate distance and angle
        distance, angle = calculate_distance_and_angle(start_point, red_dot)

        # Display distance and angle on the frame
        cv2.putText(frame, f'Distance: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Angle: {angle:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Update the path with the new position
        path.append((int(distance * np.cos(np.radians(angle))), int(distance * np.sin(np.radians(angle)))))

        # Draw path on the visualization window
        path_visualization = np.copy(background_image)
        for point in path:
            # Adjust the coordinates based on the starting point
            cv2.circle(path_visualization, (point[0]  + background_image.shape[1] // 2,
                                            point[1]  + background_image.shape[0] // 2),
                       1, (0, 0, 255), -1)

    else:
        cv2.putText(frame, 'Error', (100, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Tracking', frame)
    cv2.imshow('Path Visualization', path_visualization)
    

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()


