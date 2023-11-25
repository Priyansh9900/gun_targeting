import cv2 
import numpy as np
import streamlit as st
import threading
import sounddevice as sd
from geometry_utils import calculate_distance_and_angle
from visualization import draw_path, log_pink_circle_coordinates, update_pink_circle_coordinates, save_series_shots, calculate_score, calculate_aim_trace_speed, calculate_inside_ring_avg
from log_utils import log_distance_angle
import time



          # streamlit functioning
st.set_page_config(layout="wide")
"# This is Aim Trainer"
'''

## This is simulation for 2 different shots

Although we used same video

'''

start_button = st.button("start processing")
          # streamlit functioning

          
          
if start_button :
    print("start")

    # Set the clap template (you can adjust this based on your clap sound)
    clap_template = np.array([1, 1, 1])

    # List to store pink circle coordinates
    pink_circle_coordinates = []
    scores = []
    aim_time=[]
    aim_start_time = None
    inside_ring = []

    aim_trace_speed = []
    aim_trace_speed2 = []
    aim_trace_positions = []

    inside_ring_avg = []

    #pink circle ceter
    pink_circle_center = (0,0)


    start_d, start_a = 0, 0


    def isDetected():
        def callback(indata, frames, time2, status):
            global clap_detected, clap_index, inside_ring, aim_trace_positions, aim_trace_speed, aim_trace_speed2  # Use the global keywords

            if status:
                print(status)

            # Calculate the energy of the audio signal
            energy = np.sum(np.square(indata[:, 0]))

            # Check if the energy exceeds the threshold
            if not clap_detected and energy > 5:
                print("CLAPPED!")
                clap_detected = True
                clap_index = len(path)  # Set the clap index to the current path length

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
                    distance = np.sqrt((dot_x)**2 + (dot_y)**2)

                    # Check if the dot was inside a radius of 6 from the background center
                    if distance <= 6:
                        inside_ring2 += abs(prev_dot_time - dot_time)
                
                    # Update prev_dot_time for the next iteration
                    prev_dot_time = dot_time
                inside_ring.append(inside_ring2)

                # Calculate aim trace speed
                aim_trace_speed.append(calculate_aim_trace_speed(aim_trace_positions[-frame_rate:], frame_rate) * 0.01)
                aim_trace_speed2.append(calculate_aim_trace_speed(aim_trace_positions[-int(frame_rate * 0.25):], frame_rate) * 0.01)

        # Set up the default microphone input
        sample_rate = sd.query_devices(None, 'input')['default_samplerate']

        with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
            "Listening for a clap..."
            try:
                while not clap_detected:
                    pass  # Keep the function running until a clap is detected
            except KeyboardInterrupt:
                pass  # Allow the user to interrupt the function with Ctrl+C




    for run_num in range(1):  # Set the number of runs here

        # Read the video
        video = cv2.VideoCapture('Test_Videos/video-test-4.mp4')
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))
        video.set(cv2.CAP_PROP_FPS, frame_rate)

        # Define the new resolution
        new_width = 500
        new_height = 500

        # Set the new resolution for the video
        video.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

        # Reset variables for each run
        clap_detected = False
        clap_index = 0
        path = []

        ok, frame = video.read()

        resized_frame = cv2.resize(frame, (500, 500))

        # Select ROI
        bbox = cv2.selectROI(resized_frame)

#streamlit        
        st.write(bbox)
        start_point = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

        # Start the clap detection in a separate thread
        clap_thread = threading.Thread(target=isDetected)
        clap_thread.start()

        # Load an image as the background for the visualization window
        background_image = cv2.imread('bg.png')  # Replace with your image file
        if background_image is None:
            raise ValueError("Error: Background image not found!")

        # Set a fixed size for the background
        background_size = (500, 500)  # Adjust the size as needed
        background_image = cv2.resize(background_image, background_size)

        # Create a new window for path visualization with the size of the ROI
        path_visualization = np.copy(background_image)

        # Initialize the tracker with the ROI
        tracker = cv2.TrackerKCF_create()
        ok = tracker.init(resized_frame, bbox)

        # Initialize the path with the starting position
        start_point = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        path = [(time.time(), start_point[0], start_point[1])]


        # Define the codec and create a video writer object for frame tracking
        fourcc_tracking = cv2.VideoWriter_fourcc(*'XVID')
        out_tracking = cv2.VideoWriter(f'Output/frame_tracking_{run_num}.avi', fourcc_tracking, frame_rate, (resized_frame.shape[1], resized_frame.shape[0]))

        # Define the codec and create a video writer object for path visualization
        fourcc_path = cv2.VideoWriter_fourcc(*'XVID')
        out_path = cv2.VideoWriter(f'Output/path_visualization_{run_num}.avi', fourcc_path, frame_rate, (path_visualization.shape[1], path_visualization.shape[0]))

        # Define the log file name
        log_filename = f'Output/distance_angle_log_{run_num}.csv'
    
    
        #image container for stream lit pathvisualization
        col1, col2 = st.columns(2)
        img_container1 = st.empty()
        img_container2 = st.empty()
        txt_container = st.write("hey what i'm doing ")

        while True:
            ok, frame = video.read()
        
            if not ok:
                break  # Break out of the loop if there are no more frames to read

            resized_frame = cv2.resize(frame, (500, 500))
            # Write the frame to the tracking video
            out_tracking.write(resized_frame)

            ok, bbox = tracker.update(resized_frame)
            if ok:
                (x, y, w, h) = [int(v) for v in bbox]

                # Draw rectangle
                cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (0, 255, 0), 2, 1)
                # Calculate center coordinates
                center_x = x + w // 2
                center_y = y + h // 2
                if start_d == 0 or start_a == 0:
                    start_d, start_a = calculate_distance_and_angle((center_x, center_y), (resized_frame.shape[1], resized_frame.shape[0]))
                    st.write(center_x, center_y, resized_frame.shape[1], resized_frame.shape[0], "CENTERS")

                # Draw red dot at the center
                red_dot = (center_x, center_y)
                red = (0, 0, 255)
                green = (0, 255, 0)

                # Check for the condition to start the timer
                if ((center_x - background_image.shape[1] // 2) ** 2 + (
                        center_y - background_image.shape[0] // 2) ** 2) ** 0.5 <= 276:
                    aim_start_time = time.time() if aim_start_time is None else aim_start_time

                # Check for the transition and draw the appropriate dot
                if clap_detected and len(path) >= clap_index:
                    # Draw green dot
                    cv2.circle(resized_frame, red_dot, 3, green, -1)

                    # Check for the condition to end the timer
                    if ((center_x - background_image.shape[1] // 2) ** 2 + (
                            center_y - background_image.shape[0] // 2) ** 2) ** 0.5 <= 276:
                        aim_end_time = time.time()

                        # Calculate the aim time
                        aim_time.append(aim_end_time - aim_start_time)
                    
                else:
                    # Draw red dot
                    cv2.circle(resized_frame, red_dot, 3, red, -1)

                # Calculate distance and angle
                distance, angle = calculate_distance_and_angle(start_point, red_dot)

                # Display distance and angle on the frame
                cv2.putText(resized_frame, f'Distance: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(resized_frame, f'Angle: {angle:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Update the path with the new position
                path.append((time.time(), int(distance * np.cos(np.radians(angle))),
                             int(distance * np.sin(np.radians(angle)))))

                # Call draw_path and update pink_circle_coordinates
                path_visualization, pink_circle_center = draw_path(path, background_image, clap_detected, clap_index, frame_rate, aim_start_time, start_d, start_a)

                # Calculate distance and angle
                distance, angle = calculate_distance_and_angle(start_point, red_dot)

                # Get the current time in milliseconds
                time_ms = int(round(time.time() * 1000))

                # Log distance and angle data
                log_distance_angle(log_filename, time_ms, distance, angle)

                # Write the frame to the video file for path visualization
                out_path.write(path_visualization)

            else:
                cv2.putText(resized_frame, 'Error', (100, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            with col1:
                img_container1.image(resized_frame, channels="BGR")
                
            with col2:
                img_container2.image(path_visualization, channels="BGR")
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

# line break
        st.image('Output/series_shots.png')

        with open("Output/pink_circle_coordinates.log", "r") as f:
            file_contents = f.read()
            st.write(file_contents)
         

        st.write("---")

        update_pink_circle_coordinates(pink_circle_coordinates, pink_circle_center)
        calculate_score(scores, pink_circle_center, background_image.shape)

        inside_ring_avg.append(calculate_inside_ring_avg(pink_circle_coordinates, path))

        # Join the clap detection thread after finishing the run
        clap_thread.join()

        # Release resources after each run
        out_path.release()
        out_tracking.release()

        # Release the video capture resources
        video.release()
        cv2.destroyAllWindows()


    log_pink_circle_coordinates('Output/pink_circle_coordinates.log', pink_circle_coordinates,background_image, scores, aim_time, inside_ring, aim_trace_speed, aim_trace_speed2, inside_ring_avg)
    final_bg = np.copy(background_image)
    save_series_shots(pink_circle_coordinates, final_bg)
