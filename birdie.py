import cv2
import numpy as np
import datetime
import time
import os

def detect_motion(frame, avg_frame, motion_threshold=25, min_area=500): # function that detects motion in front of the camera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if avg_frame is None:
        avg_frame = gray.copy().astype("float")
        return avg_frame, False

    cv2.accumulateWeighted(gray, avg_frame, 0.5)
    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
    
    thresh = cv2.threshold(frame_delta, motion_threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    if int(cv2.__version__.split('.')[0]) >= 4:
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        return avg_frame, True
    
    return avg_frame, False

def detect_good_lighting(frame, brightness_threshold=110): # function to determine whether a camera frame has good lighting
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()
    return brightness > brightness_threshold

def create_daily_directory(base_dir): # create a directory for video storage
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    daily_dir = os.path.join(base_dir, current_date)
    if not os.path.exists(daily_dir):
        os.makedirs(daily_dir)
    return daily_dir

# Create the 'bird' base directory if it doesn't exist
base_directory = "bird"
os.makedirs(base_directory, exist_ok=True)

cap = cv2.VideoCapture(0)  # initialize the capture device
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

codec = cv2.VideoWriter_fourcc(*'mp4v')

avg_frame = None
recording = False
last_motion_time = time.time()
light_condition = False

while True: #main loop
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    avg_frame, motion_detected = detect_motion(frame, avg_frame)
    good_lighting = detect_good_lighting(frame)

    if motion_detected: #loop that saves the video if motion is detected (ie a bird is present)
        last_motion_time = time.time()
        if not recording:
            # Start recording
            recording = True
            start_time = time.time()
            daily_directory = create_daily_directory(base_directory)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_type = "light" if good_lighting else "motion" #sort into light and motion based on lighting in the frame
            video_filename = os.path.join(daily_directory, f"{video_type}_{timestamp}.mp4")
            out = cv2.VideoWriter(video_filename, codec, fps, (frame_width, frame_height))
        
    if recording:
        out.write(frame)
        cv2.imshow("Video", frame)

        # Stop recording after 1 minute or if no motion for 5 seconds
        if time.time() - start_time >= 60 or time.time() - last_motion_time > 5:
            recording = False
            out.release()
            os.chmod(video_filename, 0o664)  # Adjust file write permissions

    key = cv2.waitKey(1)
    if key == 27:  # Press Esc to exit
        if recording:
            out.release()
            os.chmod(video_filename, 0o664)  # Adjust file write permissions
        break

cap.release()
cv2.destroyAllWindows()
