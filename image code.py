# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:14:11 2023

@author: kian.imani
"""
#
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import time

def calculate_angle(a, b, c):
    a = np.array(a) # First5
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def calculate_knee_distance(landmarks, mp_pose):
    left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
    right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
    
    # Calculate Euclidean distance between left and right knees
    distance = np.linalg.norm(left_knee - right_knee)
    
    return distance

def calculate_pelvic_slope(landmarks, mp_pose):
    # Get the positions of the pelvic markers
    marker1 = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
    marker2 = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y])

    # Calculate the slope between the pelvic markers
    if (marker2[0] - marker1[0]) != 0:  # Prevent division by zero
        slope = (marker2[1] - marker1[1]) / (marker2[0] - marker1[0])
    else:
        slope = float('inf')  # Handle the case of vertical line

    return slope

def calculate_eyes_slope(landmarks, mp_pose):
    # Get the positions of the left and right eye markers
    left_eye = np.array([landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER].y])
    right_eye = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER].y])

    # Calculate the slope between the eyes markers
    if (right_eye[0] - left_eye[0]) != 0:  # Prevent division by zero
        slope = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])
    else:
        slope = float('inf')  # Handle the case of vertical line

    return slope


def calculate_slope(point1, point2):
    # Calculate slope between two points
    if (point2.x - point1.x) != 0:  # Prevent division by zero
        return (point2.y - point1.y) / (point2.x - point1.x)
    else:
        return float('inf')
for folder_num in range(1,21):
    folder_name = str(folder_num)
    # Read the image
    image_path = f"C:/Users/kian.imani/Desktop/folders/irandoc/images/{folder_name}/front.jpg"
    frame = cv2.imread(image_path)
    
    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        frame = cv2.resize(frame, (1920, 1080))
        
        # Make detection
        results = pose.process(image)
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        # Extract the coordinates of specified landmarks
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    
    #convert the normalized coordinates to pixel coordinates
    #h, w, _ = image.shape
    #left_ankle_px = [int(left_ankle[0] * w), int(left_ankle[1] * h)]
    #left_hip_px = [int(left_hip[0] * w), int(left_hip[1] * h)]
    #left_knee_px = [int(left_knee[0] * w), int(left_knee[1] * h)]
    #right_ankle_px = [int(right_ankle[0] * w), int(right_ankle[1] * h)]
    #right_hip_px = [int(right_hip[0] * w), int(right_hip[1] * h)]
    #right_knee_px = [int(right_knee[0] * w), int(right_knee[1] * h)]
    
    
        
        text_color=(255,255,255)
        #angle = calculate_angle(shoulder, elbow, wrist)
        slope = calculate_slope(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        slope_round= round(slope, 5)
        if slope_round >= 0.1 :
            text_color=(255,0,0)
        if slope_round <= -0.1:
            text_color = (255,0,0)
        # Visualize angle
        #cv2.putText(image, str(slope_round), 
        #               tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)
        
    
    ################################################################3
        left_angle = calculate_angle(left_ankle,left_knee,left_hip)
        right_angle = calculate_angle(right_ankle,right_knee,right_hip)
        knee_distance = calculate_knee_distance(landmarks, mp_pose)
        pelvic_slope = calculate_pelvic_slope(landmarks, mp_pose)
        eyes_slope = calculate_eyes_slope(landmarks, mp_pose)
        shoulder_slope = calculate_slope(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        
        # Visualize angle
        #cv2.putText(image, f"Angle: {angle}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.putText(image, f"Knee Distance: {knee_distance}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.putText(image, f"Pelvic Slope: {pelvic_slope}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.putText(image, f"Eyes Slope: {eyes_slope}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.putText(image, f"Shoulder Slope: {shoulder_slope}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
       
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        cv2.imshow('Mediapipe Feed', image)
        
        folder_name = f"{folder_name}"  # Replace with your folder name
        results_dict = {
            "Left knee angle": left_angle,
            "right knee angle": right_angle,
            "Knee Distance": knee_distance,
            "Pelvic Slope": pelvic_slope,
            "Eyes Slope": eyes_slope,
            "Shoulder Slope": shoulder_slope
        }
        
        # Create DataFrame from dictionary
        df = pd.DataFrame(results_dict, index=[folder_name])
    
        # Write DataFrame to Excel
        output_file = f"{folder_name}.xlsx"
        df.to_excel(output_file)
        print(f"Results saved to {output_file}")
        # Save the analyzed image
        output_path = f"{folder_name}.jpg"
        cv2.imwrite(output_path, image)
        print(f"Analyzed image saved at {output_path}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
