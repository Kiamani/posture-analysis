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
    a = np.array(a) # First
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

def calculate_slope(point1, point2):
    # Calculate slope between two points
    if (point2.x - point1.x) != 0:  # Prevent division by zero
        return (point2.y - point1.y) / (point2.x - point1.x)
    else:
        return float('inf')

# Read the image
image_path = "C:/Users/kian.imani/Desktop/folders/irandoc/images/4/front.jpg"  # Replace "your_image.jpg" with the path to your image
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
    
    text_color=(255,255,255)
    #angle = calculate_angle(shoulder, elbow, wrist)
    slope = calculate_slope(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    slope_round= round(slope, 5)
    if slope_round >= 0.1 :
        text_color=(255,0,0)
    if slope_round <= -0.1:
        text_color = (255,0,0)
    # Visualize angle
    cv2.putText(image, str(slope_round), 
                   tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)
    


               
    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                             )               
    cv2.imshow('Mediapipe Feed', image)
    
    # Save the analyzed image
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"Analyzed image saved at {output_path}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
