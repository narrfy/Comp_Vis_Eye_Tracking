# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 18:02:28 2025

@author: jacob, nate
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
# need to have cmake installed already for dlib. had to pip install dlib after downloading cmake on my machine
import dlib # landmark detection for faces, used to get eyelids for blink detection
# after installing cmake also need to: pip install opencv-python numpy dlib imutils
import imutils
from tkinter import *
from threading import *
from scipy.spatial import distance as dist # used to find euclidiean dist btw eyelids
from imutils import face_utils # used to get landmark ids of eyes



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

colors = ["black", "red", "green", "blue"]
currentColor = 0

main = Tk()
canvas = Canvas(main, width = 600, height = 600)
canvas.pack()

userX = 325
userY = 325

canvas.create_rectangle(userX,userY,userX+50,userY+50,fill="black")

updateDirection = 60

cap = cv2.VideoCapture(0)

# Variables 
right_thresh = 0.3
left_thresh = 0.3

blink_thresh = 0.45

left_wink_count = 0
right_wink_count = 0

count_frame = 0
succ_frame = 2
wink_frame_thresh = 2


# face Detection and landmarking model
detector = dlib.get_frontal_face_detector() 
landmark_predict = dlib.shape_predictor( 
'Model/shape_predictor_68_face_landmarks.dat') 

def update():
    global userX
    global userY
    global colors
    global currentColor
    
    global right_thresh
    global left_thresh
    
    global blink_thresh
    
    global right_wink_count
    global left_wink_count
    
    global count_frame
    global succ_frame
    global wink_frame_thresh
    
    ret, frame = cap.read()
    
    
    # Eye landmarks 
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 
    
    # get grayscale frame from capture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect where a face is in gray captured frame 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    id_faces = detector(gray)
    
    for face in id_faces:
        # landmarks for face detection
        shape = landmark_predict(gray, face)
        # convert to list of (x y) coords
        shape = face_utils.shape_to_np(shape)
        
        # get landmarks for both eyes
        lefteye = shape[L_start: L_end]
        righteye = shape[R_start: R_end]
        
        # calculate the EAR for both eyes
        left_EAR = calculate_EAR(lefteye) 
        right_EAR = calculate_EAR(righteye)
        
        # Avg of left and right eye EAR 
        avg = (left_EAR+right_EAR)/2
        
        # detect if blinked
        if avg < blink_thresh:  # Detect a blink if the average EAR is below the blink threshold
            count_frame += 1
        else: 
            if count_frame >= succ_frame:
                colorChange()
                print('blink')  # Blink detected
            count_frame = 0
        
        # detect left wink with similar approach
        if left_EAR < left_thresh and right_EAR > right_thresh:
            left_wink_count += 1
            right_wink_count = 0

            if left_wink_count >= wink_frame_thresh and right_EAR > blink_thresh:  # make sure the other eye is open
                print("Left wink") # never really prints, right EAR th resh probably too low or somehting
            elif left_wink_count >= wink_frame_thresh:
                print("Left wink, but other eye might be closing")
                print('resetting canvas')
                reset_canvas()
                
        # detect right wink
        elif right_EAR < right_thresh and left_EAR > left_thresh:
            right_wink_count += 1
            left_wink_count = 0
            
            if right_wink_count >= wink_frame_thresh and left_EAR > blink_thresh:  # make sure the other eye is open
                print("Right wink") # similar thresholding issue i think 
            elif right_wink_count >= wink_frame_thresh:
                print("Right wink, but other eye might be closing")
                print('filling canvas')
                fill_canvas()
        else:
            left_wink_count = 0
            right_wink_count = 0

            
        
        
        
    for (x,y,w,h) in faces:
        # get gray face reigon
        roi_gray = gray[y:y+h, x:x+w]
        # get color face reigon
        roi_color = frame[y:y+h, x:x+w]
        # detect eyes in colored face reigon
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        try:
            eyeOne = eyes[0]
            #print("Eye 1 X:", eyeOne[0])
            #print("Eye 1 Y:", eyeOne[1])
        except Exception:
            print("No eyes found in face")
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # gets 2 rows in eyes (limit to only two eyes)
        for (ex,ey,ew,eh) in eyes[:2]:
            
            if (ey - eyeOne[1]) < -10 or (ey - eyeOne[1]) > 10:
                continue
            
            # draw rectange around eye detection area
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            
            # get center of eyes at detected (x + w/2, y + h/2)
            e_center = (int(ex + ew/2), int(ey + eh/2))
            
            #print("Eye Center X", e_center[0])
            #print("Eye Center Y", e_center[1])
            # Eye center X (L - 70, R - 160)
            # Eye center Y (B - 100)
            
            # draw circle around eye
            #eye_circle = cv2.circle(roi_color,e_center,15,(0,255,0),2)
            
            # draw circle at center of eye
            eye_circle = cv2.circle(roi_color,e_center,10,(0,255,0),2)
            
            # get reigons of interest again
            eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]
            
            # threshold to help find contour
            #eye_thresh = cv2.adaptiveThreshold(eye_roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            _, eye_thresh = cv2.threshold(eye_roi_gray, 50, 255, cv2.THRESH_BINARY_INV)
            # find contour of eye
            contours, _ = cv2.findContours(eye_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # sort high->low if needed
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            
            
            if len(contours) > 0:
                # get best guess for where we think the pupil is
                pupil = max(contours, key=cv2.contourArea)
                # bound roi
                x1, y1, w1, h1 = cv2.boundingRect(pupil)
                
                # place dot where pupil is
                p_center = (int(x1 + w1/2), int(y1 + h1/2))
                p_center_face = (int((x1 + w1/2) + ex), int((y1 + h1/2) + ey))

                
                cv2.circle(eye_roi_color, p_center, 3, (0, 0, 255),-1)
                
                
                # draw a line from center to center to visualize direction...
                # not really so great right now, probably need to compare to p_center to something else
                #print('e centerX: ' + str(e_center[0]))
                #print('e centerY: ' + str(e_center[1]))
                #print('p centerX: ' + str(p_center_face[0]))
                #print('p centerY: ' + str(p_center_face[1]))
                
                
                if p_center_face[0] > e_center[0] and (userX > 0):
                    userX -= 1
                elif p_center_face[0] < e_center[0] and (userX < 575):
                    userX += 1
                    
                    
                if (p_center_face[1] > e_center[1]) and (userY < 575):
                    userY += 1
                elif p_center_face[1] < e_center[1] and (userY > 0):
                    userY -= 1
                
                
                
                
                '''
                attemmpted normalizing that didnt work correctly
                
                # Range between 60 - 150
                ex = int(e_center[1] / 90)
                
                # Range between 80 - 110
                ey = int(e_center[1] / 30)
                
                # Range between 20-40
                px = int(p_center[0] / 20)
                
                # Range between 20-30
                py = int(p_center[1] / 10)
                
                # Directions the eye is looking
                pDir = ((px-ex),(py-ey))        
                #print(pDir)
                
                cv2.line(frame, e_center, p_center, (0, 255, 0), 2)
            
                if pDir[0] <= 0:
                    
                elif pDir[0] >= 0:
                    userX += 2
                    
                if pDir[1] <= 0:
                    userY -= 2
                elif pDir[1] >= 0:
                    userY += 2
                '''
                
                
                canvas.create_rectangle(userX,userY,userX+25,userY+25,fill=colors[currentColor],outline=colors[currentColor])
                
        
            #rel = ((ex-px),(ey-py))
            
            #print(rel)
            
        #print("________________")   
    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        main.destroy()
        return
    main.after(17, update) # (16.67) Got this number from how many miliseconds a frame lasts in 60fps
    

# For when you implement blink detection, something to do after a certain amount of time
# Everything in changing the colors themselves has already been changed and tested
def colorChange():
    global colors
    global currentColor
    
    currentColor += 1
    
    if currentColor == len(colors):
        currentColor = 0
  
        
# function to clacluate eye aspect ratio. if this ratio is below a certian thresh it means the eye blinked
def calculate_EAR(eye):
    
    # clac vertical dist
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    
    # calc horizontal dist
    x1 = dist.euclidean(eye[0], eye[3])
    
    # calc eye aspect ratio
    EAR = (y1+y2) / x1
    return EAR

def reset_canvas():
    global userX, userY
    canvas.delete("all")  # Clears all items on the canvas
    canvas.create_rectangle(0, 0, 600, 600, fill="white")  # Set background to white
    
    canvas.create_rectangle(userX, userY, userX + 50, userY + 50, fill=colors[currentColor], outline=colors[currentColor])  # Recreate the rectangle in the center

def fill_canvas():
    global userX, userY
    canvas.delete("all")  # Clears all items on the canvas
    # Fill the entire canvas with the current color
    canvas.create_rectangle(0, 0, 600, 600, fill=colors[currentColor], outline=colors[currentColor])

    canvas.create_rectangle(userX, userY, userX + 50, userY + 50, fill=colors[currentColor], outline=colors[currentColor])  # Recreate the rectangle in the center

    
    
    
update()
main.mainloop()
cap.release()
cv2.destroyAllWindows()
