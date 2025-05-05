# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 18:02:28 2025

@author: jacob, nate
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from tkinter import *
from threading import *



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

main = Tk()
canvas = Canvas(main, width = 600, height = 600)
canvas.pack()

userX = 500
userY = 500

canvas.create_rectangle(userX,userY,userX+50,userY+50,fill="black")


cap = cv2.VideoCapture(0)

def update():
    global userX
    global userY
    ret, frame = cap.read()
    
    # get grayscale frame from capture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect where a face is in gray captured frame 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        # get gray face reigon
        roi_gray = gray[y:y+h, x:x+w]
        # get color face reigon
        roi_color = frame[y:y+h, x:x+w]
        # detect eyes in colored face reigon
        eyes = eye_cascade.detectMultiScale(roi_gray)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # gets 2 rows in eyes (limit to only two eyes)
        for (ex,ey,ew,eh) in eyes[:2]:
            
            # draw rectange around eye detection area
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            
            # get center of eyes at detected (x + w/2, y + h/2)
            e_center = (int(ex + ew/2), int(ey + eh/2))
            
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
                
                cv2.circle(eye_roi_color, p_center, 3, (0, 0, 255),-1)
                
                
                # draw a line from center to center to visualize direction...
                # not really so great right now, probably need to compare to p_center to something else
                #print('e center: ' + str(e_center))
                #print('p center: ' + str(p_center))
                
                
                # Range between 60 - 150
                ex = int(e_center[0] / 90)
                
                # Range between 80 - 110
                ey = int(e_center[1] / 30)
                
                # Range between 20-40
                px = int(p_center[0] / 20)
                
                # Range between 20-30
                py = int(p_center[1] / 10)
                
                # Directions the eye is looking
                pDir = ((px-ex),(py-ey))        
                print(pDir)
                
                cv2.line(frame, e_center, p_center, (0, 255, 0), 2)
            
                if pDir[0] <= 0:
                    userX -= 1
                elif pDir[0] >= 0:
                    userX += 1
                    
                if pDir[1] <= 0:
                    userY -= 1
                elif pDir[1] >= 0:
                    userY += 1
                
                canvas.create_rectangle(userX,userY,userX+50,userY+50,fill="black")
                
            
            #rel = ((ex-px),(ey-py))
            
            #print(rel)
            
            
    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        main.destroy()
        return
    main.after(10, update)
    
update()
main.mainloop()
cap.release()
cv2.destroyAllWindows()