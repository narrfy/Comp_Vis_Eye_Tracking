# CS420_FP
# README

# data / installs
unable to upload model due to size, please download dataset and put into Model folder: 
  https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat

also must have dlib and imutils installed:
  pip install opencv-python numpy dlib imutils
  ^ directly install
  or might need to download through anaconda navigator...
  
# running
How to run code well:
  all you need to do after installing everything required is to run the file (420final.py)
  then wait for the camera and the canvas to load, and drag so you can see both tabs at the same time

IMPORTANT
  should have good lighting, no sun towards camera
  i reccomend holding your phone flashlight behind your camera for better eye and pupil detection
  should be eye level and center with the camera, otherwise it can detect other featuers or not detect the eyes

# controls
facial controls:
  blink to cycle through colors you draw, KRGB
  left wink to clear the canvas
  right wink to fill the canvas with the current color
