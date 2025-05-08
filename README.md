# CS420_FP
# README

# data / installs
unable to upload model due to size, please download dataset and put into Model folder: <br/>
  https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat<br/>

also must have dlib and imutils installed:<br/>
  pip install opencv-python numpy dlib imutils<br/>
  ^ directly install<br/>
  or might need to download through anaconda navigator...<br/>
  
# running
How to run code well:<br/>
  all you need to do after installing everything required is to run the file (420final.py)<br/>
  then wait for the camera and the canvas to load, and drag so you can see both tabs at the same time<br/>

IMPORTANT<br/>
  should have good lighting, no sun towards camera<br/>
  i reccomend holding your phone flashlight behind your camera for better eye and pupil detection<br/>
  should be eye level and center with the camera, otherwise it can detect other featuers or not detect the eyes<br/>

# controls
facial controls:<br/>
  blink to cycle through colors you draw, KRGB<br/>
  left wink to clear the canvas<br/>
  right wink to fill the canvas with the current color<<br/>
