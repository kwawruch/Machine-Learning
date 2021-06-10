import cv2
import numpy as np
import cv2 as cv
import os
from keras import models
import tensorflow as tf

model = models.load_model('main_model.h5')
#cam
cam_source = 'http://192.168.18.6:4747/video'
cam = cv.VideoCapture(cam_source)

#text cam
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50,465)
fontScale = 0.5
color = (0,255,0) #BGR
thickness = 1

if not cam.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cam.read()
    if not ret:
        print("Can't receive stream...")
        break
    image = cv.resize(frame, (64, 64)) # rozdzielczosc
    image = image[:, :, [2, 1, 0]] # na RGB
    image = image.reshape(1, 64, 64, 3) # dodatkowy wymiar
    image = image.astype('float32') # typ na float32
    image = image/255 # skalowanie
    prediction = model.predict(image)[0][0]
    pred = prediction*100
    pred = int(np.rint(pred))
    prediction_class = model.predict(image).argmax(axis=-1)
    if(int(prediction_class) == 0):
        pred_label = "Healthy"
    elif(int(prediction_class) == 1):
        pred_label = "Rotten"
    else:
        pred_label = "Unknown"
    pred = "Prediction: "+str(pred)+"% "+pred_label
    if prediction > 0.05:
        print(pred,prediction, " ", prediction_class)
    if prediction < 0.4:
        image = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    image = cv2.putText(frame, pred, org,font,fontScale,color,thickness,cv2.LINE_AA)
    cv.imshow('Cam', frame)
    if cv.waitKey(1) == ord('q'):
        break
cam.release()
cv.destroyAllWindows()
