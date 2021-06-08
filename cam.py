import numpy as np
import cv2 as cv
import os
from keras import models
import tensorflow as tf

cam_source = 'http://192.168.18.6:4747/video'
cam = cv.VideoCapture(cam_source)

model = models.load_model('model.h5')
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
    prediction_class = model.predict(image).argmax(axis=-1)
    if prediction < 0.4:
        frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    print(prediction, " ", prediction_class)
    cv.imshow('Cam', frame)
    if cv.waitKey(1) == ord('q'):
        break
cam.release()
cv.destroyAllWindows()
