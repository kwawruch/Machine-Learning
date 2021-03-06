import numpy as np
import cv2 as cv
from keras import models
import os
#color percentage
def retColor(percent):
    (r, g, b) = (0, 0, 0)
    if(percent>=50):
        r=510-percent*6
        g=255
        if(r<0):
            r=0
    else:
        g=percent*6
        r=255
        if(g>255):
            g=255
    return (b, g, r) #BGR

model = models.load_model('model_bg_200ep_no_2_denses.h5')
#cam
cam_source = 'videos/Ostatni.mp4'
cam = cv.VideoCapture(cam_source)
#text cam
font = cv.FONT_HERSHEY_SIMPLEX
org = (10,20)
fontScale = 0.5
thickness = 1
#labels name
folders = []
for x in os.listdir('data/train/'):
    folders.append(x)

if not cam.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cam.read()
    #frame = cv.resize(frame, (640, 640))
    if not ret:
        print("Can't receive stream...")
        break
    image = cv.resize(frame, (64, 64))
    image = image[:, :, [2, 1, 0]]
    image = image.reshape(1, 64, 64, 3)
    image = image.astype('float32')
    image = image/255
    prediction_class = model.predict(image).argmax(axis=-1)
    p_class_id = int(prediction_class)
    if(int(prediction_class) == 0):
        pred_label = folders[0]
    elif(int(prediction_class) == 1):
        pred_label = folders[1]
    elif(int(prediction_class) == 2):
        pred_label = folders[2]
    else:
        pred_label = "Unknown"
    prediction = model.predict(image)[0][p_class_id]
    p_t = str(model.predict(image))
    pred = prediction*100
    pred = int(np.rint(pred))
    pred_txt = "Prediction: "+str(pred)+"% "+pred_label
    print(pred_txt,prediction, " ", prediction_class," ",p_t)
    image = cv.putText(frame, pred_txt, org, font, fontScale, (0, 0, 0), (thickness+1), cv.LINE_AA)
    image = cv.putText(frame, pred_txt, org, font, fontScale, retColor(pred), thickness, cv.LINE_AA)
    cv.imshow('Cam', frame)
    if cv.waitKey(1) == ord('q'):
        break
cam.release()
cv.destroyAllWindows()()
