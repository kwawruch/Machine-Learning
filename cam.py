import cv2
import numpy as np
import cv2 as cv
import os
from keras import models
from PIL import Image
import tensorflow as tf
from imageai.Detection import VideoObjectDetection

cam_source = 'http://192.168.18.6:4747/video'

path = os.getcwd()
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
    im = Image.fromarray(frame, 'RGB')
    im = im.resize((64,64))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    if prediction < 0.4:
        frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    print(prediction)
    cv.imshow('Cam', frame)
    if cv.waitKey(1) == ord('q'):
        break
cam.release()
cv.destroyAllWindows()



'''
if not cam.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, stream = cam.read()
    if not ret:
        print("Can't receive stream...")
        break
    cv.imshow('Camera 1', stream)
    if cv.waitKey(1) == ord('q'):
        break
cam.release()
cv.destroyAllWindows()
'''

'''
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(path , "model.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(camera_input=cam,
                                output_file_path=os.path.join(path, "Cam")
                                , frames_per_second=24, log_progress=True)
print(video_path)
'''