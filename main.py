import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import tensorflow as tf
import cv2
import os
import numpy as np

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range = 40,
    zoom_range = 0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip = True,
    vertical_flip=True,
    )
test_datagen = ImageDataGenerator(
    rescale=1/255
)
img_size = 64
batch_size = 30


train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True,
    seed=3,
    class_mode='categorical',
    color_mode='rgb'
    #save_to_dir='data/tmp'
)
print(train_generator.labels)
print(type(train_generator))

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True,
    seed=3,
    class_mode='categorical',
    color_mode='rgb'
)
print(train_generator.samples)
print(test_generator.samples)
x,y = test_generator.next()

model = Sequential()
model.add(Conv2D(32, 3, activation="relu", input_shape=(img_size, img_size, 3)))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.6))
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.4))
model.add(Conv2D(64, 3, activation="relu"))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)
model.summary()

results = model.evaluate(x,y, batch_size=1)
print("results:",results)

predict = model.predict(x[0].reshape(1,img_size,img_size,3))
print("prediction:",predict)

#model.save("model.h5")
#tf.saved_model.save(model, "saved-model/1/")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
