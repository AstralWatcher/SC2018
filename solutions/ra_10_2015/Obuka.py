import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf
import os
from keras.models import load_model
from keras.layers import LeakyReLU
from tensorflow.python.client import device_lib
from keras.layers.core import Activation

print(device_lib.list_local_devices())

batch_size = 128
num_classes = 10
epochs = 3

img_rows, img_columns = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(1, 1, img_rows, img_columns)
    x_test = x_test.reshape(1, 1, img_rows, img_columns)
    input_shape = (1, img_rows, img_columns)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_columns, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_columns, 1)
    input_shape = (img_rows, img_columns, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test)

print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

#fig = plt.figure()
#plt.subplot(2, 1, 1)
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='lower right')

#plt.subplot(2, 1, 2)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper right')

#plt.tight_layout()

model.save('7od22.h5')