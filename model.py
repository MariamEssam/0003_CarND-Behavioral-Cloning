import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D
import matplotlib.pyplot as plt 

def NVIDIANet(model):
    model.add(Convolution2D(24,5,5,subsample=(2, 2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,subsample=(2,2),activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def preprocess_image(image):
    #Crop the image
    new_image=image[60:140,:,:]
    #Convert the image to YUV
    new_image=cv2.cvtColor(new_image,cv2.COLOR_RGB2YUV)
    return new_image

#Read all the lines from the csv file
lines = []
with open(r'''./data/driving_log.csv''') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    image_center = cv2.imread('./data/IMG/' + line[0].split('\\')[-1].replace(" ", ""))
    image_left = cv2.imread('./data/IMG/' + line[1].split('\\')[-1].replace(" ", ""))
    image_right = cv2.imread('./data/IMG/' + line[2].split('\\')[-1].replace(" ", ""))
    images.append(preprocess_image(image_center))
    images.append(preprocess_image(image_left))
    images.append(preprocess_image(image_right))
    correction = 0.25
    measurements_center = (float(line[3]))
    measurements_left = (float(line[4]) + correction)
    measurements_right = (float(line[5]) - correction)
    measurements.append(measurements_center)
    measurements.append(measurements_left)
    measurements.append(measurements_right)
#Add images and measurements as x annd y for training set
x_training = np.array(images)
y_training = np.array(measurements)

#Compose the model
model = Sequential()
model.add(Lambda(lambda x:(x / 255) - 0.5, input_shape=(80,320,3)))
model = NVIDIANet(model)
model.compile(loss='mse',optimizer='adam')
history_object=model.fit(x_training,y_training,validation_split=0.2,shuffle=True,epochs=5)
#Save Model
model.save('model.h5')
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
