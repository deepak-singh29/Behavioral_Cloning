import os
import csv
import numpy as np
import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Lambda,Dropout
from keras.layers.convolutional import Convolution2D
from keras import optimizers

import cv2

def read_csv_lines():
    samples = []
    with open('./driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Infinite Loop generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                try:
                    name = './IMG/'+batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(name)
                    steering_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(steering_angle)
                    #left image augmetation
                    l_name = './IMG/' + batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(l_name)
                    steering_angle = float(batch_sample[3]) + 0.20
                    images.append(left_image)
                    angles.append(steering_angle)
                    # right image augmetation
                    r_name = './IMG/' + batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(r_name)
                    steering_angle = float(batch_sample[3]) - 0.30
                    images.append(right_image)
                    angles.append(steering_angle)
                except ValueError:
                    print(name)

            # trim image to only see section with road
            X_train = np.asfarray(images)
            y_train = np.asfarray(angles)
            #print(X_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)

train_samples, validation_samples = read_csv_lines()
#print(train_samples)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
#x_train,y_train =next(train_generator)
#print(x_train.shape)

#print(next(train_generator))
validation_generator = generator(validation_samples, batch_size=32)
#print(next(validation_generator))

#-----------------------------------------------------------------
# tmp = []
# r_img = cv2.imread('./IMG/center_2016_12_01_13_30_48_287.jpg')
# tmp.append(r_img)
# random_img = np.asfarray(tmp)
# print(random_img.shape)
#-----------------------------------------------------------------
channel, height, width = 3,160,320
# crop_img = Cropping2D(cropping=((50,20), (0,0)), input_shape=(channel, row, col))

def preprocess(x):
    import tensorflow as tf
    #print("Preprocess input shape",x.shape)
    x1 = x/127.5 -1
    channels = tf.unstack(x1, axis=-1)
    x2 = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
    #x3 = tf.image.rgb_to_grayscale(x2)
    x4 = tf.image.resize_images(x2,[66,200])
    #print("Preprocess output shape",x4.shape)
    return x4

# Partially implemented NVIDIA network
def nvidia_model():#drop_rate
    model = Sequential()
    ## Cropping Layer
    ## image output 3x100x320
    model.add(Cropping2D(cropping=((50, 20), (1, 1)), input_shape=(height, width,channel),dim_ordering = 'tf'))
    ## Normalization Layer
    model.add(Lambda(preprocess, input_shape=(100, 318,3),output_shape=(66, 200,3)))#
    ## image output 3x66x200
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu', name='FC1'))
    #model.add(Dropout(drop_rate))
    model.add(Dense(100, activation='relu', name='FC2'))
    #model.add(Dropout(drop_rate))
    model.add(Dense(50, activation='relu', name='FC3'))
    #model.add(Dropout(drop_rate))
    model.add(Dense(10, activation='relu', name='FC4'))
    #model.add(Dropout(drop_rate))
    model.add(Dense(1))
    return model
model1 = nvidia_model() #nvidia_model(0.5)

adm = optimizers.Adam(lr = 0.0009)
model1.compile(loss='mse', optimizer=adm)

model1.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=3)
model1.save('model.h5')