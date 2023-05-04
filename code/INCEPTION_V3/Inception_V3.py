from IPython import display
#%matplotlib notebook
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import *
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import RandomFlip
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

np.random.seed(42)
tf.random.set_seed(42)


height, width = 299,299
batch_size=64



original_dataset = 'resized_dataset/'

                           ### Image pre-processing

def image_generator(height,width):
    datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.2,)


    train_ds = datagen.flow_from_directory(original_dataset,
            batch_size=batch_size,subset="training",
            #color_mode = 'grayscale',
            shuffle=True,class_mode='binary',
            target_size=(height, width),
            classes={'no': 0., 'yes': 1.})
    val_ds = datagen.flow_from_directory(original_dataset,subset="validation",
            #seed=123,#color_mode = 'grayscale',
            class_mode='binary',
            target_size=(height, width),
            batch_size=batch_size,
            classes={'no': 0., 'yes': 1.} )
    return train_ds, val_ds

    
train_ds, val_ds = image_generator(height,width)






                                   ## inception v3


tf.keras.backend.clear_session()
input_shape = (height, width, 3)
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)


base_model.trainable = True # when its TRUE we decrease the learning rate
model_inceptionv3 = tf.keras.Sequential()


                  ###   Real time Data Augmentation
model_inceptionv3.add(tf.keras.layers.RandomFlip(mode= 'horizontal_and_vertical'))
model_inceptionv3.add(tf.keras.layers.RandomRotation((0.1),fill_mode='constant',fill_value=0))


model_inceptionv3.add(base_model)

model_inceptionv3.add(tf.keras.layers.GlobalAveragePooling2D())
model_inceptionv3.add(tf.keras.layers.Dense(128, activation = 'relu'))
model_inceptionv3.add(tf.keras.layers.BatchNormalization())
model_inceptionv3.add(tf.keras.layers.Dropout(0.2))


model_inceptionv3.add(tf.keras.layers.Flatten())
model_inceptionv3.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model_inceptionv3.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),metrics=['acc'])


from pickle import TRUE
from tensorflow._api.v2.random import shuffle
history = model_inceptionv3.fit(train_ds,validation_data=(val_ds),batch_size=16,epochs=30,shuffle=TRUE, verbose=True)



model_inceptionv3.summary()
model_inceptionv3.save('inceptionV3_dataset1.h5')


fig, axs =plt.subplots(1,2)
axs[0].plot(history.history['acc'])
axs[0].plot(history.history['val_acc'])
axs[0].set_title('model acc')
axs[0].set_ylabel('acc')
axs[0].set_xlabel('epoch')



axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('model loss')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
fig.legend(['train', 'test'], loc='lower right')
plt.show()

train_result = model_inceptionv3.evaluate(train_ds)
val_result = model_inceptionv3.evaluate(val_ds)