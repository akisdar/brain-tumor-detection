from IPython import display
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import *
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from tensorflow import keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten , Dense, RandomRotation,RandomFlip




dataset_1 = 'resized_dataset/'

dataset_2 = 'brain_tumor_2/'


image_directory = dataset_1


SIZE = 224


no_tumor = os.listdir(image_directory +'/no/')
yes_tumor = os.listdir(image_directory +'/yes/')
dataset=[]
label= []






for i, image_name in enumerate(no_tumor):
    
    if (image_name.split('.')[1]== 'jpg' or 'jpeg'):
        image = cv2.imread(image_directory +'no/' + image_name)
        image= Image.fromarray(image,'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))
        label.append(0)


for i, image_name in enumerate(yes_tumor):
    if (image_name.split('.')[1]== 'jpg' or 'jpeg'):
        image = cv2.imread(image_directory +'yes/' + image_name)
        image= Image.fromarray(image,'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))
        label.append(1)        


dataset = np.array(dataset)
labels = np.array(label)


x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2,random_state=0,shuffle=True)




x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



                ## MODEL **VGG19**


tf.keras.backend.clear_session()

base_model = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False #if its TRUE make the adam optimizer smaller




model_vgg19 = tf.keras.Sequential()
model_vgg19.add(tf.keras.layers.RandomFlip(mode='horizontal_and_vertical'))
model_vgg19.add(tf.keras.layers.RandomRotation((0.2),fill_mode='constant',fill_value=0))
model_vgg19.add(base_model)
base_model.trainable = False
model_vgg19.add(tf.keras.layers.Flatten())

model_vgg19.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model_vgg19.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['acc'])



history = model_vgg19.fit(x_train,y_train,validation_data= (x_test,y_test)  ,batch_size=16, epochs=30, shuffle=True, verbose=True)
model_vgg19.summary()
model_vgg19.save('model_vgg19.h5')

                     ## ACCURACY AND LOSS PLOTS



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

train_result = model_vgg19.evaluate(x_train,y_train)
val_result = model_vgg19.evaluate(x_test,y_train)



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

                      ### Image Pre-processing

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


base_model.trainable = True
model_inceptionv3 = tf.keras.Sequential()

                         ###  Real time Data Augmentation

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

