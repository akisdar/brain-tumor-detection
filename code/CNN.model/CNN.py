import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation,Dropout, Flatten , Dense,RandomFlip,RandomRotation

import tensorflow as tf


image_directory = 'resized_dataset/'
SIZE = 128

no_tumor = os.listdir(image_directory+ 'no/')
yes_tumor = os.listdir(image_directory+ 'yes/')
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


x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.25,random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

### MODEL 

model = Sequential()

#model.add(RandomFlip(mode='horizontal_and_vertical'))
#model.add(RandomRotation(factor=0.2,fill_value=0))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer= "he_uniform"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), kernel_initializer= "he_uniform"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['Accuracy'])
history = model.fit(x_train,y_train,batch_size=4,verbose=1,epochs=30,validation_data= (x_test,y_test), shuffle=True )
model.summary()



#plt.imshow(x_test[23])
#plt.show()
#print(model(x_test[23].reshape(1, 64, 64, 3)), y_test[23])
model.save('CNN_model.h5')


fig, axs =plt.subplots(1,2)
axs[0].plot(history.history['Accuracy'])
axs[0].plot(history.history['val_Accuracy'])
axs[0].set_title('model accuracy')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')



axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('model loss')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
fig.legend(['train', 'test'], loc='lower right')
plt.show()















print(len(dataset))
print(len(label))

#Print the 25 images
fig, axs = plt.subplots(5, 5, figsize=(12, 14), tight_layout=True)
rows = 5
cols = 5
ind = 0
for r in range(rows):
    for c in range(cols):
        axs[r, c].imshow(dataset[ind])
        axs[r, c].set_title('{}'.format(label[ind]))
        axs[r, c].set_xticks([])
        axs[r, c].set_yticks([])
        ind += 1
#plt.show()