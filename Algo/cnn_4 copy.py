
import os
from PIL import Image
from PIL import ImageFilter
import numpy as np
import keras
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D,GlobalAveragePooling2D
from keras.models import Sequential


num_classes = 2
epochs = 50
batch_size = 4
n_filters = 32
n_pool = 2
n_conv = 2
img_rows, img_cols = 32, 32

os.chdir("/Users/maryameshraghievari/desktop/imgs1")
labels = []

file = "imgs1_labels.txt"
lbl = np.genfromtxt(file)
for i in range(0, len(lbl)):
    if lbl[i] == 0 :
        labels.append( -1)
    elif lbl[i] == 1 :
        labels.append( 1)
    else:
        continue

#print(labels)

path = "/Users/maryameshraghievari/desktop/imgs1"
valid_images = [".png"]
names = os.listdir(path)


data = []
for name in names:
    ext = os.path.splitext(name)[1]
    if ext.lower() not in valid_images:
        continue
    img = Image.open(name)
    img = img.resize((img_rows, img_cols))
    img = np.array(img)[np.newaxis, :, :, :3]
    data.append(img)

data = np.concatenate(data)

n = 90
#n = 900
x_train = data[:n].astype(np.float32)
y_train = labels[:n]
x_test = data[n:].astype(np.float32)
y_test = labels[n:]
x_train /= 255
x_test /= 255

#print (x_train)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', input_shape=[32,32,3]))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid', input_shape=[32,32,3])) #Convolution
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid'))
model.add(Activation('relu'))
#model.add(GlobalAveragePooling2D())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))


opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


def train():
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

train()




