'''インポート'''
import numpy as np
import sys
#matplotlib inline
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras import backend as K

import tensorflow as tf
import glob
import imageio
import numpy as np
import os
from tensorflow.keras import layers
from tensorflow.keras import metrics
import time
import cv2 
from tensorflow.keras.utils import plot_model

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.applications import VGG16

#GPUメモリ使用制限
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

'''データセットの読み込み'''
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
'''バッチサイズ、クラス数、エポック数の設定'''
batch_size=64
num_classes=10
epochs=60
'''one-hotベクトル化'''
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''shape表示'''
print("x_train : ", x_train.shape)
print("y_train : ", y_train.shape)
print("x_test : ", x_test.shape)
print("y_test : ", y_test.shape)

def set_seed(seed=200):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    #np.random.seed(seed)
    # for built-in random
    #random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=x_train.shape[1:])
vgg16_model.summary()

#inputs = tf.keras.Input(shape=vgg16_model.output_shape[1:])
#x = tf.keras.layers.Flatten(name='flatten')(inputs)
#x = tf.keras.layers.Dense(units=192, activation='relu', name='dense1')(x)
#x = tf.keras.layers.Dense(units=224, activation='relu', name='dense2')(x)
#x = tf.keras.layers.Dense(units=96, activation='relu', name='dense3')(x)
#x = tf.keras.layers.Dense(units=32, activation='relu', name='dense4')(x)
#outputs = tf.keras.layers.Dense(units=10, activation='softmax', name='outputs')(x)

#top_model = tf.keras.Model(inputs, outputs, name='top_model')
set_seed(0)
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(layers.Dense(units=192, activation='relu'))
top_model.add(layers.Dense(units=224, activation='relu'))
top_model.add(layers.Dropout(0.5))
top_model.add(layers.Dense(units=96, activation='relu'))
top_model.add(layers.Dense(units=32, activation='relu'))
#top_model.add(layers.Dense(units=224, activation='relu'))
#top_model.add(layers.Dense(units=64, activation='relu'))
#top_model.add(layers.Dense(units=96, activation='relu'))
#top_model.add(layers.Dense(units=384, activation='relu'))
top_model.add(layers.Dense(units=10, activation='softmax'))
top_model.summary()

model = Model(inputs=vgg16_model.input, outputs=top_model(vgg16_model.output))
model.summary()

for layer in model.layers[:15]:
    layer.trainable = False
    print(layer)

#opt=keras.optimizers.Adam(0.0001)
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test))

'''結果の可視化'''
fig_acc = plt.figure(figsize=(10,7))
plt.plot(history.history['accuracy'], color='b', linewidth=3)
plt.plot(history.history['val_accuracy'], color='r', linewidth=3)
plt.tick_params(labelsize=18)
plt.ylabel('accuracy', fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.legend(['training', 'test'], loc='best', fontsize=20)
fig_loss= plt.figure(figsize=(10,7))
plt.plot(history.history['loss'], color='b', linewidth=3)
plt.plot(history.history['val_loss'], color='r', linewidth=3)
plt.tick_params(labelsize=18)
plt.ylabel('loss', fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.legend(['training', 'test'], loc='best', fontsize=20)
plt.show()

fig_acc.savefig("Transfer00_acc_ep30.jpg")
fig_loss.savefig("Transfer00_loss_ep30.jpg")