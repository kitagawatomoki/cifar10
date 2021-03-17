import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow.keras import metrics
import time
from IPython import display
import cv2 
from PIL import Image

'''インポート'''
import numpy as np
import sys
#%matplotlib inline
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras import backend as K

#GPUメモリ使用制限
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


def set_seed(seed=4):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    #np.random.seed(seed)
    # for built-in random
    #random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

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

set_seed(0)
inputs = tf.keras.Input(shape=x_train.shape[1:])
#block1
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',name='block1_conv1')(inputs)
x = tf.keras.layers.BatchNormalization(name='bn1')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',name='block1_conv2')(x)
x = tf.keras.layers.BatchNormalization(name='bn2')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name='block1_pool')(x)

x = tf.keras.layers.Dropout(0.2)(x)
#block2
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',name='block2_conv1')(x)
x = tf.keras.layers.BatchNormalization(name='bn3')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',name='block2_conv2')(x)
x = tf.keras.layers.BatchNormalization(name='bn4')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name='block2_pool')(x)

x = tf.keras.layers.Dropout(0.2)(x)
#block3
x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block3_conv1')(x)
x = tf.keras.layers.BatchNormalization(name='bn5')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block3_conv2')(x)
x = tf.keras.layers.BatchNormalization(name='bn6')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block3_conv3')(x)
x = tf.keras.layers.BatchNormalization(name='bn7')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name='block3_pool')(x)

x = tf.keras.layers.Dropout(0.2)(x)
#block4
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv1')(x)
x = tf.keras.layers.BatchNormalization(name='bn8')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv2')(x)
x = tf.keras.layers.BatchNormalization(name='bn9')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv3')(x)
x = tf.keras.layers.BatchNormalization(name='bn10')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name='block4_pool')(x)

x = tf.keras.layers.Dropout(0.2)(x)
#block5
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block5_conv1')(x)
x = tf.keras.layers.BatchNormalization(name='bn11')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block5_conv2')(x)
x = tf.keras.layers.BatchNormalization(name='bn12')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block5_conv3')(x)
x = tf.keras.layers.BatchNormalization(name='bn13')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name='block5_pool')(x)

x = tf.keras.layers.Flatten(name='flatten')(x)
x = tf.keras.layers.Dense(units=4096, activation='relu', name='fc1')(x)
x = tf.keras.layers.Dense(units=4096, activation='relu', name='fc2')(x)
outputs = tf.keras.layers.Dense(units=10, activation='softmax', name='outputs')(x)

model_vgg16 = tf.keras.Model(inputs, outputs, name='model_vgg16')
model_vgg16.summary()

'''optimizer定義'''
optimizer=keras.optimizers.Adam()
#optimizer=tf.keras.optimizers.Adam(1e-4)

model_vgg16.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
'''データ正規化'''
x_train=x_train.astype('float32')
x_train/=255
x_test=x_test.astype('float32')
x_test/=255
'''fit'''
history=model_vgg16.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

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

fig_acc.savefig("vgg16_acc_ep60.jpg")
fig_loss.savefig("vgg16_loss_ep60.jpg")