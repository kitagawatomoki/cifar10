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
from tensorflow.keras.utils import plot_model


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
epochs=40
'''one-hotベクトル化'''
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''shape表示'''
print("x_train : ", x_train.shape)
print("y_train : ", y_train.shape)
print("x_test : ", x_test.shape)
print("y_test : ", y_test.shape)

inputs = tf.keras.Input(shape=x_train.shape[1:])
#block1
x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=(1,1), padding='same',name='block1_conv1')(inputs)
x1 = tf.keras.layers.BatchNormalization(name='bn1')(x1)
x1 = tf.keras.layers.Activation('relu')(x1)

x1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name='block1_pool')(x1)

#block2
x2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',name='block2_conv1')(x1)
x2_1 = tf.keras.layers.BatchNormalization(name='bn2')(x2_1)
x2_1 = tf.keras.layers.Activation('relu')(x2_1)
x2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',name='block2_conv2')(x2_1)
x2_1 = tf.keras.layers.BatchNormalization(name='bn3')(x2_1)
x2_1 = tf.keras.layers.Activation('relu')(x2_1)

add1 = tf.keras.layers.Add()([x1,x2_1])

x2_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',name='block2_conv3')(add1)
x2_2 = tf.keras.layers.BatchNormalization(name='bn4')(x2_2)
x2_2 = tf.keras.layers.Activation('relu')(x2_2)
x2_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',name='block2_conv4')(x2_2)
x2_2 = tf.keras.layers.BatchNormalization(name='bn5')(x2_2)
x2_2 = tf.keras.layers.Activation('relu')(x2_2)

add2 = tf.keras.layers.Add()([x2_1,x2_2])

x2_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',name='block2_conv5')(add2)
x2_3 = tf.keras.layers.BatchNormalization(name='bn6')(x2_3)
x2_3 = tf.keras.layers.Activation('relu')(x2_3)
x2_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',name='block2_conv6')(x2_3)
x2_3 = tf.keras.layers.BatchNormalization(name='bn7')(x2_3)
x2_3 = tf.keras.layers.Activation('relu')(x2_3)

add3 = tf.keras.layers.Add()([x2_2,x2_3])
conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same')(x2_3)
#block3
x3_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same',name='block3_conv1')(add3)
x3_1 = tf.keras.layers.BatchNormalization(name='bn8')(x3_1)
x3_1 = tf.keras.layers.Activation('relu')(x3_1)
x3_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',name='block3_conv2')(x3_1)
x3_1 = tf.keras.layers.BatchNormalization(name='bn9')(x3_1)
x3_1 = tf.keras.layers.Activation('relu')(x3_1)

add4 = tf.keras.layers.Add()([conv1,x3_1])

x3_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',name='block3_conv3')(add4)
x3_2 = tf.keras.layers.BatchNormalization(name='bn10')(x3_2)
x3_2 = tf.keras.layers.Activation('relu')(x3_2)
x3_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',name='block3_conv4')(x3_2)
x3_2 = tf.keras.layers.BatchNormalization(name='bn11')(x3_2)
x3_2 = tf.keras.layers.Activation('relu')(x3_2)

add5 = tf.keras.layers.Add()([x3_1,x3_2])

x3_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',name='block3_conv5')(add5)
x3_3 = tf.keras.layers.BatchNormalization(name='bn12')(x3_3)
x3_3 = tf.keras.layers.Activation('relu')(x3_3)
x3_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',name='block3_conv6')(x3_3)
x3_3 = tf.keras.layers.BatchNormalization(name='bn13')(x3_3)
x3_3 = tf.keras.layers.Activation('relu')(x3_3)

add6 = tf.keras.layers.Add()([x3_2,x3_3])

x3_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',name='block3_conv7')(add6)
x3_4 = tf.keras.layers.BatchNormalization(name='bn14')(x3_4)
x3_4 = tf.keras.layers.Activation('relu')(x3_4)
x3_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',name='block3_conv8')(x3_4)
x3_4 = tf.keras.layers.BatchNormalization(name='bn15')(x3_4)
x3_4 = tf.keras.layers.Activation('relu')(x3_4)

add7 = tf.keras.layers.Add()([x3_3,x3_4])
conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='same')(x3_4)
#block4
x4_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='same',name='block4_conv1')(add7)
x4_1 = tf.keras.layers.BatchNormalization(name='bn16')(x4_1)
x4_1 = tf.keras.layers.Activation('relu')(x4_1)
x4_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv2')(x4_1)
x4_1 = tf.keras.layers.BatchNormalization(name='bn17')(x4_1)
x4_1 = tf.keras.layers.Activation('relu')(x4_1)

add8 = tf.keras.layers.Add()([conv2,x4_1])

x4_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv3')(add8)
x4_2 = tf.keras.layers.BatchNormalization(name='bn18')(x4_2)
x4_2 = tf.keras.layers.Activation('relu')(x4_2)
x4_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv4')(x4_2)
x4_2 = tf.keras.layers.BatchNormalization(name='bn19')(x4_2)
x4_2 = tf.keras.layers.Activation('relu')(x4_2)

add9 = tf.keras.layers.Add()([x4_1,x4_2])

x4_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv5')(add9)
x4_3 = tf.keras.layers.BatchNormalization(name='bn20')(x4_3)
x4_3 = tf.keras.layers.Activation('relu')(x4_3)
x4_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv6')(x4_3)
x4_3 = tf.keras.layers.BatchNormalization(name='bn21')(x4_3)
x4_3 = tf.keras.layers.Activation('relu')(x4_3)

add10 = tf.keras.layers.Add()([x4_2,x4_3])

x4_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv7')(add10)
x4_4 = tf.keras.layers.BatchNormalization(name='bn22')(x4_4)
x4_4 = tf.keras.layers.Activation('relu')(x4_4)
x4_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv8')(x4_4)
x4_4 = tf.keras.layers.BatchNormalization(name='bn23')(x4_4)
x4_4 = tf.keras.layers.Activation('relu')(x4_4)

add11 = tf.keras.layers.Add()([x4_3,x4_4])

x4_5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv9')(add11)
x4_5 = tf.keras.layers.BatchNormalization(name='bn24')(x4_5)
x4_5 = tf.keras.layers.Activation('relu')(x4_5)
x4_5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv10')(x4_5)
x4_5 = tf.keras.layers.BatchNormalization(name='bn25')(x4_5)
x4_5 = tf.keras.layers.Activation('relu')(x4_5)

add12 = tf.keras.layers.Add()([x4_4,x4_5])

x4_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv11')(add12)
x4_6 = tf.keras.layers.BatchNormalization(name='bn26')(x4_6)
x4_6 = tf.keras.layers.Activation('relu')(x4_6)
x4_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv12')(x4_6)
x4_6 = tf.keras.layers.BatchNormalization(name='bn27')(x4_6)
x4_6 = tf.keras.layers.Activation('relu')(x4_6)

add13 = tf.keras.layers.Add()([x4_5,x4_6])
conv3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding='same')(x4_6)
#block5
x5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding='same',name='block5_conv1')(add13)
x5_1 = tf.keras.layers.BatchNormalization(name='bn28')(x5_1)
x5_1 = tf.keras.layers.Activation('relu')(x5_1)
x5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block5_conv2')(x5_1)
x5_1 = tf.keras.layers.BatchNormalization(name='bn29')(x5_1)
x5_1 = tf.keras.layers.Activation('relu')(x5_1)

add14 = tf.keras.layers.Add()([conv3,x5_1])

x5_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block5_conv3')(add14)
x5_2 = tf.keras.layers.BatchNormalization(name='bn30')(x5_2)
x5_2 = tf.keras.layers.Activation('relu')(x5_2)
x5_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block5_conv4')(x5_2)
x5_2 = tf.keras.layers.BatchNormalization(name='bn31')(x5_2)
x5_2 = tf.keras.layers.Activation('relu')(x5_2)

add15 = tf.keras.layers.Add()([x5_1,x5_2])

x5_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block5_conv5')(add15)
x5_3 = tf.keras.layers.BatchNormalization(name='bn32')(x5_3)
x5_3 = tf.keras.layers.Activation('relu')(x5_3)
x5_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block5_conv6')(x5_3)
x5_3 = tf.keras.layers.BatchNormalization(name='bn33')(x5_3)
x5_3 = tf.keras.layers.Activation('relu')(x5_3)

add16 = tf.keras.layers.Add()([x5_2,x5_3])

x5_3 = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block5_pool')(add16)

x = tf.keras.layers.Flatten(name='flatten')(x5_3)
x = tf.keras.layers.Dense(units=1000, activation='relu', name='fc1')(x)
outputs = tf.keras.layers.Dense(units=10, activation='softmax', name='outputs')(x)

model_ResNet = tf.keras.Model(inputs, outputs, name='model_ReaNet')
model_ResNet.summary()

'''optimizer定義'''
optimizer=keras.optimizers.Adam()
model_ResNet.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
'''データ正規化'''
x_train=x_train.astype('float32')
x_train/=255
x_test=x_test.astype('float32')
x_test/=255
'''fit'''
history=model_ResNet.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

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

fig_acc.savefig("ResNet01_acc_ep40.jpg")
fig_loss.savefig("ResNet01_loss_ep40.jpg")
