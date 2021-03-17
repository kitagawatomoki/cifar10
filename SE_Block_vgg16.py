'''インポート'''
import numpy as np
import sys
#%matplotlib inline
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import image # keras.preprocessing.image APIを利用する。画像拡張用の関数が用意されている。
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras import backend as K

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
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

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, GlobalAveragePooling2D, Dense, Multiply, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, MobileNet
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import History

from keras.activations import linear
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import os, pickle, zipfile, glob

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

#tensorboard コールバック関数の設定
tb_cd = TensorBoard(log_dir='/misc/Work20/kitagawatomoki/attention/atteniton_logs', histogram_freq=1)
#DataAugmentation処理
def dataaugmentation(dataset, labelset):
    datagen =  ImageDataGenerator(rotation_range = 20, horizontal_flip = True, height_shift_range = 0.2,width_shift_range = 0.2,zoom_range = 0.2, channel_shift_range = 0.2)
    agdata =[]
    label = []
    count = 0
    for x in dataset:
        x = x[np.newaxis,:]
        for d in datagen.flow(x, batch_size=1):
            agdata.append(d[0, :])
            label.append(labelset[count])
            if(len(agdata) % 4) == 0:
                count+=1
                break
    
    dataset = np.concatenate([dataset, agdata])
    labelset = np.concatenate([labelset, label])
    return dataset, labelset
'''データセットの読み込み'''
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, y_train = dataaugmentation(x_train, y_train)


'''バッチサイズ、クラス数、エポック数の設定'''
batch_size=256
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





def se_block(input, channels, r=8):
    # Squeeze
    x = GlobalAveragePooling2D()(input)
    # Excitation
    x = Dense(channels//r, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)
    return Multiply()([input, x])
#########################モデル定義#######################################
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
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',name='block2_conv1')(se_block(x, 64))
x = tf.keras.layers.BatchNormalization(name='bn3')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',name='block2_conv2')(x)
x = tf.keras.layers.BatchNormalization(name='bn4')(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name='block2_pool')(x)

x = tf.keras.layers.Dropout(0.2)(x)

#block3
x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name='block3_conv1')(se_block(x, 128))
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
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block4_conv1')(se_block(x, 256))
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
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',name='block5_conv1')(se_block(x, 512))
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
#x = tf.keras.layers.Dense(units=512, activation='relu', name='fc1')(x)
#x = tf.keras.layers.Dropout(0.2)(x)
#x = tf.keras.layers.Dense(units=512, activation='relu', name='fc2')(x)
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

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) # 訓練セットの入力と教師ラベル合体させてTensorに変換
valid_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) # 検証も同様に処理
AUTOTUNE = tf.data.experimental.AUTOTUNE # 処理を最適化するためのおまじない（自動チューニング設定）
train_ds = train_ds.shuffle(len(x_train)) # 訓練データをシャッフルする。引数にはデータ数を指定すると完全なシャッフルが行われる。len(x_train)は60000。
train_ds = train_ds.repeat(1) # 1 epochで使われるデータの回数。1の場合，1epochで1回しか使われない。引数を空欄にすると無限に使われる。
train_ds = train_ds.batch(batch_size) # ミニバッチを作る。1バッチ32個のデータ。
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) # 訓練中に次のバッチを取り出すための処理。
valid_ds = valid_ds.batch(batch_size) # 検証データはシャッフルする必要ないので，バッチ化のみの処理でOK
'''fit'''
history=model_vgg16.fit(train_ds, epochs=epochs, validation_data=(valid_ds), callbacks=[tb_cd])

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

fig_acc.savefig("SE_Block03_acc_ep60.jpg")
fig_loss.savefig("SE_Block03_loss_ep60.jpg")