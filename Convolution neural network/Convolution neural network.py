from __future__ import print_function

import os
import keras
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import random as rn
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, show
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
from glob import glob
from os.path import splitext
import pandas as pd
import cv2
np.random.seed(1337)  # for reproducibility
rn.seed(1337)
from keras.models import Model
from keras.models import Sequential
from keras.layers import GaussianDropout,Embedding,ZeroPadding2D,AveragePooling2D
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Dense, Flatten, UpSampling2D, core, Convolution2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.callbacks import TensorBoard
from keras.preprocessing import image
import csv
from skimage.filters import unsharp_mask
from skimage.restoration import denoise_bilateral
from skimage.restoration import denoise_wavelet
from keras.layers import BatchNormalization,Activation, regularizers
from skimage.filters import roberts, sobel
from skimage import segmentation
from scipy.signal import wiener
from keras.preprocessing import image
from skimage.filters import sobel_v
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration
from keras.layers import Input
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, add, GlobalAvgPool2D
from skimage import color, data, restoration
from skimage import exposure
from cv2 import createCLAHE
import time

# CNN(train/test)
AStart = time.time()
K.set_image_data_format('channels_last')

smooth = 1.

datachannel = 1
im_W = 100
im_H = 100
SavePath = r"C:\Users\Desktop\_/"
savepath_train = r"C:\Users\Desktop\_/"
savepath_test = r"C:\Users\Desktop\_/"
train_savepath = os.listdir(savepath_train)
test_savepath = os.listdir(savepath_test))

x_train = np.empty((len(train_savepath), im_W, im_H, datachannel), dtype="uint8")
v_train = [0] * len(train_savepath)
for i in range(len(train_savepath)):
    if (int(train_savepath[i][0]) == 0):
        v_train[i] = 0
    if ((int(train_savepath[i][0]) == 1)):
        v_train[i] = 1
    img_train = Image.open(savepath_train + train_savepath[i])
    npimg_train = np.asarray(img_train, dtype="uint8")
    # img_adapteq_train = exposure.equalize_adapthist(npimg_train, clip_limit=0.01)
    # X_unsharp_train=unsharp_mask(img_adapteq_train,radius=3,amount=2)
    im_train = misc.imresize(npimg_train, (im_W, im_H))
    rsimg_train = im_train.reshape(im_W, im_H, datachannel)
    x_train[i, :, :, :] = rsimg_train

x_test = np.empty((len(test_savepath), im_W, im_H, datachannel), dtype="uint8")
v_test = [0] * len(test_savepath)
for il in range(len(test_savepath)):
    if (int(test_savepath[il][0]) == 0):
        v_test[il] = 0
    if ((int(test_savepath[il][0]) == 1)):
        v_test[il] = 1
    img_test = Image.open(savepath_test + test_savepath[il])
    npimg_test = np.asarray(img_test, dtype="uint8")
    # img_adapteq_test = exposure.equalize_adapthist(npimg_test, clip_limit=0.01)
    # X_unsharp_test=unsharp_mask(img_adapteq_test,radius=3,amount=2)
    im_test = misc.imresize(npimg_test, (im_W, im_H))
    rsimg_test = im_test.reshape(im_W, im_H, datachannel)
    x_test[il, :, :, :] = rsimg_test

AEnd = time.time()  # 計時結束
print("It cost %f sec" % (AEnd - AStart))
print(AEnd - AStart)  # 原型長這樣

x_train_flo = x_train.astype('float32')
x_test_flo = x_test.astype('float32')
x_Train=(x_train_flo/255)
x_Test=(x_test_flo/255)
y_Train = pd.get_dummies(pd.Series(v_train)).values.astype('float32')
y_Test = pd.get_dummies(pd.Series(v_test)).values.astype('float32')


SVGG_Net = Sequential()
SVGG_Net.add(Conv2D(32,(3,3))
SVGG_Net.add(Conv2D(32,(3,3))
SVGG_Net.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
SVGG_Net.add(Conv2D(64,(3,3))
SVGG_Net.add(Conv2D(64,(3,3))
SVGG_Net.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
SVGG_Net.add(Conv2D(128,(3,3))
SVGG_Net.add(Conv2D(128,(3,3))
SVGG_Net.add(Conv2D(128,(3,3))
SVGG_Net.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
SVGG_Net.add(Conv2D(128,(3,3))
SVGG_Net.add(Conv2D(128,(3,3))
SVGG_Net.add(Conv2D(128,(3,3))
SVGG_Net.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
SVGG_Net.add(Conv2D(128,(3,3))
SVGG_Net.add(Conv2D(128,(3,3))
SVGG_Net.add(Conv2D(128,(3,3))
SVGG_Net.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
SVGG_Net.add(Flatten())
SVGG_Net.add(Dense(1000,activation='relu'))
SVGG_Net.add(Dense(2,activation='softmax'))
#SVGG_Net.summary()


tStart = time.time()
if __name__ == '__main__':
    model = SVGG_Net()
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
filepath = SavePath + "weights-seg-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
#learning_rate_function=ReduceLROnPlateau(monitor='val_acc', patience=10, verbose=1, factor=0.001, min_lr=0.48)
Show_result=Alexnet.fit(x_Train,y_Train,
            validation_data=(x_Test, y_Test),
            batch_size=30,
            epochs=60,
            callbacks=[checkpoint],
            verbose=1)
tEnd = time.time()#計時結束
print ("It cost %f sec" % (tEnd - tStart))#會自動做近位
print (tEnd - tStart)#原型長這樣