import glob
from sklearn.model_selection import train_test_split
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import *
from efficientnet.tfkeras import EfficientNetB4
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import utils
from keras import layers
import efficientnet.keras as efn
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


import sys
import pandas as pd
import os
import glob

#"/home/joyhyuk/final_hyuk/a_data_16/train"
img_list_0 = glob.glob("./a_data_16/train/0/*.jpg")
img_list_1 = glob.glob("./a_data_16/train/1/*.jpg")
img_list_2 = glob.glob("./a_data_16/train/2/*.jpg")
img_list_3 = glob.glob("./a_data_16/train/3/*.jpg")

cnt_0 = len(img_list_0)
cnt_1 = len(img_list_1)
cnt_2 = len(img_list_2)
cnt_3 = len(img_list_3)

vimg_list_0 = glob.glob("./a_data_16/valid/0/*.jpg")
vimg_list_1 = glob.glob("./a_data_16/valid/1/*.jpg")
vimg_list_2 = glob.glob("./a_data_16/valid/2/*.jpg")
vimg_list_3 = glob.glob("./a_data_16/valid/3/*.jpg")

vcnt_0 = len(vimg_list_0)
vcnt_1 = len(vimg_list_1)
vcnt_2 = len(vimg_list_2)
vcnt_3 = len(vimg_list_3)

total_train = cnt_0+cnt_1+cnt_2+cnt_3
total_valid = vcnt_0 + vcnt_1 + vcnt_2 + vcnt_3
#wj=n_samples / (n_classes * n_samplesj)
cw = {}
cw[0] = total_train/(4*cnt_0)
cw[1] = total_train/(4*cnt_1)
cw[2] = total_train/(4*cnt_2)
cw[3] = total_train/(4*cnt_3)

print("CW >", cw)
print("total train >", total_train)
print("total valid >", total_valid)

"""
위암, 

EfficientNetB4로 분류하는 코드 입니다. 
"""

n_classes = 4 
img_width, img_height = 299, 299
train_data_dir = 'a_data_16/train' # train에 해당된 폴더 경로 지정
validation_data_dir = 'a_data_16/valid' # valid에 해당된 폴더 경로 지정
nb_train_samples = total_train # train에 저장된 파일 개수
nb_validation_samples =total_valid # valid에 저장된 파일 개수
batch_size =16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model = Sequential()
model.add(efn.EfficientNetB4(weights="imagenet", include_top=False, pooling='avg'))
model.add(layers.Dense(n_classes, activation="softmax"))
model = utils.multi_gpu_model(model, gpus=4)
model.compile(metrics=['acc'], loss='categorical_crossentropy', optimizer='adam')

checkpointer = ModelCheckpoint(filepath='eff_c16_best_model_4class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history.log')

history = model.fit(train_generator, validation_data=validation_generator, epochs=150, callbacks=[csv_logger, checkpointer])

model.save('eff_c16_model_trained_4class.hdf5')
