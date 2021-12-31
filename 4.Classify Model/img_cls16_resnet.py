import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import numpy as np

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






K.clear_session()

n_classes = 4 
img_width, img_height = 299, 299
train_data_dir = 'a_data_16/train'
validation_data_dir = 'a_data_16/valid'
nb_train_samples =total_train # 4730 # 2250 #75750
nb_validation_samples = total_valid #526 # 750 #25250
batch_size =32 

train_datagen = ImageDataGenerator(
    rescale=1. / 255)

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


resnet = ResNet50(include_top=False)
x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(n_classes,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='a_resnet_16_best_model_4class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history.log')

history = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=50,
                    verbose=1,
                    class_weight = cw,
                    callbacks=[csv_logger, checkpointer])

model.save('a_resnet_16_model_trained_class.hdf5')
