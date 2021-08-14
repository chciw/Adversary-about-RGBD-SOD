# a few changes are needed if used for model_rosa
import os
import cv2
import tensorflow as tf
import keras
from keras import backend as K
import h5py
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from model import *
import numpy as np

def load():
    f = h5py.File('./train_data/merge2.h5','r')
    f.keys()
    x = f['x'][:]
    y = f['y'][:]
    z = f['z'][:]
    # x2=f['x2'][:]
    val_x = f['x_val'][:]
    val_y = f['y_val'][:]
    val_z = f['z_val'][:]
    # val_x2 = f['X_val_2'][:]
    f.close()
    # return x, y,z,x2,val_x,val_y,val_z,val_x2
    return x, y,z,val_x,val_y,val_z

def deal_score(score):
    for i in range(len(score)):
        score[i]=1-score[i]
    return score

# dimensions of our images.
img_width,  img_height =  224,224

epochs = 20
batch_size = 8

model = vgg16_deep_fuse_model(img_width,img_height)
# model = vgg16_deep_model(img_width,img_height)
# model = get_test_model(img_width,img_height)
#exit()

model_checkpoint = ModelCheckpoint('./checkpoints/vgg16_deep_fuse_512.{val_loss:.3f}.hdf5', monitor='loss',verbose=1, save_weights_only=False,period=10,save_best_only=False)

train_continue = 1
if train_continue:
    model.load_weights('./checkpoints/vgg16_deep_fuse_512.0.152.hdf5',by_name=True)

mode_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=1, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.0001)

#images,masks ,deep1,biImg,val_images,val_y,deep2,val_biImg= load()
images,masks ,deep1,val_images,val_y,deep2= load()

model.fit([images,deep1],[masks],batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([val_images,deep2],[val_y]),callbacks=[model_checkpoint,mode_lr])
#model.fit([images,deep1,biImg],[masks],batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([val_images,deep2,val_biImg],[val_y]),callbacks=[model_checkpoint,mode_lr])
#model.fit([image,deep],masks,batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2,callbacks=[model_checkpoint,mode_lr])

