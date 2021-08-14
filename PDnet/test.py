import os
import cv2
import tensorflow as tf
import keras
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator,array_to_img
from keras.models import *
from keras.layers import *
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import h5py
from attack import getInput
from model import *
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np

rootdir = './dataset/LFSD/Data/'
img_width, img_height=224,224
model = vgg16_deep_fuse_model(img_width, img_height)
model.load_weights('./checkpoints/vgg16_deep_fuse_512.0.152.hdf5', by_name=True)
# filename='1500'

for root, dirs, files in os.walk(rootdir+'Img/'):
        for file in files:
            print(file)
            filename = os.path.splitext(file)[0]
            if os.path.splitext(file)[1]=='.txt':
                    continue
            img,gt,deep,biImg=getInput(rootdir+'Img/'+filename+'.jpg',rootdir+'GT/'+filename+'.png',rootdir+'deep/'+filename+'.bmp',img_width,img_height)
            probs=model.predict([img,deep])[0] # (1,480, 640, 1/2)->B,H,W,C
            # print(probs)
            # cv2.imshow('u p',probs)
            # cv2.waitKey(0)
                
            # for model_rosa
            # labels = probs.argmax(axis=2).astype('uint8')[:img_height, :img_width]
            # label_im = Image.fromarray(labels, 'P')
            # _PALETTE = [0,0,0,255,255,255]
            # label_im.putpalette(_PALETTE)
            # label_im.save(rootdir+'SWSPred/'+filename+'.png')
            # for model    
            test = array_to_img(probs)
            test.save('./results/'+filename+'.png')
