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
# from u_net_test import get_enhanced_score

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
            probs=model.predict([img,deep])[0] # (1,480, 640, 1)->B,H,W,C
            # print(probs)
            # cv2.imshow('u p',probs)
            # cv2.waitKey(0)

            # labels = probs.argmax(axis=2).astype('uint8')[:img_height, :img_width]
            # label_im = Image.fromarray(labels, 'P')
            # _PALETTE = [0,0,0,255,255,255]
            # label_im.putpalette(_PALETTE)
            # label_im.save(rootdir+'SWSPred/'+filename+'.png')
            test = array_to_img(probs)
            test.save('./results/'+filename+'.png')
'''
def load():
    f = h5py.File('./bbs_data/stere.h5', 'r')
    f.keys()
    X = f['x_val'][:]
    z= f['z_val'][:]
    f.close()
    return X,z

images = load()
image = images[0]
print (image.shape)
deep = images[1]
print (deep.shape)

# dimensions of our images.
img_width,  img_height = 224,224 
#mask_width, mask_height = 120, 120

################################################################################
#TN=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

#model = get_model(img_width,img_height)
model = vgg16_deep_fuse_model(img_width,img_height)
# model = vgg161_model(img_width,img_height)

model.load_weights('checkpoints/vgg16_deep_fuse_512.0.209.hdf5')
#model.load_weights('checkpoints/msra_96x96_weight.0.19.hdf5',by_name=False)
#model.load_weights('checkpoints/new_fine_msra_96x96_weight.0.165.hdf5',by_name=False)
#model.load_weights('checkpoints/new2_vgg_msra_192x192x2_weight.0.184.hdf5',by_name=False)


# score=get_enhanced_score(deep)
path='./results/'
if not os.path.exists(path):
    os.makedirs(path)
img_pre=model.predict([image,deep],batch_size=1, verbose=1)
for i in range(img_pre.shape[0]):
    #if i>200:
        #break
    img = img_pre[i]
    img = array_to_img(img)
    img.save(path+"/%d.png"%(1+i))
'''



