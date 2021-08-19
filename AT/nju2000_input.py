"""
Utilities for importing the nju2000 dataset.

Each image in the dataset is a numpy array of shape (224, 224, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2

version = sys.version_info

class nju2000Data(object):
    """
    Inputs to constructor
    =====================
        The training data containing 1485 examples, the test data
        containing 500 examples 
    """
    def __init__(self):
        root='D:/PycharmProject/PDNet_available/dataset/nju2000'
        train_img_path =root+'/train/Img/'
        train_deep_path=root+'/train/deep/'
        train_gt_path=root+'/train/GT/'
        test_img_path =root+'/test/Img/'
        test_deep_path=root+'/test/deep/'
        test_gt_path=root+'/test/GT/'

        size=224
        train_img = np.zeros((1500, size, size, 3), dtype='float32')
        train_deep=np.zeros((1500,size,size,1), dtype='uint8')
        train_labels = np.zeros((1500,size,size,1), dtype='uint8')
        train_names=[]
        test_img = np.zeros((485, size, size, 3), dtype='float32')
        test_deep=np.zeros((485,size,size,1), dtype='uint8')
        test_labels = np.zeros((485,size,size,1), dtype='uint8')
        test_names=[]

        i=0
        for root, dirs, files in os.walk(train_img_path):
            for file in files:
                filename = os.path.splitext(file)[0]
                if  os.path.splitext(file)[1]=='.txt':
                    continue
                train_img[i,:,:,:]=cv2.resize(cv2.imread(train_img_path+file),(size,size),interpolation=cv2.INTER_CUBIC)
                train_deep[i,:,:,0]=cv2.resize(cv2.imread(train_deep_path+filename+'.png',0),(size,size),interpolation=cv2.INTER_CUBIC)
                train_labels[i,:,:,0]=cv2.resize(cv2.imread(train_gt_path+filename+'.png',0),(size,size),interpolation=cv2.INTER_CUBIC)
                train_names.append(filename)
                i=i+1
            print(i)
        i=0
        for root,dirs,files in os.walk(test_img_path):
            for file in files:
                filename = os.path.splitext(file)[0]
                if os.path.splitext(file)[1] == '.txt':
                    continue
                test_img[i,:,:,:]=cv2.resize(cv2.imread(test_img_path+file),(size,size),interpolation=cv2.INTER_CUBIC)
                test_deep[i,:,:,0]=cv2.resize(cv2.imread(test_deep_path+filename+'.png',0),(size,size),interpolation=cv2.INTER_CUBIC)
                test_labels[i,:,:,0]=cv2.resize(cv2.imread(test_gt_path+filename+'.png',0),(size,size),interpolation=cv2.INTER_CUBIC)
                test_names.append(filename)
                i=i+1
        print(i)

        self.train_data = DataSubset(train_img, train_deep,train_labels,train_names)
        self.eval_data = DataSubset(test_img,test_deep, test_labels,test_names)

class DataSubset(object):
    def __init__(self, xs, ds,ys,names):
        self.xs = xs
        self.n = xs.shape[0]
        self.ds=ds
        self.ys = ys
        self.names=names
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ds = self.ds[self.cur_order[self.batch_start: batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            batch_ns = self.names[self.cur_order[self.batch_start: batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ds,batch_ys,batch_ns
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ds = self.ds[self.cur_order[self.batch_start: batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        batch_ns = self.names[self.cur_order[self.batch_start: batch_end], ...]
        self.batch_start += batch_size
        return batch_xs, batch_ds,batch_ys,batch_ns
