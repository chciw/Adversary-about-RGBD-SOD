"""
Utilities for importing the NJUD dataset.

Each image in the dataset is a numpy array of shape (256, 256, 3), with the values
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

class NJUDData(object):
    """
    Inputs to constructor
    =====================
        The training data containing 1485 examples, the test data
        containing 500 examples 
    """
    def __init__(self, path):
        root='/home/jackice/PycharmProjects/pythonProject1'
        train_img_path =root+'/dataset/NJUD/train/train_images/'
        train_deep_path=root+'/dataset/NJUD/train/train_depth/'
        train_gt_path=root+'/dataset/NJUD/train/train_masks/'
        test_img_path =root+'/dataset/NJUD/test/Data/Img/'
        test_deep_path=root+'/dataset/NJUD/test/Data/deep/'
        test_gt_path=root+'/dataset/NJUD/test/Data/GT/'

        train_img = np.zeros((1485, 256, 256, 3), dtype='uint8')
        train_deep=np.zeros((1485,256,256,1), dtype='uint8')
        train_labels = np.zeros((1485,256,256,1), dtype='uint8')
        train_names=[]
        test_img = np.zeros((500, 256, 256, 3), dtype='uint8')
        test_deep=np.zeros((500,256,256,1), dtype='uint8')
        test_labels = np.zeros((500,256,256,1), dtype='uint8')
        test_names=[]

        i=0
        for root, dirs, files in os.walk(train_img_path):
            for file in files:
                filename = os.path.splitext(file)[0]
                if  os.path.splitext(file)[1]=='.txt':
                    continue
                train_img[i,:,:,:]=cv2.imread(train_img_path+file)
                train_deep[i,:,:,0]=cv2.imread(train_deep_path+filename+'.jpg',0)
                train_labels[i,:,:,0]=cv2.imread(train_gt_path+filename+'.png',0)
                train_names.append(filename)
                i=i+1
            print(i)
        i=0
        for root,dirs,files in os.walk(test_img_path):
            for file in files:
                filename = os.path.splitext(file)[0]
                if os.path.splitext(file)[1] == '.txt':
                    continue
                test_img[i,:,:,:]=cv2.imread(test_img_path+file)
                test_deep[i,:,:,0]=cv2.imread(test_deep_path+filename+'.jpg',0)
                test_labels[i,:,:,0]=cv2.imread(test_gt_path+filename+'.png',0)
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
            batch_ds = self.ds[self.cur_order[self.batch_start: batch_end]]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end],...]
            batch_ns = self.names[self.cur_order[self.batch_start: batch_end]]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ds,batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ds = self.ds[self.cur_order[self.batch_start: batch_end]]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end],...]
        batch_ns = self.names[self.cur_order[self.batch_start: batch_end]]
        self.batch_start += batch_size
        return batch_xs, batch_ds,batch_ys,batch_ns
