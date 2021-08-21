"""
Utilities for importing the nju2000 and nlpr dataset.

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

class TrainData(object):
    """
    Inputs to constructor
    =====================
        The training data containing 1500+500=2000 examples, the test data
        containing 485+498=983 examples
    """
    def __init__(self):
        root='/home/jackice/PycharmProjects/Adversary-about-RGBD-SOD-main/dataset/nju2000'
        root2='/home/jackice/PycharmProjects/Adversary-about-RGBD-SOD-main/dataset/NLPR'
        
        train_img_path =root+'/train/Img/'
        train_deep_path=root+'/train/deep/'
        train_gt_path=root+'/train/GT/'
        test_img_path =root+'/test/Img/'
        test_deep_path=root+'/test/deep/'
        test_gt_path=root+'/test/GT/'
        
        train_img_path2 =root2+'/train/Img/'
        train_deep_path2=root2+'/train/deep/'
        train_gt_path2=root2+'/train/GT/'
        test_img_path2 =root2+'/test/Img/'
        test_deep_path2=root2+'/test/deep/'
        test_gt_path2=root2+'/test/GT/'

        size=224
        train_img = np.zeros((2000, size, size, 3), dtype='float32')
        train_deep=np.zeros((2000,size,size,1), dtype='uint8')
        train_labels = np.zeros((2000,size,size,1), dtype='uint8')
        train_names=[]
        test_img = np.zeros((983, size, size, 3), dtype='float32')
        test_deep=np.zeros((983,size,size,1), dtype='uint8')
        test_labels = np.zeros((983,size,size,1), dtype='uint8')
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
        for root, dirs, files in os.walk(train_img_path2):
            for file in files:
                filename = os.path.splitext(file)[0]
                if  os.path.splitext(file)[1]=='.txt':
                    continue
                train_img[i,:,:,:]=cv2.resize(cv2.imread(train_img_path2+file),(size,size),interpolation=cv2.INTER_CUBIC)
                train_deep[i,:,:,0]=cv2.resize(cv2.imread(train_deep_path2+filename+'.png',0),(size,size),interpolation=cv2.INTER_CUBIC)
                train_labels[i,:,:,0]=cv2.resize(cv2.imread(train_gt_path2+filename+'.png',0),(size,size),interpolation=cv2.INTER_CUBIC)
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
        for root,dirs,files in os.walk(test_img_path2):
            for file in files:
                filename = os.path.splitext(file)[0]
                if os.path.splitext(file)[1] == '.txt':
                    continue
                test_img[i,:,:,:]=cv2.resize(cv2.imread(test_img_path2+file),(size,size),interpolation=cv2.INTER_CUBIC)
                test_deep[i,:,:,0]=cv2.resize(cv2.imread(test_deep_path2+filename+'.png',0),(size,size),interpolation=cv2.INTER_CUBIC)
                test_labels[i,:,:,0]=cv2.resize(cv2.imread(test_gt_path2+filename+'.png',0),(size,size),interpolation=cv2.INTER_CUBIC)
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
            batch_ns = []
            for idx in self.cur_order[self.batch_start: batch_end]:
                # print(idx)
                batch_ns.append(self.names[idx])
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
        batch_ns=[]
        for idx in self.cur_order[self.batch_start: batch_end]:
            # print(idx)
            batch_ns.append(self.names[idx])
        self.batch_start += batch_size
        return batch_xs, batch_ds,batch_ys,batch_ns

class AugmentedTrainData(object):
    """
    Data augmentation wrapper over a loaded dataset.
    Inputs to constructor
    =====================
        - raw_TrainData: the loaded nju2000 dataset, via the TrainData class
        - sess: current tensorflow session
        - model: current model (needed for input tensor)
    """
    def __init__(self, raw_TrainData, sess, model):
        assert isinstance(raw_TrainData, TrainData)
        self.image_size = 224

        # create augmentation computational graph for rgb
        self.x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, self.image_size + 4, self.image_size + 4),
            self.x_input_placeholder)
        cropped = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size,self.image_size,3]), padded)
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
        self.augmented = flipped

        # create augmentation computational graph for d
        self.d_input_placeholder = tf.placeholder(tf.uint8, shape=[None, 224, 224, 1])
        padded_d = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, self.image_size + 4, self.image_size + 4),
                           self.d_input_placeholder)
        cropped_d = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size,self.image_size,1]), padded_d)
        flipped_d = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped_d)
        self.augmented_d = flipped_d

        # self.label_names = raw_TrainData.label_names
        self.train_data = AugmentedDataSubset(raw_TrainData.train_data, sess,
                                              self.x_input_placeholder,
                                              self.d_input_placeholder,
                                              self.augmented,
                                              self.augmented_d)
        self.eval_data = AugmentedDataSubset(raw_TrainData.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.d_input_placeholder,
                                             self.augmented,
                                             self.augmented_d)

class AugmentedDataSubset(object):
    def __init__(self, raw_datasubset, sess, x_input_placeholder,d_input_placeholder,augmented,augmented_d):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.d_input_placeholder = d_input_placeholder
        self.augmented = augmented
        self.augmented_d=augmented_d

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size, multiple_passes,reshuffle_after_pass)
        # images = raw_batch[0].astype(np.float32)
        return self.sess.run([self.augmented,self.augmented_d],
                             feed_dict={self.x_input_placeholder:raw_batch[0],
                                        self.d_input_placeholder:raw_batch[1]}), raw_batch[2]