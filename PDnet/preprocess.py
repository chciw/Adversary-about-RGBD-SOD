import cv2
import numpy as np
from PIL import Image
import h5py
import operator
import random

# PIC_PATH2 = 'D:/hh/deeplearning/Database/IRFF_dataset/RGBD_for_train/RGB/'
# SALDEEP_PATH2 = 'D:/hh/deeplearning/Database/IRFF_dataset/RGBD_for_train/depth/'
# SALMASK_PATH2='D:/hh/deeplearning/Database/IRFF_dataset/RGBD_for_train/GT/'
# test
VAL_PATH = '/home/jackice/PycharmProjects/pythonProject1/dataset/NLPR/test/Img/'
VALDEEP_PATH = '/home/jackice/PycharmProjects/pythonProject1/dataset/NLPR/test/deep/'
VALMASK_PATH='/home/jackice/PycharmProjects/pythonProject1/dataset/NLPR/test/GT/'
# train
PIC_PATH2 = '/home/jackice/PycharmProjects/pythonProject1/dataset/NLPR/train/Img/'
SALDEEP_PATH2 = '/home/jackice/PycharmProjects/pythonProject1/dataset/NLPR/train/deep/'
SALMASK_PATH2='/home/jackice/PycharmProjects/pythonProject1/dataset/NLPR/train/GT/'

datalist = open(PIC_PATH2+'list.txt','r')
namelist=[l.strip('\n') for l in datalist.readlines()]
sallist= open(SALMASK_PATH2+'list.txt','r')
sallist=[l.strip('\n') for l in sallist.readlines()]
deplist= open(SALDEEP_PATH2+'list.txt','r')
deplist=[l.strip('\n') for l in deplist.readlines()]

val_datalist = open(VAL_PATH+'list.txt','r')
val_namelist=[l.strip('\n') for l in val_datalist.readlines()]
val_sallist = open(VALMASK_PATH+'list.txt','r')
val_sallist=[l.strip('\n') for l in val_sallist.readlines()]
deplist2= open(VALDEEP_PATH+'list.txt','r')
deplist2=[l.strip('\n') for l in deplist2.readlines()]
print(deplist2)


input_h=224
input_w=224
output_h=224
output_w=224
NumSample=len(namelist)
val_num = len(val_namelist)

X1 = np.zeros((NumSample,input_h, input_w,3), dtype='float32')
Y1 = np.zeros((NumSample,output_h,output_w,1), dtype='uint8')
Z1=np.zeros((NumSample,output_h,output_w,1), dtype='uint8')
X2 = np.zeros((NumSample,input_h, input_w,3), dtype='float32')
NumAll = NumSample+val_num
print (NumAll)
print(val_num)
VAL_X = np.zeros((val_num,input_h, input_w,3), dtype='float32')
VAL_Y = np.zeros((val_num,output_h,output_w,1), dtype='uint8')
VAL_Z= np.zeros((val_num,output_h,output_w,1), dtype='uint8')
VAL_X2 = np.zeros((val_num,input_h, input_w,3), dtype='float32')
for i in range(val_num):
    img = cv2.imread(VAL_PATH+val_namelist[i], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print img.shape
    img = cv2.resize(img,(input_w,input_h),interpolation=cv2.INTER_CUBIC)
    img=img.astype(np.float32)/255.
    VAL_X[i]=img
    
    label = cv2.imread(VALMASK_PATH+val_sallist[i],cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label,(output_w,output_h),interpolation=cv2.INTER_CUBIC)
    label = label.astype(np.float32)
    label /=255
    label[label > 0.5]=1
    label[label <=0.5]=0
    label=label.astype(np.uint8)
    VAL_Y[i]=label.reshape(output_h,output_w,1)

    deep = cv2.imread(VALDEEP_PATH + deplist2[i], cv2.IMREAD_GRAYSCALE)
    deep = cv2.resize(deep, (output_w, output_h), interpolation=cv2.INTER_CUBIC)
    deep = deep.astype(np.float32)
    deep /= 255
    # deep=deep*score2[i]
    deep = deep.astype(np.float32)
    VAL_Z[i] = deep.reshape(output_h, output_w, 1)

    img2=cv2.imread(VAL_PATH+val_namelist[i], cv2.IMREAD_COLOR)
    img2 = cv2.bilateralFilter(src=img2, d=0, sigmaColor=100, sigmaSpace=5)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
    img2 = img2.astype(np.float32) / 255.
    VAL_X2[i] = img2
# for i in range(NumSample):
# #name1 = namelist[i][0:namelist[i].index('.')]
# #   name2 = sallist[i][0:sallist[i].index('_')]
# #   if name1 != name2:
# #       print error
#     img = cv2.imread(PIC_PATH+namelist[i], cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # print (img.shape)
#     img = cv2.resize(img,(input_w,input_h),interpolation=cv2.INTER_CUBIC)
#     #print img.shape
#     #cv2.imshow('show',img)
#     img=img.astype(np.float32)/255.
#     #img = img /255.
#     #img -= 1.
#     if(operator.eq(img.shape , (input_h,input_w,3))):
#         #img = img.transpose(2,0,1).reshape(3, input_h, input_w)
#         X1[i]=img
#     else:
#         print ('error')
#
#     #np.set_printoptions(threshold='nan')
#     label = cv2.imread(SALMAP_PATH+sallist[i],cv2.IMREAD_GRAYSCALE)
#     #label = loadSaliencyMapSUN(names[i])
#     label = cv2.resize(label,(output_w,output_h),interpolation=cv2.INTER_CUBIC)
#     #cv2.imshow('label',label)
#     #cv2.waitKey(0)
#     label = label.astype(np.float32)
#     label /=255
#     label[label > 0.5]=1
#     label[label <=0.5]=0
#     label=label.astype(np.uint8)
# #	print 'data',X1[i]
# #	print 'label',label
#     #Y1.append(label.reshape(1,48*48))
#     Y1[i]=label.reshape(output_h,output_w,1)


for i in range(NumSample):
    img =cv2.imread(PIC_PATH2+namelist[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print (img.shape)
    img = cv2.resize(img,(input_w,input_h),interpolation=cv2.INTER_CUBIC)
    img=img.astype(np.float32)/255.
    X1[i]=img

    label = cv2.imread(SALMASK_PATH2+sallist[i],cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label,(output_w,output_h),interpolation=cv2.INTER_CUBIC)
    label = label.astype(np.float32)
    label /=255
    label[label > 0.5]=1
    label[label <=0.5]=0
    label=label.astype(np.uint8)
    Y1[i]=label.reshape(output_h,output_w,1)

    deep = cv2.imread(SALDEEP_PATH2 + deplist[i], cv2.IMREAD_GRAYSCALE)
    deep = cv2.resize(deep, (output_w, output_h), interpolation=cv2.INTER_CUBIC)
    deep = deep.astype(np.float32)
    deep /= 255
    # deep=deep*score1[i]
    deep = deep.astype(np.float32)
    Z1[i] = deep.reshape(output_h, output_w, 1)

    img2 =cv2.imread(PIC_PATH2+namelist[i])
    img2 = cv2.bilateralFilter(src=img2, d=0, sigmaColor=100, sigmaSpace=5)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2,(input_w,input_h),interpolation=cv2.INTER_CUBIC)
    img2=img2.astype(np.float32)/255.
    X2[i]=img2


# random.seed(1)
# rand=range(NumAll)
# random.shuffle(list(rand))
# split_at = int(NumAll * 0.88)
# x , y = X1[list(rand[0:split_at])],Y1[list(rand[0:split_at])]
# val_x ,val_y = X1[list(rand[split_at:])],Y1[list(rand[split_at:])]

f = h5py.File('./train_data/nju2000.h5','w')

f['x'] = X1
f['y'] = Y1
f['z'] = Z1
f['x2']=X2
f['x_val'] = VAL_X
f['y_val'] = VAL_Y
f['z_val']=VAL_Z
f['X_val_2']=VAL_X2
f.close()

#data_to_save = (X1, Y1)
#f = file('data_msra_200x150_T.cPickle', 'wb')
#pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
#f.close()
