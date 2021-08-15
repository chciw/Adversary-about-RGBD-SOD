import h5py
import numpy as np
import cv2
import os

rotate if wh is hw
readdir='./dataset/NLPR/test/OriginSizeData/deep/'
writedi='./dataset/NLPR/test/Data/deep/'
for root, dirs, files in os.walk(readdir):
        for file in files:
                print(readdir+file)
                if os.path.splitext(file)[1]=='.txt':
                        continue
                img=cv2.imread(readdir+file)
                # cv2.imshow('origin',img)
                # print('origin size ',img.shape)# h,w
                if (img.shape)[0]==480:# if h ==480 then rotate
                    img=np.rot90(img)
                cv2.imwrite(writedi+file,img)
