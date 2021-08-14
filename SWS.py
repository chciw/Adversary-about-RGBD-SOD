import random
import cv2
import os
from skimage import img_as_float, img_as_int
from skimage.segmentation import slic, mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

def SWS(img):
    image = img_as_float(cv2.imread(img)) # w=378 h=596
    numSegments=40  # 400->500x500 np.unique(seg)总是比这里指定的少
    segments = slic(image, n_segments=numSegments, sigma=5,start_label=1) # (596,378) (h,w)

    # plt_img=io.imread(img)
    # plt.subplot(131)
    # plt.title('image')
    # plt.imshow(plt_img)
    # plt.subplot(132)
    # plt.title('segments')
    # plt.imshow(segments)
    # plt.subplot(133)
    # plt.title('image and segments')
    # plt.imshow(mark_boundaries(plt_img, segments))
    # plt.show()

    # print("segments:\n", segments)
    print("np.unique(segments):", np.unique(segments))

    # output=np.zeros(image.shape,dtype='float64')
    # cv2.imshow('origin', image)
    for (i, segVal) in enumerate(np.unique(segments)):
        # 为segment构造超像素mask
        # print("[x] inspecting segment {}, for {}".format(i, segVal))
        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255
        # cv2.imshow("Mask", mask)

        sPixel=np.multiply(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) > 0)
        # cv2.imshow("Applied", sPixel)
        # 在mask范围内打乱像素
        h_sp=np.argwhere(mask==255)[:,0] # 所有行的第0个数据
        w_sp=np.argwhere(mask==255)[:,1] # 所有行的第1个数据
        h_min,h_max=np.min(h_sp),np.max(h_sp)
        w_min,w_max=np.min(w_sp),np.max(w_sp)
        n_swap=(h_max-h_min)*(w_max-w_min)//20 # 像素交换次数

        while n_swap!=0:
            h1=random.randint(h_min, h_max)
            w1 = random.randint(w_min, w_max)
            if mask[h1,w1]==0:
                continue
            h2 = random.randint(h_min, h_max)
            w2 = random.randint(w_min, w_max)
            if mask[h2,w2]==0:
                continue
            if h1!=h2 or w1!=w2 :
                tmp=image[h1,w1,:]
                image[h1,w1,:]=image[h2,w2,:]
                image[h2,w2,:]=tmp
            n_swap=n_swap-1
        # cv2.imshow('sp',np.multiply(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) > 0))
        # cv2.waitKey(0)
        # output+=sPixel
        # cv2.waitKey(0)
    # cv2.imshow('SWS',image)
    # cv2.waitKey(0)
    cv2.imwrite(rootdir + 'SWSDeep/' + file, image*255)
    # return image

rootdir = './dataset/NLPR/test/rosa/'
# input_w=320
# input_h=320

# list=[]
# f=open('extracted.txt')
# line=f.readline()
# while line:
#     list.append(line.split('.')[0])
#     line=f.readline()
# f.close()

for root, dirs, files in os.walk(rootdir+'deep/'):
        for file in files:
                print(file)
                filename=os.path.splitext(file)[0]
                # if filename in list:
                #     continue
                if os.path.splitext(file)[1]=='.txt':
                        continue
                # img = cv2.imread(rootdir+'after/'+file)
                # img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
                # cv2.imwrite(rootdir+'GT/'+file,img)
                SWS(rootdir+'deep/'+file)

