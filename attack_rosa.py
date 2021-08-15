import os
from model import *
import numpy as np
from keras.preprocessing.image import array_to_img
from keras import backend as K
from attack import getInput

rootdir = './dataset/NLPR/test/Data/'
img_width, img_height =480,640

MAX_ITER = 30
step_size = 0.003
bound = 20 / 255

# list=[]
# for root, dirs, files in os.walk('./dataset/NJUD/test/rosa/pred/'):
#         for file in files:
#                 list.append(os.path.splitext(file)[0])

for root, dirs, files in os.walk(rootdir+'Img/'):
        for file in files:
                print(file)
                filename = os.path.splitext(file)[0]
                # if filename in list:
                #         continue
                if os.path.splitext(file)[1]=='.txt':
                        continue
                model = vgg16_deep_fuse_model(img_width, img_height)
                model.load_weights('./checkpoints/vgg16_deep_fuse_512.0.323.hdf5', by_name=True)

                img,gt,deep,img2=getInput(rootdir+'Img/'+filename+'.jpg',rootdir+'GT/'+filename+'.png',rootdir+'deep/'+filename+'.png',img_width,img_height)
                img_pred=model.predict([img,deep]) # (1,480, 640, 1)->B,H,W,C
                # before = array_to_img(img_pred[0])
                # before.save(rootdir+'before/'+filename+'.png')

                img_pred=img_pred[0] # 480,640,1
                target=np.argwhere(img_pred>=0)# 所有位置索引
                idx=np.argwhere((img_pred>(125/255)))# 显著区域索引
                h=idx[:,0]
                w=idx[:,1]
                saliency_map=np.zeros((img_height,img_width,1))
                saliency_map[h,w,0]=1# 0 1显著图

                #start
                input_tensors = [
                        model.inputs[0],# input1_0 numpy数组 RGB
                        model.inputs[1],# input2_0 numpy数组 D
                        model.sample_weights[0], # 各个样本的权值，一样就都填 1，是numpy数组
                        model.targets[0], # 输入的标签，是numpy数组
                        K.learning_phase(), # 默认为0，表示test
                    ]
                mean_0=np.mean(img[0,:,:,0])
                mean_1=np.mean(img[0,:,:,1])
                mean_2=np.mean(img[0,:,:,2])
                mean_4=np.mean(deep)
                img[0,:,:,0]=img[0,:,:,0]-mean_0
                img[0,:,:,1]=img[0,:,:,1]-mean_1
                img[0,:,:,2]=img[0,:,:,2]-mean_2
                deep=deep-mean_4

                x_rgb=img.copy()
                x_deep=deep.copy()

                a = tf.ones((1, img_height, img_width, 1))
                mask = np.zeros((img_height, img_width, 1))
                itr = 1
                e = 0.0
                while itr<MAX_ITER and len(target)!=0 and e<bound:
                        print('iter %d'% itr)
                        itr = itr + 1
                        # 目标像素mask
                        mask[mask==1]==0
                        # mask=np.zeros((img_height,img_width,1))
                        h=target[:,0]
                        w=target[:,1]
                        mask[h,w,0]=1

                        # 规定输入输出的计算关系
                        b = (a - model.output) * mask
                        grads = K.gradients(b, model.inputs)
                        # 编译计算图。这条语句以后， get_gradients就是一个可用的Keras函数
                        get_gradients = K.function(inputs=input_tensors,outputs=grads)
                        dx=get_gradients([img, deep,np.ones(1),  gt, 0 ])
     
                        tf.keras.backend.clear_session()
                        tf.reset_default_graph()
                        model = vgg16_deep_fuse_model(img_width, img_height)
                        model.load_weights('./checkpoints/vgg16_deep_fuse_512.0.323.hdf5', by_name=True)
                        
                        c=model.output * mask
                        grads_2=K.gradients(c, model.inputs)
                        get_gradients_2 = K.function(inputs=input_tensors,outputs=grads_2)
                        dx_2=get_gradients_2([img, deep, np.ones(1),  gt, 0 ])
                        
                        p_rgb=dx[0]-dx_2[0]
                        p_deep=dx[1]-dx_2[1]
                        
                        p_rgb=step_size*p_rgb/np.max(p_rgb)
                        p_deep=step_size*p_deep/np.max(p_deep)
                        img=img+p_rgb
                        deep=deep+p_deep

                        a=np.abs(np.max(img-x_rgb))
                        b=np.abs(np.max(deep-x_deep))
                        e=a if a>b else b
                        print('inf. norm of r :%f' % e)

                        img_pred = model.predict([img, deep])[0]
                        idx = np.argwhere((img_pred > (125 / 255)))
                        h = idx[:, 0]
                        w = idx[:, 1]
                        new_saliency_map = np.zeros((img_height,img_width,1))
                        new_saliency_map[h, w, 0] = 1
                        target=np.argwhere(saliency_map==new_saliency_map)
                        print('%d target pixels remained'% len(target))
                print('iterations end when itr is:%d'% itr)

                img[0,:,:,0]=img[0,:,:,0]+mean_0
                img[0,:,:,1]=img[0,:,:,1]+mean_1
                img[0,:,:,2]=img[0,:,:,2]+mean_2
                deep=deep+mean_4
                img[img>1]=1
                img[img<0]=0
                deep[deep>1]=1
                deep[deep<0]=0

                a = array_to_img(img[0])
                a.save('./dataset/NLPR/test/rosa/rgb/'+filename+'.jpg')
                b = array_to_img(deep[0])
                b.save('./dataset/NLPR/test/rosa/deep/'+filename+'.png')
                after = array_to_img(img_pred)
                after.save('./dataset/NLPR/test/rosa/pred/'+filename+'.png')

                tf.keras.backend.clear_session()
                tf.reset_default_graph()
