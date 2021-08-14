from model import *
import numpy as np
from keras.preprocessing.image import array_to_img
import os
from keras import backend as K
from attack import getInput

def myCE_final(y_pred,guide,mask):
    l = -guide * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0)) * mask
    return l
def myCELoss_final(guide,mask):
    def l_final(y_ture,y_pred):
        return myCE_final(y_pred,guide,mask)
    return l_final

def getG(img,deep):
    source_maps = model.predict([img, deep]).flatten()
    source_b = [1 if e >= 0.5 else 0 for e in source_maps]

    mask=np.ones_like(source_maps)
    # guide_b=[1 if e >=0.5 else 0 for e in guide_maps.flatten()]
    idx=np.argwhere(source_b==guide_b)
    mask[idx]=0
    mask=mask.reshape(guide_maps.shape)

    myloss = myCELoss_final(guide_maps,mask)
    model.compile(loss=myloss, optimizer='SGD')

    get_gradients = K.function(inputs=input_tensors, outputs=grads)
    dx = get_gradients([img, deep, np.ones(1), gt, 0])
    d_rgb = dx[0][0]  # 224,224,3
    d_deep = dx[1][0]  # 224,224,1
    return d_rgb,d_deep

rootdir = './dataset/NLPR/test/'
img_width,img_height=480,640

# list=[]
# f=open('extracted.txt')
# line=f.readline()
# while line:
#     list.append(line.split('.')[0])
#     line=f.readline()
# f.close()
# for root, dirs, files in os.walk(rootdir+'mlafter/'):
#         for file in files:
#                 list.append(os.path.splitext(file)[0])

for root, dirs, files in os.walk(rootdir+'Data/Img/'):
        for file in files:
                print(file)
                filename = os.path.splitext(file)[0]
                # if filename in list:
                #     continue
                if os.path.splitext(file)[1]=='.txt':
                    continue
                model = vgg16_deep_fuse_model(img_width,img_height)
                model.load_weights('./checkpoints/vgg16_deep_fuse_512.0.323.hdf5',by_name=True)
                input_tensors = [
                        model.inputs[0],  # input1_0 numpy数组 RGB
                        model.inputs[1],  # input2_0 numpy数组 D
                        model.sample_weights[0],  # 各个样本的权值，一样就都填 1，是numpy数组
                        model.targets[0],  # 输入的标签，是numpy数组
                        K.learning_phase(),  # 默认为0，表示test
                    ]
                grads = K.gradients(model.total_loss, model.inputs)

                img,gt,deep,biimg=getInput(rootdir+'Data/Img/'+filename+'.jpg',rootdir+'Data/GT/'+filename+'.png',rootdir+'Data/deep/'+filename+'.png',img_width,img_height)
                x_rgb=img.copy()
                x_deep=deep.copy()

                source_maps = model.predict([img, deep])
                guide_maps=1-source_maps
                guide_b = [1 if e >= 0.5 else 0 for e in guide_maps.flatten()]
                # print('min guide is:%f'%np.min(guide_maps))

                itr=1
                MAX_ITER=500
                step_size=0.003
                bound=255/255
                while itr<MAX_ITER:
                    print('iter:%d' % itr )
                    d_rgb,d_deep=getG(img,deep)

                    img=img+step_size*np.sign(d_rgb)
                    deep=deep+np.sign(d_deep)*step_size
                    img[img>1]=1
                    img[img<0]=0
                    deep[deep<0]=0
                    deep[deep>1]=1

                    a=img-x_rgb
                    b=deep-x_deep
                    if np.max(a)>=bound or np.max(b)>=bound:
                        print('end when itr=%d'%itr)
                        break;
                    # pred = model.predict([img, deep])
                    # tmp=array_to_img(pred[0])
                    # tmp.save(rootdir+'tmp/'+filename+str(itr)+'.png')
                    itr=itr+1

                a = array_to_img(img[0])
                a.save(rootdir+filename+'Img.png')
                b = array_to_img(deep[0])
                b.save(rootdir+filename+'Deep.png')
                adver_pred = model.predict([img, deep])
                after = array_to_img(adver_pred[0])
                after.save(rootdir+filename+'.png')

                tf.keras.backend.clear_session()
                tf.reset_default_graph()


