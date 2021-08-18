import numpy as np
from PIL import Image

datalist=np.load('./pgdx.npy')
print(type(datalist))
print(type(datalist[0,:,:,:]))
for i in range(500):
    array=datalist[i,:,:,:]
    img=Image.fromarray(np.uint8(array))
    img.save('./imgs/'+str(i)+'.jpg')