list=[]
merge=h5py.File('./train_data/merge2.h5','w')

d=h5py.File('./train_data/nju2000.h5','r')
list.append(d)
e=h5py.File('./train_data/nlpr.h5','r')
list.append(e)

# a = h5py.File('./train_data/ssd100.h5','r')
# list.append(a)
# b=h5py.File('./train_data/rgbd135.h5','r')
# list.append(b)
# c=h5py.File('./train_data/lfsd.h5','r')
# list.append(c)

for index,f in enumerate(list):
        print(index)
        print(f.keys())
        if index==0:

                sx = f['x'][:]
                sy = f['y'][:]
                sz = f['z'][:]
                sx2=f['x2'][:]
                sval_x = f['x_val'][:]
                sval_y = f['y_val'][:]
                sval_z = f['z_val'][:]
                sval_x2 = f['X_val_2'][:]
        elif index==1:
                x = f['x'][:]
                y = f['y'][:]
                z = f['z'][:]
                x2=f['x2'][:]
                val_x = f['x_val'][:]
                val_y = f['y_val'][:]
                val_z = f['z_val'][:]
                val_x2 = f['X_val_2'][:]

                sx=np.concatenate((sx,x),axis=0)
                sy = np.concatenate((sy, y), axis=0)
                sz = np.concatenate((sz, z), axis=0)
                sx2 = np.concatenate((sx2, x2), axis=0)
                sval_x=np.concatenate((sval_x,val_x),axis=0)
                sval_y = np.concatenate((sval_y, val_y), axis=0)
                sval_z = np.concatenate((sval_z, val_z), axis=0)
                sval_x2 = np.concatenate((sval_x2, val_x2), axis=0)
        # else:
        #         # x = f['x'][:]
        #         # y = f['y'][:]
        #         # z = f['z'][:]
        #         # x2=f['x2'][:]
        #         val_x = f['x_val'][:]
        #         val_y = f['y_val'][:]
        #         val_z = f['z_val'][:]
        #         val_x2 = f['X_val_2'][:]
        #
        #         # sx=np.concatenate((sx,x),axis=0)
        #         # sy = np.concatenate((sy, y), axis=0)
        #         # sz = np.concatenate((sz, z), axis=0)
        #         # sx2 = np.concatenate((sx2, x2), axis=0)
        #         sval_x = np.concatenate((sval_x, val_x), axis=0)
        #         sval_y = np.concatenate((sval_y, val_y), axis=0)
        #         sval_z = np.concatenate((sval_z, val_z), axis=0)
        #         sval_x2 = np.concatenate((sval_x2, val_x2), axis=0)
# a.close()
# b.close()
# c.close()
d.close()
e.close()
merge['x'] = sx
merge['y'] = sy
merge['z'] = sz
merge['x2']=sx2
merge['x_val'] = sval_x
merge['y_val'] = sval_y
merge['z_val']=sval_z
merge['X_val_2']=sval_x2
merge.close()
