# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:33:36 2017

@author: diz
"""
import tensorflow as tf
import glob
import scipy


#

_classes = classes_thin

n_classes = len(_classes) #7 for thick

tf.reset_default_graph();
keep_prob = tf.placeholder(tf.float32, name='keep_prob')                                           

batch_x = tf.placeholder(tf.float32, [None,32,32,3], name='batch_x')
fc2 = MixNet(batch_x,n_classes)

softmax = tf.nn.softmax(fc2);        
winner = tf.argmax(softmax,1)

saver = tf.train.Saver();

#%%

def test_new_data(xv):
    sess = tf.get_default_session()
    top  = sess.run([winner],   feed_dict={batch_x : xv , keep_prob: 1.0})     
    return top[0]



#%%

save_file = '/media/avarfolomeev/storage/Data/Voxels/Signs/nets/thin-130.ckpt' #./mixNet0_named.ckpt' #'./nets/thin/mixtNet0_named.ckpt.final1'

print ("testing " , save_file)

X = Xgn_val[:100]
nSamples = len(X)
batchSize = 16

with tf.Session() as sess:

    saver.restore(sess, save_file)

    top = np.zeros(nSamples, dtype = int)

    for batch in range(0,nSamples,batchSize):
        if (batch+batchSize >= nSamples):
            batchSize = nSamples - batch;
        print(batch, batchSize)
        top[batch:batch+batchSize] = test_new_data(X[batch:batch+batchSize]); #Xgn_test,y_test)

print( top.shape)

#%%

dirSorted = '/media/avarfolomeev/storage/Data/Voxels/Signs/results/thin'

dirsOut = []

for c in _classes:
    p = os.path.join(dirSorted,c)
    os.makedirs(p)
    dirsOut.append(p)


#%%
batchSize = 100
filenames =  [f for f in glob.iglob(root_dir + '/**/*.png', recursive=True)]

nSamples = len(filenames)
results = np.zeros(batchSize,dtype=int)
images = np.zeros((batchSize,32,32,3))


with tf.Session() as sess:
    
    saver.restore(sess, save_file)


    for batchStart in range(0, nSamples, batchSize):
        if (batchStart + batchSize >=  nSamples):
             batchSize = nSamples - batchStart;
             
        for b in range (batchSize):
            f = b + batchStart;
            img = scipy.misc.imread(filenames[f])
            (w,h,c) = img.shape
            
            if ( abs (w-h) > 12) :
                continue;
                
            if ( w != h):
                w = min(w,h)
                img = img[:w,:w,:]
            img = scipy.misc.imresize(img,(32,32))
            imf = np.float32(img);
            imf = (imf - 128) / 128.
            images[b] = imf
        
        results = test_new_data(images); #Xgn_test,y_test)
        
        for b in range(batchSize):
            f = b + batchStart
            try:
                os.symlink(filenames[f], os.path.join(dirsOut[results[b]],
                           os.path.split(filenames[f])[1]))
            except Exception:
                print(os.path.split(filenames[f])[1], " exists")
                    

        print(batchStart, ' from ', nSamples)




