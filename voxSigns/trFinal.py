#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:19:38 2017

@author: avarfolomeev
"""

import tensorflow as tf
import numpy as np


EPOCHS = 200
BATCH_SIZE = 64

tf.reset_default_graph()


keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')                                           


#sigs are 32x32x3
batch_x = tf.placeholder(tf.float32, [None,32,32,1], name = "input_image")
# 
batch_y = tf.placeholder(tf.int32, (None), name = 'labels')


n_classes = 11 #37 #7 for thic

ohy = tf.one_hot(batch_y,n_classes);
fc2 = MixNetArr(batch_x, keep_prob, n_classes)

step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-4
learning_rate = tf.train.exponential_decay(starter_learning_rate, step, 
                                          50, 1, staircase=True)

starter_learning_rate = 1e-3

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=ohy))
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.minimize(loss_op, global_step = step)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(ohy, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver();

#%%

def eval_data(xv, yv):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    # If dataset.num_examples is not divisible by BATCH_SIZE
    # the remainder will be discarded.
    # Ex: If BATCH_SIZE is 64 and training set has 55000 examples
    # steps_per_epoch = 55000 // 64 = 859
    # num_examples = 859 * 64 = 54976
    #
    # So in that case we go over 54976 examples instead of 55000.
    _nSamples = xv.shape[0]
    _batchSize = BATCH_SIZE
    total_acc, total_loss = 0, 0
    sess = tf.get_default_session()
    

    for batch_start in range(0,_nSamples,_batchSize):
        if (batch_start+_batchSize >= _nSamples):
            _batchSize = _nSamples - batch_start;

        bx = xv[batch_start:batch_start + _batchSize]
        if (len(bx.shape) < 4):
            bx = np.expand_dims(bx,-1)
        
        by = yv[batch_start:batch_start + _batchSize]
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={batch_x : bx, batch_y: by, keep_prob: 1.0})
        total_acc += (acc * bx.shape[0])
        total_loss += (loss * bx.shape[0])
    return total_loss/_nSamples, total_acc/_nSamples

    


#%%
    
save_net = './nets/arrows-5.ckpt'

with tf.Session() as sess:

    # Train model
    sess.run(tf.global_variables_initializer())

    loss = 0
    

    _nSamples = Xgn_t.shape[0]
    idx = np.arange(_nSamples)
    _batchSize = BATCH_SIZE
        
    val_acc_max = 0.9    
    
    for epoch in range(EPOCHS):
        np.random.shuffle(idx)
        for _batchStart in range(0,_nSamples,_batchSize):
            if (_batchStart+_batchSize >= _nSamples):
                _batchSize = _nSamples - _batchStart;

            bx = Xgn_t[idx[_batchStart:_batchStart + _batchSize]]
            if (len(bx.shape) < 4):
                bx = np.expand_dims(bx,-1)
            
            by = Yg_t[idx[_batchStart:_batchStart + _batchSize]]
    
            _,loss = sess.run([train_op, loss_op], feed_dict={batch_x: bx, batch_y: by, keep_prob: 0.5})
    
        trn_loss, trn_acc = eval_data(Xgn_t, Yg_t)
        val_loss, val_acc = eval_data(X_val, y_val)
        print("EPOCH {} ...".format(epoch+1), 
              "Learning rate", "%.9f" % sess.run(learning_rate))
        print("Validation loss = {:.3f}".format(val_loss), 
              "Validation accuracy = {:.3f}".format(val_acc))
        print("Train loss (drop) = {:.5f}".format(loss), 
              "Train loss = {:.5f}".format(trn_loss), 
              "Train acc  = {:.5f}".format(trn_acc) )
        print()
        
        if (val_acc > val_acc_max ):
            val_acc_max = val_acc;
            saver.save(sess, save_net, global_step=epoch)
                #save empty file with loss value           
            fn =  save_net + '-' + str(epoch) + '.' + str(val_acc)
            f = open(fn,"wb")
            f.close()

    
    saver.save(sess,save_file)    
    
    # Evaluate on the test data
    #tst_loss, tst_acc = eval_data(Xgn_test, y_test)
    #print("Test loss = {:.3f}".format(tst_loss), "Test accuracy = {:.3f}".format(tst_acc))



#%%
    
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

base_dir = '/media/avarfolomeev/storage/Data/Voxels/arrows/'

load_net = base_dir + 'nets/arrows-4.ckpt'

saver = tf.train.import_meta_graph(load_net + '.meta')
saver.restore(sess,load_net)


model2 = tf.get_default_graph()

writer = tf.summary.FileWriter('/tmp/log/tf', sess.graph)
writer.close()


batch_x = tf.placeholder(tf.float32, [None,32,32,1])

input_image = model.get_tensor_by_name('placeholder_1:0')
keep_prob = model.get_tensor_by_name('keep_prob:0')
nn_output = model.get_tensor_by_name('lin2:0')
    