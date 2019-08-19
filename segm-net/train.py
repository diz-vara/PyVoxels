# -*- coding: utf-8 -*-
"""
Created on 20190708_1430

@author: avarfolomeev
"""

import numpy as np
import os.path
import tensorflow as tf
import numpy as np
import helper
import sys


from read_ontology import read_ontology


base_dir = '/media/avarfolomeev/storage/Data/'

try:
    base_dir = os.environ['BASE_DATA_PATH']
except:
    pass
#%%
def train_nn(sess, net_name, epochs, batch_size, 
             dataset_dir, image_shape, ontology,
             train_op, loss, conf_matrix, print_op, #n, d,
             saver, 
             input_image, corr_label, nn_output, 
             keep_prob, learning_rate, base = 0):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    nets_dir = dataset_dir + '/nets/'
    try:
        os.makedirs(nets_dir)
    except:
        pass

    dataset_file = dataset_dir + '/dataset.txt'
    save_net = nets_dir + net_name;
    min_loss_file = nets_dir + net_name + '_min_loss.txt';
    lr_file = nets_dir + net_name + '_lr.txt';
    
    merged = tf.summary.merge_all()
    lr = 0.1
    min_loss = 1

    num_classes = len (ontology)

    sum_true = tf.reduce_sum(conf_matrix,axis=0)
    sum_pred = tf.reduce_sum(conf_matrix, axis = 1)
    t_intersect = tf.diag_part(conf_matrix)
    t_union = sum_true + sum_pred - t_intersect



    train_summary = tf.summary.scalar('train_loss', loss)
    val_summary = tf.summary.scalar('val_loss', loss)

    writer = tf.summary.FileWriter(dataset_dir + '/logs/' + net_name)

    print('training')


    for epoch in range (epochs):
        
        print (net_name + ' epoch {} ({}) '.format(epoch+base, epoch))
        sys.stdout.flush()
        
        get_train_batches_fn = build_batch_fn(dataset_file, 'train',
                                              image_shape, num_classes);
        
        get_val_batches_fn = build_batch_fn(dataset_file, 'val',
                                              image_shape, num_classes);
        bnum = 0
        #read updated lr from the file
        try:
            lr = float(open(lr_file).read())
        except:
            0;    
        _lr = lr #* 0.9965                     
        print(" LR = {:g}".format(lr))
        open(lr_file,'w').write(str(_lr))
        cum_loss = 0
        for image, label in get_train_batches_fn(batch_size):
            image = image[:batch_size]
            label = label[:batch_size]


            # sess.run(tf.local_variables_initializer())
            #sess.run(loss.initializer)                                     
            train_loss, _, t_summary = sess.run([loss, train_op, train_summary], #, n, d],
                                     feed_dict={
                                                 input_image:image, 
                                                 corr_label:label,
                                                 keep_prob:0.5, 
                                                 learning_rate:lr})
            


            cum_loss += train_loss
            sys.stdout.write('\rTrain ' + str(bnum*batch_size) + '  ' + str(train_loss) + '      \r') # + str(_n) + ' ' + str(_d) + '                         \r')
            sys.stdout.flush()      
            bnum = bnum + 1  
        train_loss = cum_loss / bnum
        sys.stdout.write("\r\nTrain loss: " + str(train_loss) + '\r\n')
        sys.stdout.flush()      
        bnum = 0
        
        cum_loss = 0
        intersect = np.zeros(num_classes)
        union = np.zeros(num_classes)
        for image, label in get_val_batches_fn(batch_size):
            image = image[:batch_size]
            label = label[:batch_size]
            val_loss, v_summary, _intersect, _union = sess.run([loss, val_summary, t_intersect, t_union],
                                     feed_dict={input_image:image, 
                                                corr_label:label,
                                                keep_prob:1, learning_rate:lr})
            intersect = intersect + _intersect
            union = union + _union

            sys.stdout.write('\rVal ' + str(bnum*batch_size) + '  ' + str(val_loss) + '   \r')
            sys.stdout.flush()      
            cum_loss += val_loss
            bnum = bnum + 1    
               
        val_loss = cum_loss / bnum
        
        writer.add_summary(t_summary, epoch)
        writer.add_summary(v_summary, epoch)
        #writer.add_summary(merged, epoch)
        print("\r\nLoss = {:g} {:g}".format(train_loss, val_loss))     
        print()
        
        union[union < 1] = 1
        IoU = intersect / union
        

        # read updated min_loss from the file
        try:
            min_loss = float(open(min_loss_file).read())
        except:
            pass;    
        
        if (val_loss < min_loss):
            print("saving at step {:d}".format(epoch+base))     
            min_loss = val_loss;
            saver.save(sess, save_net,
                       global_step=epoch+base)
            #save empty file with loss value           
            fn =  save_net + '-' + str(epoch+base) + '.' + str(val_loss)
            with open(fn,"w") as f:
                for cls in range(num_classes):
                    f.write("%15.15s, %.2f\n" % (ontology[cls].name, IoU[cls]) )
                    

                                        
            open(min_loss_file,'w').write(str(min_loss))
           
            
#%%
def build_batch_fn(_dataset_descriptor, _split, _image_shape, _num_classes):
    #print(_dataset_descriptor)
    try:
        _base_dir = open(_dataset_descriptor,'rt').readline().rstrip('\n')
    except:    
        _base_dir = os.path.split(_dataset_descriptor)[0]
    
    
    _fn = helper.gen_batch_function(_base_dir,
                                   _split,_image_shape, _num_classes)
    return (_fn)

#%%
def ce_loss(labels, logits, weights = 1.):

    labels = tf.cast(labels, tf.float32)
    logits = tf.nn.softmax(logits)
    cross_entropy = tf.reduce_sum(labels * tf.log(logits), axis = 0)
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits = logits, name = "cross-ent")
    loss = tf.reduce_mean(cross_entropy);

    return loss


def dice_loss(labels, logits, weights = 1.):
    smooth = tf.constant(1.)
    epsilon = tf.constant(1e-12)
    labels = tf.cast(labels, tf.float32)
    not_labels = tf.cast(1-labels, tf.float32)

    prob = tf.nn.softmax(logits)
    #logits = logits - tf.reduce_min(logits, axis = 3, keep_dims= True)
    #logits = logits / tf.reduce_max(logits, axis = 3, keep_dims = True) #tf.nn.softmax(logits)


    numerator = 2. * (tf.reduce_sum(labels * prob, axis=[0])) # - tf.reduce_sum(not_labels * prob, axis=[0]))
    denominator = tf.reduce_sum( labels + prob, axis=[0])

    #numerator = tf.reduce_sum(labels * prob , axis=[0]) + tf.reduce_sum(not_labels * (1 - prob), axis=[0])
    num_classes22 = tf.cast(tf.shape(labels)[-1], tf.float32)

    result = (weights * numerator) / (denominator + smooth)
    zero = tf.constant(0)
    return 1. - tf.reduce_mean(result) #, zero, zero


def generalized_dice_loss(labels, logits):

    smooth = tf.constant(1e-17)
    shape = tf.TensorShape(logits.shape).as_list()
    labels = tf.cast(labels,tf.float32)
    depth = int(shape[-1])
    #labels = tf.one_hot(labels, depth, dtype=tf.float32)
    logits = tf.nn.softmax(logits)
    weights = 1.0 / (tf.reduce_sum(labels, axis=[0,1,2])**2 + 1)

    numerator = tf.reduce_sum(labels * logits, axis=[0,1,2])


    #numerator = tf.reduce_sum(numerator)

    denominator = tf.reduce_sum(labels + logits, axis=[0, 1, 2])
    #denominator = tf.reduce_sum(denominator)
    score = (numerator + smooth)/(denominator + smooth)

    loss = - tf.log(tf.reduce_mean(score))
    return loss, tf.reduce_sum(score), tf.reduce_sum(logits)




            
#%%
def optimize(nn_output, corr_label, learning_rate, num_classes, class_weights, loss_name = 'cross_entropy'):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    labels = tf.reshape(corr_label, [-1,num_classes])
    labels = tf.cast(labels, tf.float32)
    labels0 = labels[:,0];
    class_filter = tf.squeeze(tf.where(tf.not_equal(labels0,1)),1)

    logits = tf.reshape(nn_output,(-1,num_classes))

    gt = tf.gather(labels,class_filter)
    prediction = tf.gather(logits,class_filter)


    y_true = tf.math.argmax(gt,axis=-1)
    y_pred = tf.math.argmax(prediction , axis=-1)

    conf_matrix = tf.confusion_matrix(y_true, y_pred, num_classes)

    print_op = tf.print( tf.reduce_min(logits), tf.reduce_max(logits))
    
    print ('Using loss = ', loss_name)
    if (loss_name == 'dice'):
        loss = dice_loss(labels, logits, 1.) #class_weights) #,n,d
    else:
        #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.nn.weighted_cross_entropy_with_logits(gt, prediction, pos_weight= class_weights)
        #loss = loss * class_weights
        loss = tf.reduce_mean(loss)


    #ce = ce_loss(labels, logits, class_weights) #,n,d

    return loss, conf_matrix, print_op #, n, d
         
