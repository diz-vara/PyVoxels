import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import tensorflow.contrib.slim as slim
import time
import labels_vox as lbl
import sys


import pickle

from read_ontology import read_ontology
from train import *

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--dataset', type=str, default="UK-4", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=768, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
parser.add_argument('--model_name', type=str, default="OS-44-new", help='Model name')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--prefix', type=str, default="", help='optional prefix to separate n/w')
parser.add_argument('--suffix', type=str, default="", help='optional suffix to separate n/w')
parser.add_argument('--loss', type=str, default="crossentropy", help='Loss function [crossentropy]')
args = parser.parse_args()




# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
#if not tf.test.gpu_device_name():
#    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
#else:
#    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

#%%
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    model = tf.get_default_graph()
    
    input_image = model.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = model.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = model.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = model.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = model.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return input_image, keep_prob,layer3_out, layer4_out, layer7_out

#tests.test_load_vgg(load_vgg, tf)

#%%
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, keep_prob, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    # 1x1 convolution of L7 ( 5 x 18 )
    l3_depth = vgg_layer3_out.shape[3].value
    l4_depth = vgg_layer4_out.shape[3].value
    l7_depth = vgg_layer7_out.shape[3].value


    layer7_conv = tf.layers.conv2d(vgg_layer7_out, l7_depth, 1,
                                padding = 'same',
                                kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                activation=tf.nn.relu,
                                name='layer7_conv1')
                                
    layer7_drop = tf.nn.dropout(layer7_conv, keep_prob=keep_prob)                            
    # upscale to 10 x 36
    layer7_up = tf.layers.conv2d_transpose(layer7_drop, l4_depth, 4,
                                             strides = (2,2),
                                             padding = 'same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             activation=tf.nn.relu,
                                             name = 'layer7_up')

    # add upscaled L7
    layer4_concat = tf.concat([vgg_layer4_out, layer7_up], -1, name = 'layer4_concat')

                                
                                
    # 1x1 convolution of L4 ( 10 x 36 )
    layer4_conv = tf.layers.conv2d(layer4_concat, l4_depth, (1,1),
                                padding = 'same',
                                kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                activation=tf.nn.relu,
                                name = 'layer4_conv1')

    #layer4_drop = tf.nn.dropout(layer4_conv, keep_prob=keep_prob)                            


    # upscale to 20 x 72
    layer4_up = tf.layers.conv2d_transpose(layer4_conv, l3_depth, 4,
                                             strides = (2,2),
                                             padding = 'same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             activation=tf.nn.relu,
                                             name = 'layer4_up')

    # add upscaled L4                                
    layer3_concat = tf.concat([vgg_layer3_out, layer4_up], -1, name = 'layer3_concat')



    # # 1x1 convolution of L3 ( 20 x 72)
    # layer3_conv = tf.layers.conv2d(layer3_concat, l3_depth*4, (1,1),
    #                             padding = 'same',
    #                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
    #                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
    #                             activation=tf.nn.relu,
    #                             name = 'layer_3_conv1')

    
    layer3_up1 = tf.layers.conv2d_transpose(layer3_concat, l3_depth*4  , 5,
                                             strides = (2,2),
                                             padding = 'same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             name = 'layer3_up1')


    #layer3_drop = tf.nn.dropout(layer3_up1, keep_prob=keep_prob)
    
    layer3_up2 = tf.layers.conv2d_transpose(layer3_up1, l3_depth*2, 5,
                                             strides = (2,2),
                                             padding = 'same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             name = 'layer3_up2')

    
    # upscale to original 160 x 572
    layer3_up = tf.layers.conv2d_transpose(layer3_up2, num_classes, 5,
                                             strides = (2,2),
                                             padding = 'same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             name = 'layer3_up')
                                
    return layer3_up
#tests.test_layers(layers)


#%%
tf.reset_default_graph();

def run():

    tf.reset_default_graph()

    timestamp = time.strftime("%Y%m%d_%H%M%S");


    ontology, colors = read_ontology(args.dataset + '/Ontology F8.csv')

    try:
        class_weights = pickle.load(open(args.dataset + '/class_weights.p','rb'))
        print('loaded weights for ', len(class_weights), ' classes')
    except:
        class_weights = 1.
        pass

    num_classes = len(ontology)

    image_shape=(args.crop_height, args.crop_width) #NB! Now it is height x width !!!

    full_model_name = args.prefix + args.model_name + args.suffix;
    model_path = args.dataset + '/nets/' + full_model_name;

    data_dir = args.dataset + '/../data/'

    timestamp = time.strftime("%Y%m%d_%H%M%S");


    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    #builder = tf.saved_model.builder.SavedModelBuilder(model_path);

    config = tf.ConfigProto(
       gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.92),
       device_count = {'GPU': 1}
    )

    lr_file = args.dataset + '/nets/' + full_model_name + '_lr.txt';
    open(lr_file,'w').write(str(args.learning_rate))

    with tf.Session(config=config) as sess:

        vgg_path = os.path.join(data_dir, 'vgg')
        
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes],
                                       name = 'correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')                                       
    
        image_in, keep_prob,l3_o, l4_o, l7_o = load_vgg(sess, vgg_path);

        print('layer3=',l3_o.shape, ', layer4=', l4_o.shape, ', layer7=',l7_o.shape)
        
        nn_output = layers(l3_o, l4_o, l7_o, keep_prob, num_classes)
    
        loss, conf_matrix, print_op  = optimize(nn_output, correct_label,  #, n, d, 
                                          learning_rate, num_classes, class_weights, args.loss)
        
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) 
        #optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.8) 

        train_op = optimizer.minimize(loss, name='train_op')

        g_vars = tf.global_variables()
        sess.run(tf.global_variables_initializer())
    
        train_nn(sess, full_model_name, args.epochs, args.batch_size, 
                 args.dataset, image_shape, ontology,
                 train_op, loss, conf_matrix, print_op, #n,d,
                 tf.train.Saver(),
                 image_in, correct_label, nn_output,
                 keep_prob, learning_rate)                                          
    

    
    
        
        #print('Saving net:')
        #builder.add_meta_graph_and_variables(sess,
        #                                     [tf.saved_model.tag_constants.SERVING])
        
        writer = tf.summary.FileWriter('/tmp/log/tf', sess.graph)
        writer.close()
        # OPTIONAL: Apply the trained model to a video
    print('AFTER sesion')
    builder.save()
    
    
    
    
    



if __name__ == '__main__':
    run()
