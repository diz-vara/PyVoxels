import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import tensorflow.contrib.slim as slim
import time


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
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
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
                                
    # upscale to 10 x 36
    layer7_up = tf.layers.conv2d_transpose(layer7_conv, l4_depth, 4,
                                             strides = (2,2),
                                             padding = 'same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             activation=tf.nn.relu,
                                             name = 'layer7_up')

    # add upscaled L7
    layer4_add = tf.add(vgg_layer4_out, layer7_up, name = 'layer4_add')

                                
                                
    # 1x1 convolution of L4 ( 10 x 36 )
    layer4_conv = tf.layers.conv2d(layer4_add, l4_depth, (1,1),
                                padding = 'same',
                                kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                activation=tf.nn.relu,
                                name = 'layer4_conv1')

    
    # upscale to 20 x 72
    layer4_up = tf.layers.conv2d_transpose(layer4_conv, l3_depth, 4,
                                             strides = (2,2),
                                             padding = 'same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             activation=tf.nn.relu,
                                             name = 'layer4_up')

    # add upscaled L4                                
    layer3_add = tf.add(vgg_layer3_out, layer4_up, name = 'layer3_add')

    # 1x1 convolution of L3 ( 20 x 72)
    layer3_conv = tf.layers.conv2d(layer3_add, l3_depth, (1,1),
                                padding = 'same',
                                kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                activation=tf.nn.relu,
                                name = 'layer_3_conv1')
    # upscale to original 160 x 572
    layer3_up = tf.layers.conv2d_transpose(layer3_conv, num_classes, 16,
                                             strides = (8,8),
                                             padding = 'same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             name = 'layer3_up')
                                
    return layer3_up
#tests.test_layers(layers)

#%%

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    result = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits = result,
                                                            name = "cross-ent")
    loss = tf.reduce_mean(cross_entropy);
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) 

    train_op = optimizer.minimize(loss)
    
    return result, train_op, loss
#tests.test_optimize(optimize)

#%%

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver();

    #lr = sess.run(learning_rate)
    #merged = tf.summary.merge_all()
    lr = 1e-4
    min_loss = 1e9
    for epoch in range (epochs):
        print ('epoch {}  '.format(epoch))
        print(" LR = {:f}".format(lr))     
        for image, label in get_batches_fn(batch_size):
            summary, loss = sess.run([train_op, cross_entropy_loss],
                                     feed_dict={input_image:image, 
                                                correct_label:label,
                                     keep_prob:0.5, learning_rate:lr})
        #writer.add_summary(summary, epoch)                          
        lr = lr * 0.9                            
        print(" Loss = {:g}".format(loss))     
        print()                        
        if (loss < min_loss):
            print("saving at step {:d}".format(epoch))     
            min_loss = loss;
            saver.save(sess, '/media/D/DIZ/Datasets/KITTI/Segmentation/net/my2-net',
                       global_step=epoch)
    
#tests.test_train_nn(train_nn)

#%%
tf.reset_default_graph();

def run():
    num_classes = len(labels_diz)
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    timestamp = time.strftime("%Y%m%d_%H%M%S");

    export_dir = './exports/' + timestamp;

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir);

    config = tf.ConfigProto(
       gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8),
       device_count = {'GPU': 1}
    )


    with tf.Session(config=config) as sess:
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function('/media/D/DIZ/Datasets/KITTI/Segmentation',
                                                   image_shape, num_classes)
    
    
        epochs = 50
        batch_size = 8
        
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes],
                                       name = 'correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')                                       
    
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
    
    
        image_in, keep_prob,l3_o, l4_o, l7_o = load_vgg(sess, vgg_path);
        nn_output = layers(l3_o, l4_o, l7_o, num_classes)
    
        logits, train_op, loss = optimize(nn_output, correct_label, 
                                          learning_rate, num_classes)
    
        print('training')
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 loss, image_in, correct_label, keep_prob, learning_rate)                                          
    
    
    
        # TODO: Save inference data using helper.save_inference_samples
        print('Saving results')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, 
                                      logits, keep_prob, image_in)
    
        
        print('Saving net:')
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.SERVING])
        
        writer = tf.summary.FileWriter('/tmp/log/tf', sess.graph)
        writer.close()
        # OPTIONAL: Apply the trained model to a video
    print('AFTER sesion')
    builder.save()
    
    
    
    
    



if __name__ == '__main__':
    run()
