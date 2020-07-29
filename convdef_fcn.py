import tensorflow as tf
import numpy as np

def batch_normal(x, is_training):
    data_bn = tf.contrib.slim.batch_norm(inputs=x, is_training=is_training, updates_collections=None)
    return data_bn

def conv2d(x, weights, biases):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding="SAME") + biases

def pool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding="SAME")
def gloable_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 256, 256, 1],padding="SAME")
def batch_normal(x, is_training):
    data_bn = tf.contrib.slim.batch_norm(inputs=x, is_training=is_training, updates_collections=None)
    return data_bn
def upsampling(x,size):
    x_shape = x.get_shape().as_list()
    upsize = size * x_shape[1]
    upsample = tf.image.resize_images(x, size=(upsize, upsize), method=0)
    return upsample


def single_conv(x,name,weights):
    with tf.variable_scope(name):
        weights1 = tf.get_variable(initializer=weights[0], name="weights1")
        biases1 = tf.get_variable(initializer=weights[1], name="biases1")

        conv1 = conv2d(x, weights1, biases1)
        conv1 = tf.nn.relu(conv1)
        return conv1
def fuse_conv(x,name,weights,biases):
    with tf.variable_scope(name):
        weights1 = tf.get_variable(initializer=weights, name="weights1")
        biases1 = tf.get_variable(initializer=biases , name="biases1")

        conv1 = conv2d(x, weights1, biases1)
        conv1 = tf.nn.relu(conv1)
        return conv1
def conv(x,name,input_dim,out_dim):
    with tf.variable_scope(name):
        weights1 = tf.get_variable(shape=[3, 3, input_dim, out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                   , name="weights1")
        biases1 = tf.get_variable(shape=[out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                  , name="biases1")
        conv1 = conv2d(x, weights1, biases1)
        conv1 = tf.nn.relu(conv1)
        return conv1
def conv1(x,name,input_dim,out_dim):    #add  bn
    with tf.variable_scope(name):
        weights1 = tf.get_variable(shape=[3, 3, input_dim, out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                   , name="weights1")
        biases1 = tf.get_variable(shape=[out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                  , name="biases1")
        conv1 = conv2d(x, weights1, biases1)
        conv2 = batch_normal(conv1,is_training= True )
        conv3 = tf.nn.relu(conv2)
        return conv3
def conv2(x,name,input_dim,out_dim):    #sigmoid
    with tf.variable_scope(name):
        weights1 = tf.get_variable(shape=[3, 3, input_dim, out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                   , name="weights1")
        biases1 = tf.get_variable(shape=[out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                  , name="biases1")
        conv1 = conv2d(x, weights1, biases1)
        conv1 = tf.nn.sigmoid(conv1)
        return conv1

def single_deconv(x,name,input_dim,out_dim,size):
    with tf.variable_scope(name):

        weights1 = tf.get_variable(shape=[3, 3, input_dim, out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                   , name="weights1")
        biases1 = tf.get_variable(shape=[out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                  , name="biases1")
        x_shape = x.get_shape().as_list()
        upsize = size * x_shape[1]
        upsample = tf.image.resize_images(x,size=(upsize,upsize),method= 0)
        conv1_1 = conv2d(upsample,weights1 ,biases1)
        conv1 = tf.nn.relu(conv1_1)
        return conv1

def pspconv(x,name,input_dim,out_dim):
    with tf.variable_scope(name):
        weights1 = tf.get_variable(shape=[1, 1, input_dim, out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                   , name="weights1")
        biases1 = tf.get_variable(shape=[out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                  , name="biases1")
        conv1 = conv2d(x, weights1, biases1)
        conv1 = tf.nn.relu(conv1)
        return conv1
def conv_7(x,name,input_dim,out_dim):
    with tf.variable_scope(name):
        weights1 = tf.get_variable(shape=[8, 8, input_dim, out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                   , name="weights1")
        biases1 = tf.get_variable(shape=[out_dim], initializer=tf.random_normal_initializer(mean= 0,stddev= 0.01)
                                  , name="biases1")
        conv1 = conv2d(x, weights1, biases1)
        conv1 = tf.nn.relu(conv1)
        return conv1


