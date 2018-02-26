"""
basic module of tensorflow
"""
import numpy as np
import tensorflow as tf
from functools import reduce

def linear_matmul(inputs,weight):
    hid_dim = weight.get_shape().as_list()[0]
    origin_shape = inputs.get_shape().as_list()
    inputs = tf.reshape(inputs,[-1,hid_dim])
    outputs = tf.matmul(inputs,weight)
    outputs = tf.reshape(outputs,origin_shape[:-1] + [-1])
    return outputs


def batch_to_time_major(inputs):
    inputs = tf.split(inputs,  num_or_size_splits=inputs.get_shape().as_list()[1] ,axis=1)
    inputs = [tf.squeeze(e,axis=1) for e in inputs]
    return inputs


def conv1d( inputs,
            output_dim,
            filter_width,
            name,
            stride=1,
            padding='SAME',
            initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=tf.nn.elu):

    kernel_shape = [filter_width,inputs.get_shape()[-1],output_dim]

    weight = tf.get_variable(name=name+'_w', shape=kernel_shape, dtype=tf.float32, initializer=initializer)
    bias = tf.get_variable(name=name+'_b', shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    outputs = tf.nn.conv1d(inputs , weight, stride=stride, padding=padding)
    outputs = tf.nn.bias_add(outputs, bias)

    return activation_fn(outputs)
    


def TDNN(inputs,
         feature_maps = [10, 10, 10, 10, 10],
         kernels = [2,3,4,5,6]):
    """
    Time-delayed Nueral Network
    https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/TDNN.py
    """
    outputs = []
    for index,kernel_dim in enumerate(kernels):
        batch_size = inputs.get_shape()[0]
        reduced_length = inputs.get_shape()[1] - kernel_dim + 1

        conv_out = conv1d(inputs, feature_maps[index], kernel_dim, 'cnn_'+str(kernel_dim), padding='VALID')
        conv_out = tf.expand_dims(conv_out,axis=2)
        #kernel_out = tf.nn.max_pool(conv_out, [1, reduced_length, 1, 1], [1, 1, 1, 1], 'VALID')
        kernel_out = tf.nn.avg_pool(conv_out,[1,reduced_length,1,1],[1,1,1,1],'VALID')
        outputs.append(tf.squeeze(kernel_out))

    return tf.concat(outputs,1)
    """
        conv_out = conv1d(inputs, feature_maps[index], kernel_dim, 'cnn_'+str(kernel_dim), stride=2, padding='VALID')
        outputs.append(tf.reshape(conv_out,[batch_size,-1]))
    return tf.concat(outputs,1)
    """



def sample3D(probability):
    """
    probability should be a 3-d tensor, [batch_size,distribution_size]
    it will generate a batch size array
    """
    shape = probability.get_shape().as_list()
    probability = tf.reshape(probability,[-1,shape[-1]])
    return tf.reshape(tf.multinomial(tf.log(probability),1),[shape[0],shape[1]])

def sample2D(probability):
    return tf.squeeze(tf.multinomial(tf.log(probability),1))

def get_seq_len(seq):
    #sequence shape: batch_size*sequence_len
    seq = tf.cast(seq,tf.float32)
    seq = tf.sign(seq)
    seq_len = tf.reduce_sum(seq,axis=1)
    return tf.stop_gradient(seq_len)