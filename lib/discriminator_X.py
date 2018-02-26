"""
this is the GAN discriminator mudule
"""
import tensorflow as tf
import tflib as tflib
import tflib.ops.linear
import tflib.ops.conv1d

DIM = 512
SEQ_LEN = 26

def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = tflib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 3, output)
    output = tf.nn.relu(output)
    output = tflib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 3, output)
    return inputs + (0.3*output)


def discriminator_X(inputs):
    output = tf.transpose(inputs, [0,2,1])
    output = tflib.ops.conv1d.Conv1D('discriminator_x.Input',200, DIM, 1, output)
    output = ResBlock('discriminator_x.1', output)
    output = ResBlock('discriminator_x.2', output)
    output = ResBlock('discriminator_x.3', output)
    output = ResBlock('discriminator_x.4', output)
    #output = ResBlock('Discriminator.5', output)

    output = tf.reshape(output, [-1, SEQ_LEN*DIM])
    output = tflib.ops.linear.Linear('discriminator_x.Output', SEQ_LEN*DIM, 1, output)
    return tf.squeeze(output,[1])