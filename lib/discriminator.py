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


def discriminator(inputs):
    output = tf.transpose(inputs, [0,2,1])
    output = tflib.ops.conv1d.Conv1D('discriminator.Input',200, DIM, 1, output)
    output = ResBlock('discriminator.1', output)
    output = ResBlock('discriminator.2', output)
    output = ResBlock('discriminator.3', output)
    output = ResBlock('discriminator.4', output)
    #output = ResBlock('Discriminator.5', output)

    output = tf.reshape(output, [-1, SEQ_LEN*DIM])
    output = tflib.ops.linear.Linear('discriminator.Output', SEQ_LEN*DIM, 1, output)
    return tf.squeeze(output,[1])
"""
def discriminator(inputs):
	
	batch_size = inputs.get_shape().as_list()[0]

	output = conv1d(inputs , 512, 1, 'discriminator_inputs')
	output = res1D(output,'res1')
	output = res1D(output,'res2')
	output = res1D(output,'res3')
	output = res1D(output,'res4')

	output = tf.reshape(output,[batch_size,-1])

	with tf.variable_scope("output") as scope:
		output = tf.contrib.layers.linear(output, 1, scope=scope)

	return tf.squeeze(output,[1])
"""