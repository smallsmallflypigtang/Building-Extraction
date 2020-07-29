
import tensorflow as tf

def conv2d(x, weights, biases):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding="SAME") + biases

def conv(x,name,input_dim,out_dim):
    with tf.variable_scope(name):
        weights1 = tf.get_variable(shape=[1, 1, input_dim, out_dim], initializer=tf.glorot_normal_initializer()
                                   , name="weights1")
        biases1 = tf.get_variable(shape=[out_dim], initializer=tf.glorot_normal_initializer()
                                  , name="biases1")
        conv1 = conv2d(x, weights1, biases1)
        conv1 = tf.nn.relu(conv1)
        return conv1

#x = tf.placeholder(tf.float32,shape= [batch_size ,16,16,512])


def non_local2(x,h,inputdim):
    outdim = int(inputdim / 2)
    a = conv(x, name='a2', input_dim=inputdim , out_dim=outdim )
    a = tf.reshape(a, shape=[-1, outdim ])
    print('a' + str(a.shape))
    b = conv(x, name='b2', input_dim=inputdim , out_dim=outdim )
    b = tf.reshape(b, shape=[-1, outdim ])
    b1 = tf.transpose(b)
    print('b1' + str(b1.shape))
    c = conv(x, name='c2', input_dim=inputdim , out_dim=outdim )
    c = tf.reshape(c, shape=[-1, outdim ])
    print('c' + str(c.shape))

    d = tf.matmul(a, b1)
    print('d' + str(d.shape))

    d2 = tf.nn.softmax(d, axis=-1)
    f = tf.matmul(d2, c)
    print('f' + str(f.shape))
    f1 = tf.reshape(f, shape=[-1, h, h,outdim ])
    print('f1' + str(f1.shape))
    e = conv(f1, name='e2', input_dim=outdim , out_dim=inputdim)
    z = x + e
    print('z' + str(z.shape))

    return z

def non_local3(x,h,inputdim):
    outdim = int(inputdim / 2)
    a = conv(x, name='a3', input_dim=inputdim , out_dim=outdim )
    a = tf.reshape(a, shape=[-1, outdim ])
    print('a' + str(a.shape))
    b = conv(x, name='b3', input_dim=inputdim , out_dim=outdim )
    b = tf.reshape(b, shape=[-1, outdim ])
    b1 = tf.transpose(b)
    print('b1' + str(b1.shape))
    c = conv(x, name='c3', input_dim=inputdim , out_dim=outdim )
    c = tf.reshape(c, shape=[-1, outdim ])
    print('c' + str(c.shape))

    d = tf.matmul(a, b1)
    print('d' + str(d.shape))

    d3 = tf.nn.softmax(d, axis=-1)
    f = tf.matmul(d3, c)
    print('f' + str(f.shape))
    f1 = tf.reshape(f, shape=[-1, h, h,outdim ])
    print('f1' + str(f1.shape))
    e = conv(f1, name='e3', input_dim=outdim , out_dim=inputdim)
    z = x + e
    print('z' + str(z.shape))

    return z




