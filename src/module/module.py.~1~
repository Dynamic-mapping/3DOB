import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

flags = tf.app.flags
args = flags.FLAGS

def ObjectDetect(inputs, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("ObjectDetect", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='in')
        net_h0 = Conv3d(net_in, args.voxel_filter, (5, 5, 5), (2, 2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h0/conv3d')
        net_h0 = BatchNormLayer(net_h0, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h0/batch_norm')

        net_h1 = Conv3d(net_h0, args.voxel_filter*2, (5, 5, 5), (2, 2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h1/conv3d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h1/batch_norm')

        net_h2 = Conv3d(net_h1, args.voxel_filter*4, (3, 3, 3), (2, 2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h2/conv3d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h2/batch_norm')

        net_h3 = Conv3d(net_h2, args.voxel_filter*4, (3, 3, 3), (1, 1, 1), act=None,
                        padding='SAME', W_init=w_init, name='h3/conv3d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h3/batch_norm')

        # cordinates output
        net_cor = Conv3d(net_h3, 24, (3, 3, 3), (1, 1, 1), act=None,
                        padding='SAME', W_init=w_init, name='cordinates')

        # objectness output
        net_obj = Conv3d(net_h3, 2, (3, 3, 3), (1, 1, 1), act=None,
                        padding='SAME', W_init=w_init, name='class')
        net_obj.outputs = tf.nn.softmax(net_obj.outputs, dim=-1)

    return net_cor, net_obj
