#!/usr/bin/python
import datetime as dt
import glob
import os
from collections import namedtuple
import tensorflow as tf
from tensorlayer.layers import *


import numpy as np
__author__ = "maxtom"
__email__  = "hitmaxtom@gmail.com"

def getScans(velo_files):
    """Helper method to parse velodyne binary files into a list of scans."""
    scan_list = []
    for filename in velo_files:
        scan = np.fromfile(filename, dtype=np.float32)
        scan_list.append(scan.reshape((-1, 4)))

    return scan_list

def getScan(velo_file):
    """Helper method to parse velodyne binary files into a list of scans."""
    scan = np.fromfile(velo_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


##=================================== function for 3D Layers ================================##
##=================================== function for 3D Layers ================================##
##=================================== function for 3D Layers ================================##

## Conv 3D
def Conv3d(net, n_filter=32, filter_size=(5, 5, 5), strides=(1, 1, 1), act = None,
           padding='SAME', W_init = tf.truncated_normal_initializer(stddev=0.02), b_init = tf.constant_initializer(value=0.0),
           W_init_args = {}, b_init_args = {}, use_cudnn_on_gpu = None, data_format = None,name ='conv3d',):

    assert len(strides) == 3, "len(strides) should be 3, Conv3d and Conv3dLayer are different."
    if act is None:
        act = tf.identity

    net = Conv3dLayer(net,
                      act = act,
                      shape = [filter_size[0], filter_size[1], filter_size[2], \
                               int(net.outputs.get_shape()[-1]), n_filter],  # 32 features for each 5x5 patch
                      strides = [1, strides[0], strides[1], strides[2], 1],
                      padding = padding,
                      W_init = W_init,
                      W_init_args = W_init_args,
                      b_init = b_init,
                      b_init_args = b_init_args,
                      name = name)
    return net

## Deconv 3D
def DeConv3d(net, n_out_channel = 32, filter_size=(5, 5, 5),
             out_size = (30, 30, 30), strides = (2, 2, 2), padding = 'SAME', batch_size = None, act = None,
             W_init = tf.truncated_normal_initializer(stddev=0.02), b_init = tf.constant_initializer(value=0.0),
             W_init_args = {}, b_init_args = {}, name ='decnn3d'):

    assert len(strides) == 3, "len(strides) should be 3, DeConv3d and DeConv3dLayer are different."
    if act is None:
        act = tf.identity
    if batch_size is None:
        batch_size = tf.shape(net.outputs)[0]

    net = DeConv3dLayer(layer = net,
                        act = act,
                        shape = [filter_size[0], filter_size[1], filter_size[2], n_out_channel, int(net.outputs.get_shape()[-1])],
                        output_shape = [batch_size, int(out_size[0]), int(out_size[1]), int(out_size[2]), n_out_channel],
                        strides = [1, strides[0], strides[1], strides[2], 1],
                        padding = padding,
                        W_init = W_init,
                        b_init = b_init,
                        W_init_args = W_init_args,
                        b_init_args = b_init_args,
                        name = name)
    return net
