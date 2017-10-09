import tensorflow as tf
import numpy as np

def Param():
    flags = tf.app.flags
    
    ## Param
    flags.DEFINE_integer("epoch",              40,           "Epoch to train [40]")
    flags.DEFINE_integer("c_epoch",            0,            "current Epoch")
    flags.DEFINE_integer("voxel_filter",       16,           "voxel filter")
    args.voxel_filter



    return flags.FLAGS


