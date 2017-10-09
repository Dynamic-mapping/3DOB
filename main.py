import os
import sys
import tensorflow as tf
from parameters import Param

from 3dob.model.model_01 import Net


# Obtain parameters
args = Param

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main(_):

    Net_model = Net

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth=True


    with tf.Session(config=config) as sess:
        model = Net_model(sess, args)

if __name__ == '__main__':
    tf.app.run()
