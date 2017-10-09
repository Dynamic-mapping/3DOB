#!/usr/bin/env python
from __future__ import division

import os
import sys
import scipy.misc
import pprint
import time
import json
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from random import shuffle
from six.moves import xrange
from collections import namedtuple
from glob import glob
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

# author designed package
import src.module.module as module
from src.util.util import *

class Net(object):
    #######################################################
    ##               Network configuration               ##
    #######################################################
    def __init__(self, sess, args):

        self.sess = sess
        self.summary = tf.summary
        
        # Configure the point clouds loader
        #cloud = pcl2.create_cloud_xyz32(header, dataset.load_scan(id)[:,:-1])
        self.getScans = getScans

        # Configure model
        self.is_train = args.is_train 

        # Configure Network module
        self.ObjectDetect = module.ObjectDetect

        # Construct network
        self._build_model(args)

        # Extract variables
        self._extract_variable()

        # Configre optimizer
        self._set_optimizer(args)
        
        # Configre log file
        self.log_dir = os.path.join(args.log_dir, args.method, args.log_name)
        if not os.path.exits(log_dir):
            os.makedirs(log_dir)

    def _build_model(self, args):
        
        # set input variables
        self.d_pc    = tf.placeholder(tf.float32, [args.batch_size, \
                                                   800, \
                                                   800, \
                                                   40, \
                                                   1], name="inputPCD")
        self.d_cor = tf.placeholder(tf.float32, [args.batch_size, 6], name="cor")
        self.d_obj = tf.placeholder(tf.float32, [args.batch_size, ], name="obj")

        # classify
        self.n_cor, self.n_obj = self.ObjectDetect(self.d_pc, is_train=True, reuse=False)

        # loss for objectdetection
        self.loss_cor = tf.reduce_mean(self.n_cor.outputs, self.d_cor)
        self.loss_obj = tf.reduce_mean(self.n_obj.outputs, self.d_obj)
        self.loss_sum = tf.add(self.loss_cor + self.loss_obj)
        
        # Make Summary
        with tf.name_scope('3DoB'):
            self.summ_cor  = tf.summary.scalar('cor', self.loss_cor)
            self.summ_obj  = tf.summary.scalar('obj', self.loss_obj)

        self.summ_merge = tf.summary.merge_all()

    def _extract_variable(self):
        self.var_obj = tl.layers.get_variable_with_name("ObjectDetect", True, True)

    def _set_optimizer(self, args):
        self.optim_class = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                   .minimize(self.loss_sum, var_list=self.var_obj)

    #######################################################
    ##             Network training and testing          ##
    #######################################################
    def train(self, args):
        
        # Initial layer's variables
        tl.layers.initialize_global_variable(self.sess)
        if args.restore == True:
            self.loadParam(args)
            print ("[*] load network done!")
        else:
            print ("[!] Initial network done!")

        # Initial global variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # Enable tf summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # Load Data files
        data_dir = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        data_files = []
        for data_name in data_dir:
            read_path = os.path.join("../data/", data_name, "pcd/*.bin")
            data_file = glob(read_path)
            data_files = data_files + data_file

        print (len(data_files))
        
        # Main loop for training
        self.iter_counter = 0

        for epoch in range(0, args.epoches):
            
            # shuffle data
            shuffle(data_files)
            print("[*] Dataset shuffled!")

            # load image data
            batch_idxs = min(len(data_files), args.train_size) // args.batch_size

            for idx in xrange(0, batch_idxs):
                
                # Get datas
                batch_files  = data_files[idx*args.batch_size:(idx+1)*args.batch_size]
                batch_pcs    = self.getScans(batch_files)
                batch_ths    = np.random.normal(loc=0.0, scale=1.0, \
                                                size=(args.batch_size, args.code_dim)).astype(np.float32)

                start_time   = time.time()
                feed_dict    = {self.d_pc: batch_pcs, self.d_truth: batch_truth}
                feed_dict.update(self.n_clf.all_drop)

                err, _ = self.sess.run([self.loss_clf, self.optim_clf], feed_dict=feed_dict)
                
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f"  % \
                      (epoch, args.epoch, idx, batch_idxs, time.time() - start_time, err))
                sys.stdout.flush()

                self.iter_counter += 1

                if np.mod(self.iter_counter, args.save_step) == 0:
                    self.saveParam(args)
                    print("[*] Saving checkpoints SUCCESS!")


        # Shutdown writer
        self.writer.close()
