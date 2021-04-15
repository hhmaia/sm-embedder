# coding: utf-8

import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform

class CenterLoss(tf.losses.Loss):

    def __init__(self,
            num_classes,
            alpha=0.5,
            centers_init=None,
            name='centerloss'):

        super().__init__(name=name)
        self.num_classes = num_classes 
        self.alpha = alpha


    def call(self, features, labels):
        return



    def get_center_loss(features, labels, alpha, num_classes):
        '''
        adapted from:
            https://github.com/EncodeTS/TensorFlow_Center_Loss
        '''

        len_features = features.get_shape()[1]
        centers = tf.Variable(
                initial_value=GlorotUniform()([num_classes, len_features]),
                dtype=tf.float32,
                trainable=False,
                name='centers')

        labels = tf.reshape(labels, [-1])
        
        centers_batch = tf.gather(centers, labels)
        loss = tf.nn.l2_loss(features - centers_batch)
        
        diff = centers_batch - features
        
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        
        # centers_update_op = tf.scatter_sub(centers, labels, diff)
        centers_update_op = centers.scatter_sub(tf.IndexedSlices(diff, labels))
        
        return loss, centers, centers_update_op

