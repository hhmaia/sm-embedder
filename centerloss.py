#_*_ coding:utf8 _*_
import functools
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform

'''
Original code from:
    https://github.com/keras-team/keras/issues/6929
'''
def _center_loss_func(features, labels, alpha, num_classes,
                      centers, feature_dim):
    assert feature_dim == features.get_shape()[1]    
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.int32)
    centers_batch = tf.gather(centers, labels)
    diff = (1 - alpha) * (centers_batch - features)
    # centers = tf.scatter_sub(centers, labels, diff)
    centers = centers.scatter_sub(tf.IndexedSlices(diff, labels))
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss


def get_center_loss(alpha, num_classes, feature_dim):
    """Center loss based on the paper "A Discriminative 
       Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """    
    # Each output layer use one independed center: scope/centers
    # centers = tf.zeros([num_classes, feature_dim])
    centers = tf.Variable(
            #initial_value=GlorotUniform()([num_classes, feature_dim]),
            initial_value=tf.zeros([num_classes, feature_dim]),
            dtype=tf.float32,
            trainable=False,
            name='centers')

    @functools.wraps(_center_loss_func)
    def center_loss(y_true, y_pred):
        return _center_loss_func(y_pred, y_true, alpha, 
                                 num_classes, centers, feature_dim)
    return center_loss
