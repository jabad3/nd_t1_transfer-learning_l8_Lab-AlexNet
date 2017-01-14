from scipy.misc import imread
import time
import tensorflow as tf
from alexnet import AlexNet
import numpy as np

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

def modified(features):
    resized = tf.image.resize_images(features, (227, 227))

    # TODO: pass placeholder as first argument to `AlexNet`.
    fc7 = AlexNet(resized, feature_extract=True)

    # NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
    # past this point, keeping the weights before and up to `fc7` frozen.
    # This also makes training faster, less work to do!
    fc7 = tf.stop_gradient(fc7)

    # TODO: Add the final layer for traffic sign classification.
    shape = (fc7.get_shape().as_list()[-1], nb_classes)
    fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
    fc8b = tf.Variable(tf.zeros(nb_classes))
    logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
    return logits


features = tf.placeholder(tf.float32, (None, 32, 32, 3))
logits = modified(features)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    new_saver = tf.train.import_meta_graph('alexnet.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    res = tf.arg_max(logits, 1)
    sess.run(res, feed_dict={features:[im1, im2]})


print(res)
