from scipy.misc import imread
import time
import tensorflow as tf
import numpy as np

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)
    
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('alexnet.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    #test_accuracy = evaluate(X_test, y_test)
    res = tf.arg_max(logits, 1)
    sess.run(res, feed_dict={features:[im1, im2]})
    #print("Test Accuracy = {:.3f}".format(test_accuracy))

print(res)
