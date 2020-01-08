import tensorflow.compat.v1 as tf
from Images import ImagesController
from numpy import isnan
from fractions import Fraction
from datetime import datetime

tf.disable_eager_execution()


class ConvNetwork:
    RUN_TRAIN = 1
    RUN_TEST = 2

    def __init__(self, vehicles_train_path, non_vehicles_train_path, vehicles_test_path, non_vehicles_test_path,
                 image_shape: tuple, conv_layers_num_neurons: tuple, fc_layers_num_neurons: tuple, batch_size: int,
                 train_classes_ratio: Fraction, train_rate: float, train_num_iterate):

        """
        Args:
            conv_layers_num_neurons: A tuple holds the numbers of neurons in the convolution layers, which are the
                primary layers of convolutional nn. The numbers in this tuple has to be ordered by the order of
                corresponding layers, starting with the first conv-layer to the the last one.
            fc_layers_num_neurons: A tuple holds the numbers of neurons in the fully-connected layers, which are the
                final layers of convolutional nn. The numbers in this tuple has to be ordered by the order of
                corresponding layers, starting with the first fc-layer to the the last one.
            train_classes_ratio: An object representing the ratio of vehicles-images to non-vehicles-images in the
                training step. A ratio of 2-to-1, e.g., means, that the network will be fed with TODO
        """

        pass


# TEMP
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
x_image = tf.reshape(x, [-1,28,28,1])  # if we had RGB, we would have 3 channels

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # uses moving averages momentum
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

