import tensorflow.compat.v1 as tf
from Images import ImagesController
from numpy import isnan
from fractions import Fraction
from datetime import datetime

tf.disable_eager_execution()


class ConvNetwork:
    RUN_TRAIN = 1
    RUN_TEST = 2

    CONST_PADDING = 'SAME'

    def __init__(self, vehicles_train_path, non_vehicles_train_path, vehicles_test_path, non_vehicles_test_path,
                 image_shape: tuple, conv_layers_num_kernels: tuple, conv_kernel_dimensions: tuple,
                 fc_layers_num_neurons: tuple, batch_size: int, train_classes_ratio: Fraction, train_rate: float,
                 train_num_iterate):

        """
        Args:
            conv_layers_num_kernels: A tuple holds the numbers of kernels in the convolution layers, which are the
                primary layers of convolutional nn. The numbers in this tuple has to be ordered by the order of
                corresponding layers, starting with the first conv-layer to the the last one.
            conv_kernel_dimensions: A 2-tuple holds the dimensions of each kernel in convolutional layers.
            fc_layers_num_neurons: A tuple holds the numbers of neurons in the fully-connected layers, which are the
                final layers of convolutional nn. The numbers in this tuple has to be ordered by the order of
                corresponding layers, starting with the first fc-layer to the the last one.
            train_classes_ratio: An object representing the ratio of vehicles-images to non-vehicles-images in the
                training step. A ratio of 2-to-1, e.g., means, that the network will be fed with 2 images of vehicles
                to every single image of non-vehicle.
        """

        self.__batch_size__ = batch_size
        self.__train_classes_ratio__ = train_classes_ratio
        self.__images_cntrl__ = ImagesController(vehicles_train_path, non_vehicles_train_path,
                                                 vehicles_test_path, non_vehicles_test_path, self.__batch_size__,
                                                 self.__train_classes_ratio__)
        self.__image_shape__ = image_shape
        self.__train_num_iterate__ = train_num_iterate
        self.__train_rate__ = train_rate
        self.__num_inputs__ = image_shape[0] * image_shape[1]  # The total number of pixels in each input image
        self.__conv_layers_num_kernels__ = conv_layers_num_kernels
        self.__conv_kernel_dimensions__ = conv_kernel_dimensions
        self.__fc_layers_num_neurons__ = fc_layers_num_neurons

        # A matrix (tensor) of raw input images. the number of columns is self.__num_inputs__,
        # while the number of rows will be given at runtime - the real batch size.
        self.__raw_input_batch__ = tf.placeholder(tf.float32, [None, self.__num_inputs__])

        # The same matrix above when converting it to a tensor of images, each one of the original dimensions of
        # the images: image_shape[0] x image_shape[1] x 1-channel.
        self.__conv_input_batch__ = tf.reshape(self.__raw_input_batch__, [-1, image_shape[0], image_shape[1], 1])

        # A matrix (tensor) of real classes. The values in this matrix at runtime correspond to the images held
        # in the matrices above.
        self.__real__ = tf.placeholder(tf.float32, [None, 1])

        self._init_conv_layers()

        # Given that invoking self._init_conv_layers declared and created the last convolutional layer, the one held
        # in self.__conv_pools__[-1], and that each max-pooling layer reduces its input by 2,
        # the lines following declare and initialize self.__conv_flat_output__ as a flat tensor of
        # the last convolutional layer. This tensor is the input of the fully-connected step of our CNN.
        num_rows_last_conv = int(image_shape[0] / (2 ** len(self.__conv_layers_num_kernels__)))  # First dimension of each one of last kernels. Numbers of rows.
        num_cols_last_conv = int(image_shape[1] / (2 ** len(self.__conv_layers_num_kernels__)))  # Second dimension of each one of last kernels. Numbers of columns.
        num_kernels_last_conv = self.__conv_layers_num_kernels__[-1]
        self.__fc_batch_input_num_cols__ = num_rows_last_conv * num_cols_last_conv * num_kernels_last_conv
        self.__conv_flat_output__ = tf.reshape(self.__conv_pools__[-1], [-1, self.__fc_batch_input_num_cols__])

        self._init_fc_layers()
        self._init_final_layer()
        self.__train_loss__ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.__real__,
                                                                                     logits=self.__final_output__))
        self.__train_update__ = tf.train.GradientDescentOptimizer(self.__train_rate__).minimize(self.__train_loss__)
        self.__sess__ = tf.Session()
        self.__initialized__ = False
        self.__closed__ = False

    def _init_conv_layers(self):
        self.__conv_weights__ = []
        self.__conv_biases__ = []
        self.__conv_outputs__ = []
        self.__conv_activations__ = []
        self.__conv_pools__ = []
        kernel_r = self.__conv_kernel_dimensions__[0]  # First dimension of each kernel. Numbers of rows.
        kernel_c = self.__conv_kernel_dimensions__[0]  # Second dimension of each kernel. Numbers of columns.
        for (i, num) in enumerate(self.__conv_layers_num_kernels__):
            if i == 0:
                prev_dim = 1
                prev_pool = self.__conv_input_batch__
            else:
                prev_dim = self.__conv_layers_num_kernels__[i - 1]
                prev_pool = self.__conv_pools__[i - 1]
            self.__conv_weights__.append(tf.Variable(
                tf.truncated_normal([kernel_r, kernel_c, prev_dim, num], stddev=0.1)))
            self.__conv_biases__.append(tf.Variable(tf.constant(0.1, shape=[num])))
            self.__conv_outputs__.append(
                tf.nn.conv2d(prev_pool, self.__conv_weights__[i], strides=[1, 1, 1, 1],
                             padding=ConvNetwork.CONST_PADDING) + self.__conv_biases__[i])
            self.__conv_activations__.append(tf.nn.relu(self.__conv_outputs__[i]))
            self.__conv_pools__.append(tf.nn.max_pool(self.__conv_activations__[i], ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding=ConvNetwork.CONST_PADDING))

    def _init_fc_layers(self):
        self.__fc_weights__ = []
        self.__fc_biases__ = []
        self.__fc_outputs__ = []
        self.__fc_activations__ = []
        for (i, num) in enumerate(self.__fc_layers_num_neurons__):
            if i == 0:
                prev_dim = self.__fc_batch_input_num_cols__
                prev_output = self.__conv_flat_output__
            else:
                prev_dim = self.__fc_layers_num_neurons__[i - 1]
                prev_output = self.__fc_activations__[i - 1]
            self.__fc_weights__.append(tf.Variable(tf.truncated_normal(shape=[prev_dim, num], stddev=0.1)))
            self.__fc_biases__.append(tf.Variable(tf.truncated_normal(shape=[num], stddev=0.1)))
            self.__fc_outputs__.append(tf.matmul(prev_output, self.__fc_weights__[i]) + self.__fc_biases__[i])
            self.__fc_activations__.append(tf.nn.relu(self.__fc_outputs__[i]))

    def _init_final_layer(self):
        self.__fc_weights__.append(tf.Variable(tf.truncated_normal(shape=[self.__fc_layers_num_neurons__[-1], 1],
                                                                   stddev=0.1)))
        self.__fc_biases__.append(tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1)))
        self.__fc_outputs__.append(tf.matmul(self.__fc_activations__[-1], self.__fc_weights__[-1]) +
                                   self.__fc_biases__[-1])
        self.__final_output__ = tf.nn.sigmoid(self.__fc_outputs__[-1])

    def init_net_session(self):
        if self.__closed__:
            raise Exception("Network has been closed.")
        self.__sess__.run(tf.global_variables_initializer())
        self.__sess__.run(tf.local_variables_initializer())
        self.__initialized__ = True

    def close(self):
        if self.__sess__ is not None:
            self.__sess__.close()
        self.__closed__ = True

    def run(self, run_type: int):
        if self.__closed__:
            raise Exception("Network has been closed.")
        elif not self.__initialized__:
            raise Exception("Network session has not been initialized yet. Call 'init_net_session' method.")

        loss_mean = 0
        loss_queue = []
        curr_loss = 0
        distance_tensor = tf.reduce_mean(tf.abs(self.__real__ - self.__final_output__))
        accuracy_queue = []
        num_true = 0

        if run_type == ConvNetwork.RUN_TRAIN:
            print("\nCNN training start. %s" % datetime.now())
            self.__images_cntrl__.init_train_data()
            num_iterate = self.__train_num_iterate__
            next_batch_getter = self.__images_cntrl__.get_next_train_batch
            fetches = [self.__train_update__, self.__train_loss__, distance_tensor]
            if num_iterate >= 50:
                interval_show_status = num_iterate // 50
            else:
                interval_show_status = 1
            kernel = "({0}, {1})".format(str(self.__conv_kernel_dimensions__[0]),
                                         str(self.__conv_kernel_dimensions__[1]))
            conv_layers = " -> ".join(map(str, self.__conv_layers_num_kernels__))
            fc_layers = str(self.__fc_batch_input_num_cols__) + " -> " + \
                        " -> ".join(map(str, self.__fc_layers_num_neurons__)) + " -> 1"
            classes_ratio = "{0} vehicles / {1} non-vehicles".format(str(self.__train_classes_ratio__.numerator),
                                                                     str(self.__train_classes_ratio__.denominator))
            print("Convolutional layers: each kernel %s, number of kernels %s.\n"
                  "Fully-connected layers: %s.\n"
                  "Training rate %f.\n"
                  "Classes ratio: %s\n"
                  "Batch size %d. Number of iterations %d.\n"
                  "Results are being printed every %d steps.\n"
                  % (kernel, conv_layers, fc_layers, self.__train_rate__, classes_ratio, self.__batch_size__,
                     num_iterate, interval_show_status))
        else:
            print("\nCNN test start. %s\n" % datetime.now())
            self.__images_cntrl__.init_test_data()
            num_iterate = (self.__images_cntrl__.num_test // self.__batch_size__) + \
                          (1 if self.__images_cntrl__.num_test % self.__batch_size__ > 0 else 0)
            next_batch_getter = self.__images_cntrl__.get_next_test_batch
            fetches = [self.__final_output__, self.__train_loss__, distance_tensor]
            interval_show_status = 0

        i = 0
        has_more_images = True
        while i < num_iterate and has_more_images:
            batch_xs, batch_ys = next_batch_getter()
            if batch_xs is None:
                has_more_images = False
            else:
                _, curr_loss, curr_distance = self.__sess__.run(fetches=fetches,
                                                                feed_dict={self.__raw_input_batch__: batch_xs,
                                                                           self.__real__: batch_ys})
                curr_accuracy = True if curr_distance <= 0.02 else False
                if run_type == ConvNetwork.RUN_TRAIN:
                    loss_queue.append(curr_loss)
                    accuracy_queue.append(curr_accuracy)
                    if i >= 10:
                        first_loss = loss_queue.pop(0)
                        first_accuracy = accuracy_queue.pop(0)
                        if first_accuracy:
                            num_true = max(0, num_true - 1)
                    else:
                        first_loss = 0
                    if curr_accuracy:
                        num_true = min(10, num_true + 1)
                    loss_mean = loss_mean - (first_loss / 10) + (curr_loss / 10)

                    if i >= 10 and i % interval_show_status == 0:
                        print("Step %d: loss %0.2f, accuracy %d percents. %s"
                              % (i, loss_mean, num_true * 10, datetime.now()))
                else:
                    loss_mean += (curr_loss / num_iterate)
                    if curr_accuracy:
                        num_true += 1

                if isnan(curr_loss):
                    print("NaN values. Process has been aborted.")
                    break
                i += 1

        if not isnan(curr_loss):
            if run_type == ConvNetwork.RUN_TRAIN:
                print("\nCNN training end. %s\nFinal loss %0.2f. Final accuracy %d percents.\n"
                      % (datetime.now(), loss_mean, num_true * 10))
            else:
                print("\nCNN test end. %s\nFinal loss %0.2f. Final accuracy %0.2f percents.\n"
                      % (datetime.now(), loss_mean, 100 * (num_true / num_iterate)))

    """
    def _init_layers(self):
        prev_num = self.__inner_layers_num_neurons__[0]
        self.__weights__ = [tf.Variable(tf.truncated_normal(shape=[self.__num_inputs__, prev_num], stddev=0.1))]
        self.__biases__ = [tf.Variable(tf.truncated_normal(shape=[prev_num], stddev=0.1))]
        self.__outputs__ = [tf.nn.relu(tf.matmul(self.__raw_input_batch__, self.__weights__[0]) + self.__biases__[0])]
        for i in range(1, self.__num_inner_layers__):
            curr_num = self.__inner_layers_num_neurons__[i]
            self.__weights__.append(tf.Variable(tf.truncated_normal(shape=[prev_num, curr_num], stddev=0.1)))
            self.__biases__.append(tf.Variable(tf.truncated_normal(shape=[curr_num], stddev=0.1)))
            self.__outputs__.append(
                tf.nn.relu(tf.matmul(self.__outputs__[i - 1], self.__weights__[i]) + self.__biases__[i]))
            prev_num = curr_num
        self.__weights__.append(tf.Variable(tf.truncated_normal(shape=[prev_num, 1], stddev=0.1)))
        self.__biases__.append(tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1)))
        self.__outputs__.append(
            tf.matmul(self.__outputs__[self.__num_inner_layers__ - 1], self.__weights__[self.__num_inner_layers__]) +
            self.__biases__[self.__num_inner_layers__])
        self.__output__ = tf.nn.sigmoid(self.__outputs__[self.__num_inner_layers__])
    """

"""
# TEMP
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
x_image = tf.reshape(x, [-1, 28, 28, 1])  # if we had RGB, we would have 3 channels

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # uses moving averages momentum
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
"""
