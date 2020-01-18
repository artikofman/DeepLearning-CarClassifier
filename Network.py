import tensorflow.compat.v1 as tf
from Images import ImagesController
from numpy import isnan
from fractions import Fraction
from datetime import datetime

tf.disable_eager_execution()


class Network:
    RUN_TRAIN = 1
    RUN_TEST = 2

    def __init__(self, vehicles_train_path, non_vehicles_train_path, vehicles_test_path, non_vehicles_test_path,
                 image_shape: tuple, inner_layers_num_neurons: tuple, batch_size: int, train_classes_ratio: Fraction,
                 train_rate: float, train_num_iterate, train_num_show_status=50):
        self.__batch_size__ = batch_size
        self.__train_classes_ratio__ = train_classes_ratio
        self.__images_cntrl__ = ImagesController(vehicles_train_path, non_vehicles_train_path,
                                                 vehicles_test_path, non_vehicles_test_path, self.__batch_size__,
                                                 self.__train_classes_ratio__)
        self.__image_shape__ = image_shape
        self.__train_num_iterate__ = train_num_iterate
        self.__train_rate__ = train_rate
        self.__num_inputs__ = image_shape[0] * image_shape[1]
        self.__inner_layers_num_neurons__ = inner_layers_num_neurons
        self.__num_inner_layers__ = len(self.__inner_layers_num_neurons__)
        self.__train_num_show_status__ = train_num_show_status
        self.__input__ = tf.placeholder(tf.float32, [None, self.__num_inputs__])
        self.__real__ = tf.placeholder(tf.float32, [None, 1])
        self._init_layers()
        # self.__loss__ = tf.reduce_mean(-self.__real__ * tf.log(self.__output__) + (1 - self.__real__) * tf.log(1 -
        # self.__output__))
        self.__loss__ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.__real__,
                                                                               logits=self.__output__))
        self.__update__ = tf.train.GradientDescentOptimizer(self.__train_rate__).minimize(self.__loss__)
        self.__sess__ = tf.Session()
        self.__initialized__ = False
        self.__closed__ = False

    def _init_layers(self):
        prev_num = self.__inner_layers_num_neurons__[0]
        self.__weights__ = [tf.Variable(tf.truncated_normal(shape=[self.__num_inputs__, prev_num], stddev=0.1))]
        self.__biases__ = [tf.Variable(tf.truncated_normal(shape=[prev_num], stddev=0.1))]
        self.__outputs__ = [tf.nn.relu(tf.matmul(self.__input__, self.__weights__[0]) + self.__biases__[0])]
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

    def init_net_session(self):
        if self.__closed__:
            raise Exception("Network has been closed.")
        self.__sess__.run(tf.global_variables_initializer())
        # self.__sess__.run(tf.local_variables_initializer())
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
        distance_tensor = tf.reduce_mean(tf.abs(self.__real__ - self.__output__))
        accuracy_queue = []
        num_true = 0

        if run_type == Network.RUN_TRAIN:
            print("\nNN training start. %s" % datetime.now())
            self.__images_cntrl__.init_train_data()
            num_iterate = self.__train_num_iterate__
            next_batch_getter = self.__images_cntrl__.get_next_train_batch
            fetches = [self.__update__, self.__loss__, distance_tensor]
            if num_iterate >= self.__train_num_show_status__:
                interval_show_status = num_iterate // self.__train_num_show_status__
            else:
                interval_show_status = 1
            layers = str(self.__num_inputs__) + " -> " + \
                     " -> ".join(map(str, self.__inner_layers_num_neurons__)) + " -> 1"
            classes_ratio = "{0} vehicles / {1} non-vehicles".format(str(self.__train_classes_ratio__.numerator),
                                                                     str(self.__train_classes_ratio__.denominator))
            print("Number of hidden layers %d. Layers: %s. Training rate %f.\n"
                  "Classes ratio: %s\n"
                  "Batch size %d. Number of iterations %d.\n"
                  "Results are being printed every %d steps.\n"
                  % (self.__num_inner_layers__, layers, self.__train_rate__, classes_ratio, self.__batch_size__,
                     num_iterate, interval_show_status))
        else:
            print("\nNN test start. %s\n" % datetime.now())
            self.__images_cntrl__.init_test_data()
            num_iterate = (self.__images_cntrl__.num_test // self.__batch_size__) + \
                          (1 if self.__images_cntrl__.num_test % self.__batch_size__ > 0 else 0)
            next_batch_getter = self.__images_cntrl__.get_next_test_batch
            fetches = [self.__output__, self.__loss__, distance_tensor]
            interval_show_status = 0

        i = 0
        has_more_images = True
        while i < num_iterate and has_more_images:
            batch_xs, batch_ys = next_batch_getter()
            if batch_xs is None:
                has_more_images = False
            else:
                _, curr_loss, curr_distance = self.__sess__.run(fetches=fetches,
                                                                feed_dict={self.__input__: batch_xs,
                                                                           self.__real__: batch_ys})
                curr_accuracy = True if curr_distance <= 0.02 else False
                if run_type == Network.RUN_TRAIN:
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
            if run_type == Network.RUN_TRAIN:
                print("\nNN training end. %s\nFinal loss %0.2f. Final accuracy %d percents.\n"
                      % (datetime.now(), loss_mean, num_true * 10))
            else:
                print("\nNN test end. %s\nFinal loss %0.2f. Final accuracy %0.2f percents.\n"
                      % (datetime.now(), loss_mean, 100 * (num_true / num_iterate)))
