from ConvNetwork import ConvNetwork
from Network import Network
from LogisticRegression import LogisticRegression
from fractions import Fraction
from tkinter import messagebox


def main():
    """
    log_regression = LogisticRegression(vehicles_train_path="C:\\Users\\orik\\DLImages\\vehicles\\Train\\",
                                        non_vehicles_train_path="C:\\Users\\orik\\DLImages\\non-vehicles\\Train\\",
                                        vehicles_test_path="C:\\Users\\orik\\DLImages\\vehicles\\Test\\",
                                        non_vehicles_test_path="C:\\Users\\orik\\DLImages\\non-vehicles\\Test\\",
                                        image_shape=(64, 64), batch_size=30,
                                        train_classes_ratio=Fraction(2, 1), train_rate=0.0025,
                                        train_num_iterate=20000)
    log_regression.init_regression_session()
    log_regression.run(log_regression.RUN_TRAIN)
    log_regression.run(log_regression.RUN_TEST)
    log_regression.close()

    net = Network(vehicles_train_path="C:\\Users\\orik\\DLImages\\vehicles\\Train\\",
                  non_vehicles_train_path="C:\\Users\\orik\\DLImages\\non-vehicles\\Train\\",
                  vehicles_test_path="C:\\Users\\orik\\DLImages\\vehicles\\Test\\",
                  non_vehicles_test_path="C:\\Users\\orik\\DLImages\\non-vehicles\\Test\\",
                  image_shape=(64, 64), inner_layers_num_neurons=(500, 300, 170, 90, 45, 20),
                  batch_size=30, train_classes_ratio=Fraction(2, 1), train_rate=0.0025,
                  train_num_iterate=100000)
    net.init_net_session()
    net.run(net.RUN_TRAIN)
    net.run(net.RUN_TEST)
    net.close()
    """

    cnn = ConvNetwork(vehicles_train_path="C:\\Users\\orik\\DLImages\\vehicles\\Train\\",
                      non_vehicles_train_path="C:\\Users\\orik\\DLImages\\non-vehicles\\Train\\",
                      vehicles_test_path="C:\\Users\\orik\\DLImages\\vehicles\\Test\\",
                      non_vehicles_test_path="C:\\Users\\orik\\DLImages\\non-vehicles\\Test\\",
                      image_shape=(64, 64), conv_layers_num_kernels=(10, 20, 30, 40, 50), conv_kernel_dimensions=(8, 8),
                      fc_layers_num_neurons=(50,), batch_size=28, train_classes_ratio=Fraction(3, 1),
                      train_rate=0.0001, train_num_iterate=100000, train_num_show_status=2000)
    cnn.init_net_session()
    cnn.run(cnn.RUN_TRAIN)
    cnn.close()

    messagebox.showinfo(title="Classifier", message="Mission complete")


if __name__ == '__main__':
    main()
