from Network import Network
from LogisticRegression import LogisticRegression
from fractions import Fraction
from tkinter import messagebox


def main():
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

    messagebox.showinfo(title="Classifier", message="Mission complete")


if __name__ == '__main__':
    main()
