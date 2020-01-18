from glob import glob
from Utils import trailing_slash
import numpy as np
import cv2
from fractions import Fraction
from random import shuffle


class ImagesController:
    V_TRAIN = 1
    NV_TRAIN = 2
    V_TEST = 3
    NV_TEST = 4

    def __init__(self, vehicles_train_path, non_vehicles_train_path, vehicles_test_path, non_vehicles_test_path,
                 batch_size, train_classes_ratio: Fraction):
        self.__v_train_path__ = trailing_slash(vehicles_train_path, False)
        self.__nv_train_path__ = trailing_slash(non_vehicles_train_path, False)
        self.__v_test_path__ = trailing_slash(vehicles_test_path, False)
        self.__nv_test_path__ = trailing_slash(non_vehicles_test_path, False)

        self.__v_train_len__ = len(glob(pathname=self.__v_train_path__ + "\\**\\*.png", recursive=True))
        self.__nv_train_len__ = len(glob(pathname=self.__nv_train_path__ + "\\**\\*.png", recursive=True))
        self.__v_test_len__ = len(glob(pathname=self.__v_test_path__ + "\\**\\*.png", recursive=True))
        self.__nv_test_len__ = len(glob(pathname=self.__nv_test_path__ + "\\**\\*.png", recursive=True))

        self.__batch_size__ = batch_size
        self.__classes_ratio__ = train_classes_ratio
        self.__num_v_batch__ = self.__classes_ratio__.numerator
        self.__num_nv_batch__ = self.__classes_ratio__.denominator
        self.__base_batch_size__ = self.__num_v_batch__ + self.__num_nv_batch__

        if self.__batch_size__ > min(self.__v_train_len__, self.__nv_train_len__):
            raise ValueError("Batch size is too high. Maximum valid value is " +
                             str(min(self.__v_train_len__, self.__nv_train_len__)))
        elif self.__batch_size__ < self.__base_batch_size__:
            raise ValueError("Batch size is too low. Minimum valid value is " + str(self.__base_batch_size__))
        elif self.__batch_size__ % self.__base_batch_size__ > 0:
            raise ValueError("Invalid batch size. Batch size has to be a multiple of " + str(self.__base_batch_size__))

        self.__v_train_list_1__: list = []
        self.__v_train_list_2__: list = []
        self.__nv_train_list_1__: list = []
        self.__nv_train_list_2__: list = []
        self.__v_test_list__: list = []
        self.__nv_test_list__: list = []

        self.__v_curr_len__ = 0
        self.__v_active_list__ = 0
        self.__nv_curr_len__ = 0
        self.__nv_active_list__ = 0
        self.__has_more_images__ = True

    def init_train_data(self):
        self.__v_train_list_1__ = glob(pathname=self.__v_train_path__ + "\\**\\*.png", recursive=True)
        shuffle(self.__v_train_list_1__)
        self.__v_train_list_2__.clear()
        self.__nv_train_list_1__ = glob(pathname=self.__nv_train_path__ + "\\**\\*.png", recursive=True)
        shuffle(self.__nv_train_list_1__)
        self.__nv_train_list_2__.clear()
        self.__v_curr_len__ = self.__v_train_len__
        self.__v_active_list__ = 1
        self.__nv_curr_len__ = self.__nv_train_len__
        self.__nv_active_list__ = 1

    def init_test_data(self):
        self.__v_test_list__ = glob(pathname=self.__v_test_path__ + "\\**\\*.png", recursive=True)
        self.__nv_test_list__ = glob(pathname=self.__nv_test_path__ + "\\**\\*.png", recursive=True)
        self.__has_more_images__ = True

    def get_next_train_batch(self):
        images = None
        classes = None
        for i in range(self.__batch_size__):
            if 0 <= (i % self.__base_batch_size__) < self.__num_v_batch__:
                path, _ = self._get_next_path(ImagesController.V_TRAIN)
                cls = 1
            else:
                path, _ = self._get_next_path(ImagesController.NV_TRAIN)
                cls = 0
            if images is None:
                images = np.array([np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).flatten()])
                classes = np.array([[cls]])
            else:
                images = np.vstack([images, np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).flatten()])
                classes = np.vstack([classes, [cls]])

        return self._normalize(images), classes

    def get_next_test_batch(self):
        images = None
        classes = None
        i = 0
        while i < self.__batch_size__ and self.__has_more_images__:
            if i % 2 == 0:
                path, self.__has_more_images__ = self._get_next_path(ImagesController.V_TEST)
                cls = 1
            else:
                path, self.__has_more_images__ = self._get_next_path(ImagesController.NV_TEST)
                cls = 0
            if images is None:
                images = np.array([np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).flatten()])
                classes = np.array([[cls]])
            else:
                images = np.vstack([images, np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).flatten()])
                classes = np.vstack([classes, [cls]])
            i += 1

        return (self._normalize(images) if images is not None else None), classes

    def _get_next_path(self, img_type: int):
        if img_type == ImagesController.V_TRAIN:
            self.__v_curr_len__ -= 1
            if self.__v_active_list__ == 1:
                file_path = self.__v_train_list_1__.pop()
                self.__v_train_list_2__.append(file_path)
                if self.__v_curr_len__ == 0:
                    self.__v_curr_len__ = self.__v_train_len__
                    self.__v_active_list__ = 2
                    shuffle(self.__v_train_list_2__)
            else:
                file_path = self.__v_train_list_2__.pop()
                self.__v_train_list_1__.append(file_path)
                if self.__v_curr_len__ == 0:
                    self.__v_curr_len__ = self.__v_train_len__
                    self.__v_active_list__ = 1
                    shuffle(self.__v_train_list_1__)
            has_more = True
        elif img_type == ImagesController.NV_TRAIN:
            self.__nv_curr_len__ -= 1
            if self.__nv_active_list__ == 1:
                file_path = self.__nv_train_list_1__.pop()
                self.__nv_train_list_2__.append(file_path)
                if self.__nv_curr_len__ == 0:
                    self.__nv_curr_len__ = self.__nv_train_len__
                    self.__nv_active_list__ = 2
                    shuffle(self.__nv_train_list_2__)
            else:
                file_path = self.__nv_train_list_2__.pop()
                self.__nv_train_list_1__.append(file_path)
                if self.__nv_curr_len__ == 0:
                    self.__nv_curr_len__ = self.__nv_train_len__
                    self.__nv_active_list__ = 1
                    shuffle(self.__nv_train_list_1__)
            has_more = True
        elif img_type == ImagesController.V_TEST:
            file_path = self.__v_test_list__.pop()
            self.__v_test_len__ -= 1
            has_more = (True if self.__v_test_len__ > 0 else False)
        else:
            file_path = self.__nv_test_list__.pop()
            self.__nv_test_len__ -= 1
            has_more = (True if self.__nv_test_len__ > 0 else False)
        return file_path, has_more

    @property
    def num_test(self):
        return self.__v_test_len__ + self.__nv_test_len__

    @staticmethod
    def _normalize(arr):
        arr = arr.astype(float)
        norm_arr = np.zeros(arr.shape, dtype=float)
        norm_arr = cv2.normalize(arr, norm_arr, 0.0, 1.0, cv2.NORM_MINMAX, dtype=-1)
        return norm_arr
        """
        var = np.var(arr)
        mean = np.mean(arr)
        return (arr - mean) / var
        """
