import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class CategoricalCircleOnCircumferenceMSE(tf.keras.losses.Loss):
    def __init__(self, shape, r1, r2, class_num, name="categorical_round_mse", **kwargs):
        super(CategoricalCircleOnCircumferenceMSE, self).__init__(name=name, **kwargs)
        assert len(shape) == 2
        assert 0 < r1
        assert 0 < class_num
        assert 0 < r2 < r1 * np.tan(2 * np.pi / (2 * class_num))
        assert 0 < r1 + r2 < np.max(shape) / 2
        self.shape = shape
        self.r1 = r1
        self.r2 = r2
        self.class_num = class_num

        self.f_list = tf.constant(self.calc_filters(shape, r1, r2, class_num), dtype=tf.float32)

    @staticmethod
    def calc_filters(shape, r1, r2, class_num):
        rads = np.linspace(0, 2 * np.pi, class_num, endpoint=False)
        x = np.arange(shape[1])
        y = np.arange(shape[0])
        xx, yy = np.meshgrid(x, y)
        xx = xx - np.mean(xx)
        yy = yy - np.mean(yy)
        f_list = []
        for rad in rads:
            p = r1 * np.cos(rad - np.pi / 2)
            q = r1 * np.sin(rad - np.pi / 2)
            f_list.append(np.where((xx - p) ** 2 + (yy - q) ** 2 <= r2 ** 2, 1, 0))
        return tf.constant(np.array(f_list), dtype=tf.float32)

    @staticmethod
    def plot(shape, r1, r2, class_num, ax=None):
        filters = CategoricalCircleOnCircumferenceMSE.calc_filters(shape, r1, r2, class_num)
        sum_image = tf.reduce_sum(filters, axis=0)
        if ax:
            ax.imshow(sum_image.numpy())
        else:
            fig = plt.figure()
            _ax = fig.add_subplot()
            _ax.imshow(sum_image.numpy())

    def get_config(self):
        config = super().get_config()
        config.update({
            "shape": self.shape,
            "r1": self.r1,
            "r2": self.r2,
            "class_num": self.class_num
        })
        return config

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_image = tf.gather(self.f_list, y_true, axis=0)
        mse = tf.reduce_mean(tf.square(y_true_image - y_pred), axis=[1, 2])
        return mse


class CategoricalRhombusOnCircumferenceMSE(tf.keras.losses.Loss):
    def __init__(self, shape, r1, r2, class_num, name="categorical_round_mse", **kwargs):
        super(CategoricalRhombusOnCircumferenceMSE, self).__init__(name=name, **kwargs)
        assert len(shape) == 2
        assert 0 < r1
        assert 0 < class_num
        assert 0 < r2 < r1 * np.tan(2 * np.pi / (2 * class_num))
        assert 0 < r1 + r2 < np.max(shape) / 2
        self.shape = shape
        self.r1 = r1
        self.r2 = r2
        self.class_num = class_num

        self.f_list = self.calc_filters(shape, r1, r2, class_num)

    @staticmethod
    def calc_filters(shape, r1, r2, class_num):
        rads = np.linspace(0, 2 * np.pi, class_num, endpoint=False)
        x = np.arange(shape[1])
        y = np.arange(shape[0])
        xx, yy = np.meshgrid(x, y)
        xx = xx - np.mean(xx)
        yy = yy - np.mean(yy)
        f_list = []
        for rad in rads:
            p = r1 * np.cos(rad - np.pi / 2)
            q = r1 * np.sin(rad - np.pi / 2)
            f_list.append(np.where(np.abs(xx - p) + np.abs(yy - q) <= r2 / 2, 1, 0))
        return tf.constant(np.array(f_list), dtype=tf.float32)

    @staticmethod
    def plot(shape, r1, r2, class_num, ax=None):
        filters = CategoricalRhombusOnCircumferenceMSE.calc_filters(shape, r1, r2, class_num)
        sum_image = tf.reduce_sum(filters, axis=0)
        if ax:
            ax.imshow(sum_image.numpy())
        else:
            fig = plt.figure()
            _ax = fig.add_subplot()
            _ax.imshow(sum_image.numpy())

    def get_config(self):
        config = super().get_config()
        config.update({
            "shape": self.shape,
            "r1": self.r1,
            "r2": self.r2,
            "class_num": self.class_num
        })
        return config

    def call(self, y_true, y_pred):
        y_true_image = tf.gather(self.f_list, y_true, axis=0)
        mse = tf.reduce_mean(tf.square(y_true_image - y_pred), axis=[1, 2])
        return mse
