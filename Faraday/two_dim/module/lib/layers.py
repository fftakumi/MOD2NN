"""レイヤーモジュール

* MO-D2NNでニューラルネットワークを形成する時に使うレイヤーのクラス群

"""

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt


class AngularSpectrum(tf.keras.layers.Layer):
    """角スペクトル法

    左右円偏光を各スペクトル法によって伝搬させる

    Attributes:
        output_dim (tuple, array):出力画像のピクセル数
        wavelength (float):入力光の波長
        wavelength_eff (float):屈折率を考慮した波長
        k(float):屈折率を考慮した波数
        z (float):伝搬距離
        d (float):ニューロンの大きさ
        n (float):伝搬媒質の屈折率
        res (array):周波数応答関数
        normalization(str): 出力する電場の正規化法であり、"max"とすれば最大光強度を1とする
        method(str):"band_limited"であれば4倍拡張は行わずに帯域制限のみ行う。"expand"tとすれば4倍拡張と帯域制限を行う。Noneだと何もしない
    """

    def __init__(self, output_dim, wavelength=633e-9, z=0.0, d=1.0e-6, n=1.0, normalization=None, method=None):
        """コンストラクタ

        クラスの初期化を行う

        Args:
            output_dim (tuple, list):出力画像のピクセル数
            wavelength (float):入力光の波長
            z (float):伝搬距離
            d (float):ニューロンの大きさ
            n (float):伝搬媒質の屈折率
            normalization(str): 出力する電場の正規化法であり、"max"とすれば最大光強度を1とする
            method(str):"band_limited"であれば4倍拡張は行わずに帯域制限のみ行う。"expand"tとすれば4倍拡張と帯域制限を行う。Noneだと何もしない
        """
        super(AngularSpectrum, self).__init__()
        self.output_dim = output_dim
        self.wavelength = wavelength
        self.wavelength_eff = wavelength / n
        self.k = 2 * np.pi / self.wavelength_eff
        self.z = z
        self.d = d
        self.n = n
        self.normalization = normalization if normalization is not None else "None"
        self.method = method if method is not None else "None"

        assert self.k >= 0.0
        assert self.z >= 0.0
        assert self.d > 0.0
        assert self.n > 0.0

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "wavelength": self.wavelength,
            "k": self.k,
            "z": self.z,
            "d": self.d,
            "n": self.n,
            "normalization": self.normalization,
            "method": self.method
        })
        return config

    @classmethod
    def from_config(cls, config):
        """クラスの初期化

        get_config()で得られる辞書方配列からクラスのインスタンスを作り直す

        Args:
            config (dict):  コンストラクタの引数を鍵に持つ辞書型配列

        Returns:
            object: configで指定した属性を持つこのクラスのインスタンス

        """
        return cls(**config)

    def build(self, input_dim):
        """定数の固定

        各スペクトル法に使う周波数応答関数を求める

        Args:
            input_dim (tuple, list): 入力画像の画素数
        """
        self.input_dim = input_dim

        width = self.input_dim[-1]
        height = self.input_dim[-2]
        u = np.fft.fftfreq(width, d=self.d)
        v = np.fft.fftfreq(height, d=self.d)
        UU, VV = np.meshgrid(u, v)
        w = np.where(UU ** 2 + VV ** 2 <= 1 / self.wavelength_eff ** 2, tf.sqrt(1 / self.wavelength_eff ** 2 - UU ** 2 - VV ** 2), 0).astype('float64')
        h = np.exp(1.0j * 2 * np.pi * w * self.z)

        if self.method == 'band_limited':
            du = 1 / (2 * width * self.d)
            dv = 1 / (2 * height * self.d)
            u_limit = 1 / (np.sqrt((2 * du * self.z) ** 2 + 1)) / self.wavelength_eff
            v_limit = 1 / (np.sqrt((2 * dv * self.z) ** 2 + 1)) / self.wavelength_eff
            u_filter = np.where(np.abs(UU) / (2 * u_limit) <= 1 / 2, 1, 0)
            v_filter = np.where(np.abs(VV) / (2 * v_limit) <= 1 / 2, 1, 0)
            h = h * u_filter * v_filter
        elif self.method == 'expand':
            self.pad_upper = math.ceil(self.input_dim[-2] / 2)
            self.pad_left = math.ceil(self.input_dim[-1] / 2)
            self.padded_width = int(input_dim[-1] + self.pad_left * 2)
            self.padded_height = int(input_dim[-2] + self.pad_upper * 2)

            u = np.fft.fftfreq(self.padded_width, d=self.d)
            v = np.fft.fftfreq(self.padded_height, d=self.d)

            du = 1 / (self.padded_width * self.d)
            dv = 1 / (self.padded_height * self.d)
            u_limit = 1 / (np.sqrt((2 * du * self.z) ** 2 + 1)) / self.wavelength_eff
            v_limit = 1 / (np.sqrt((2 * dv * self.z) ** 2 + 1)) / self.wavelength_eff
            UU, VV = np.meshgrid(u, v)

            u_filter = np.where(np.abs(UU) <= u_limit, 1, 0)
            v_filter = np.where(np.abs(VV) <= v_limit, 1, 0)

            w = np.where(UU ** 2 + VV ** 2 <= 1 / self.wavelength_eff ** 2, tf.sqrt(1 / self.wavelength_eff ** 2 - UU ** 2 - VV ** 2), 0).astype('float64')
            h = np.exp(1.0j * 2 * np.pi * w * self.z)
            h = h * u_filter * v_filter

        self.res = tf.cast(tf.complex(h.real, h.imag), dtype=tf.complex64)

    @tf.function
    def propagation(self, cximages):
        """光の伝搬関数

        methodで指定した方法に従って角スペクトル法による光の伝搬を行う

        Args:
            cximages (tensor):伝搬元の複素電波分布

        Returns:
            tensor: 伝搬後の複素電場分布
        """
        if self.method == 'band_limited':
            images_fft = tf.signal.fft2d(cximages)
            return tf.signal.ifft2d(images_fft * self.res)
        elif self.method == 'expand':
            padding = [[0, 0], [self.pad_upper, self.pad_upper], [self.pad_left, self.pad_left]]
            images_pad = tf.pad(cximages, paddings=padding)
            images_pad_fft = tf.signal.fft2d(images_pad)
            u_images_pad = tf.signal.ifft2d(images_pad_fft * self.res)
            u_images = tf.keras.layers.Lambda(lambda x: x[:, self.pad_upper:self.pad_upper + self.input_dim[-2], self.pad_left:self.pad_left + self.input_dim[-1]])(u_images_pad)
            return u_images
        else:
            images_fft = tf.signal.fft2d(cximages)
            return tf.signal.ifft2d(images_fft * self.res)

    @tf.function
    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        入力から左右円偏光をそれぞれ伝搬させ、正規化方法を指定している場合は正規化する

        Args:
            x (tensor): 各ミニバッチの左右円偏光の電場分布

        Returns:　
            tensor:伝搬後の左右円偏光の電場分布
        """
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, :, :])(x)

        u_rcp_x = self.propagation(rcp_x)
        u_lcp_x = self.propagation(lcp_x)

        rl = tf.stack([u_rcp_x, u_lcp_x], axis=1)

        if self.normalization == 'max':
            # max intensity = 1
            _maximum = tf.reduce_max(tf.sqrt(tf.abs(u_rcp_x) ** 2 + tf.abs(u_lcp_x) ** 2), axis=[-2, -1], keepdims=True)
            maximum = tf.expand_dims(_maximum, axis=1)
            return rl / tf.complex(maximum, 0.0 * maximum)

        return rl


class ImageResizing(tf.keras.layers.Layer):
    """画像のリサイズ

    前処理として、画像の解像度を変えるクラス。全ての画像の解像度を変えると大量にメモリを消費するため、純伝搬のときにその都度解像度を変えるためのクラス。
    """

    def __init__(self, output_dim):
        """コンストラクタ

        Args:
            output_dim (tuple, list): 解像度の変換後の画素数
        """
        super(ImageResizing, self).__init__()
        self.output_dim = output_dim

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim
        })
        return config

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        ミニバッチごとの画像をtf.image.resizeでoutput_dimの大きさにリサイズする。

        Args:
            x (tensor):各ミニバッチの入力画像

        Returns: 
            tensor:リサイズされた画像

        """
        x_expnad = tf.image.resize(tf.expand_dims(x, -1), self.output_dim)
        x_expnad = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0])(x_expnad)
        return x_expnad


class ImageBinarization(tf.keras.layers.Layer):
    """画像の2値化

    フォトマスク等の2値画像を使う時のクラス。閾値を堺に2値化する。

    Args:
        threshold (float): 閾値
        minimum (float): 閾値より小さいときに設定される値
        maximum (float): 閾値より大きいときに設定される値
    """
    def __init__(self, threshold=0.5, minimum=0.0, maximum=1.0):
        """コンストラクタ

        Args:
            threshold: 閾値
            minimum: 閾値より小さいときに設定される値
            maximum: 閾値より大きいときに設定される値
        """
        super(ImageBinarization, self).__init__()
        self.threshold = threshold
        self.minimum = minimum
        self.maximum = maximum

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
            "minimum": self.minimum,
            "maximum": self.maximum
        })
        return config

    @classmethod
    def from_config(cls, config):
        """クラスの初期化

        get_config()で得られる辞書方配列からクラスのインスタンスを作り直す

        Args:
            config (dict):  コンストラクタの引数を鍵に持つ辞書型配列

        Returns: 
            object: configで指定した属性を持つこのクラスのインスタンス

        """
        return cls(**config)

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        画像を2値化する

        Args:
            x (tensor):ミニバッチにおける2値化前の画像

        Returns: 
            tensor:2値化された画像

        """
        return tf.where(x >= self.threshold, self.maximum, self.minimum)


class IntensityToElectricField(tf.keras.layers.Layer):
    """光強度から左右円偏光に変換

        光強度分布を直線偏光の左右円偏光の電場分布に変換する。この時の偏光面の角度はini_thetaによって決まる。

        Args:
            ini_theta (float): 偏光面の角度
    """
    def __init__(self, output_dim, ini_theta=0.0):
        """コンストラクタ

        Args:
            output_dim:
            ini_theta:
        """
        super(IntensityToElectricField, self).__init__()
        self.output_dim = output_dim
        self.ini_theta = ini_theta

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "ini_theta": self.ini_theta
        })
        return config

    def build(self, input_dim):
        """定数の決定

        ini_thetaに従って複素ファラデー回転角を定義

        Args:
            input_dim: 入力画像の画素数

        """
        self.input_dim = input_dim
        self.theta = tf.complex(self.ini_theta, 0.0)

    @tf.function
    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        thetaを使って光強度分布から偏光面の角度がini_thetaの左右円偏光を計算する

        Args:
            x:　ミニバッチにおける光強度分布

        Returns: 
            tensor:左右円偏光の電場分布

        """
        rcp_x = tf.complex(tf.sqrt(x / 2.0), 0.0 * x) * tf.exp(-1.0j * self.theta)
        lcp_x = tf.complex(tf.sqrt(x / 2.0), 0.0 * x) * tf.exp(1.0j * self.theta)
        return tf.stack([rcp_x, lcp_x], axis=1)


class ElectricFieldToIntensity(tf.keras.layers.Layer):
    """左右円偏光から光強度の算出

    Args:
        normalization(str): 正規化の方法
    """
    def __init__(self, output_dim, normalization=None):
        """コンストラクタ

        Args:
            output_dim(tuple, list): 入力画像の画素数
            normalization(str): 正規化の方法。"max"であれば最大光強度を1にする。
        """
        super(ElectricFieldToIntensity, self).__init__()
        self.output_dim = output_dim
        self.normalization = normalization

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """

        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "normalization": self.normalization
        })
        return config

    @classmethod
    def from_config(cls, config):
        """クラスの初期化

        get_config()で得られる辞書方配列からクラスのインスタンスを作り直す

        Args:
            config (dict):  コンストラクタの引数を鍵に持つ辞書型配列

        Returns:
            object: configで指定した属性を持つこのクラスのインスタンス

        """
        return cls(**config)

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        左右円偏光から光強度に変換する

        Args:
            x (tensor):左右円偏光の電場分布

        Returns: 
            tensor:光強度分布

        """
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(x)
        rcp_y = 1.0j * rcp_x
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, :, :])(x)
        lcp_y = -1.0j * lcp_x

        tot_x = rcp_x + lcp_x
        tot_y = rcp_y + lcp_y

        intensity = tf.abs(tot_x) ** 2 / 2.0 + tf.abs(tot_y) ** 2 / 2.0

        if self.normalization == 'max':
            return intensity / tf.reduce_max(intensity)

        return intensity


class MO(tf.keras.layers.Layer):
    """磁気光学効果

    磁気光学効果による左右円偏光の変調を行う。

    Args:
         output_dim(tuple, list): 出力画像の画素数
        limitation(str): "sin" or "tanh" or "sigmoid"パラメータの制限に使う関数
        theta(float): 最大ファラデー回転角(残留磁化でのファラデー回転角)
        eta(float): 最大ファラデー楕円率(残留磁化でのファラデー楕円率)
        kernel_regularizer(tensorflow.keras.regularizers.Regularizer): 正則化を使うときの引数
        kernel_initializer (str): パラメータの初期化法
        trainable(bool): Trueなら学習する、Falseなら学習しない。
        mag(tensorflow.Variable): 学習パラメータ

    """
    def __init__(self, output_dim, limitation=None, theta=0.0, eta=0.0, kernel_regularizer=None, kernel_initializer=None, trainable=True, name=None, dtype=tf.float32, dynamic=False, **kwargs):
        """コンストラクタ

        Args:
            output_dim(tuple, list): 出力画像の画素数
            limitation(str): "sin" or "tanh" or "sigmoid"パラメータの制限に使う関数
            theta(float): 最大ファラデー回転角(残留磁化でのファラデー回転角)
            eta(float): 最大ファラデー楕円率(残留磁化でのファラデー楕円率)
            kernel_regularizer(tensorflow.keras.regularizers.Regularizer): 正則化を使うときの引数
            kernel_initializer (str): パラメータの初期化法
            trainable(bool): Trueなら学習する、Falseなら学習しない。
            name(str):レイヤーの名前
            dtype(tesorflow.dtype):パラメターのデータ型
            dynamic(bool): dynamicにするかどうか
            **kwargs:
        """
        super(MO, self).__init__(
            trainable=trainable,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
            **kwargs
        )
        self.output_dim = output_dim

        self.limitation = limitation if limitation is not None else "None"
        self.theta = tf.cast(theta, self.dtype)
        self.eta = tf.cast(eta, self.dtype)
        self.eta_max = tf.cast(abs(eta), self.dtype)
        self.alpha = tf.cast(tf.math.log((1. + self.eta) / (1. - self.eta)) / 2., self.dtype)
        self.phi_common = tf.complex(tf.constant(0., dtype=tf.float32), tf.constant(tf.math.log(1. + self.eta_max), dtype=tf.float32))
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        assert len(self.output_dim) == 2
        assert -1.0 < self.eta < 1.0

    def build(self, input_dim):
        """パラメータの宣言

        Args:
            input_dim(tuple, list): 入力画像の画素数


        """
        self.input_dim = input_dim
        self.mag = self.add_weight("magnetization",
                                   shape=[int(input_dim[-2]),
                                          int(input_dim[-1])],
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   dtype=self.dtype)
        super(MO, self).build(input_dim)

    @tf.function
    def get_limited_complex_faraday(self):
        """複素ファラデー回転角の取得

        limitationに従って複素ファラデー回転角を返す。

        Returns:
             tensor:複素ファラデー回転角

        """
        if self.limitation == "sin":
            theta = self.theta * tf.sin(self.mag)
            alpha = self.alpha * tf.sin(self.mag)
            return tf.complex(theta, alpha)
        elif self.limitation == "tanh":
            theta = self.theta * tf.tanh(self.mag)
            alpha = self.alpha * tf.tanh(self.mag)
            return tf.complex(theta, alpha)
        elif self.limitation == "sigmoid":
            theta = self.theta * (2.*tf.math.sigmoid(self.mag)-1.)
            alpha = self.alpha * (2.*tf.math.sigmoid(self.mag)-1.)
            return tf.complex(theta, alpha)

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "limitation": self.limitation,
            "theta": float(self.theta.numpy()),
            "eta": float(self.eta.numpy())
        })
        if self.kernel_regularizer:
            config.update({
                "reguralizer": self.kernel_regularizer.get_config()
            })
        return config

    @classmethod
    def from_config(cls, config):
        """クラスの初期化

        get_config()で得られる辞書方配列からクラスのインスタンスを作り直す

        Args:
            config (dict):  コンストラクタの引数を鍵に持つ辞書型配列

        Returns: 
            object: configで指定した属性を持つこのクラスのインスタンス

        """

        return cls(**config)

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        受け取った電場分布を複素ファラデー回転角を使って位相変調する。

        Args:
            x(tensor):左右円偏光の電場分布

        Returns:
            tensor:磁気光学効果によって変調された左右円偏光の電場分布

        """
        phi = self.get_limited_complex_faraday()

        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, :, :])(x)

        rcp_x_mo = rcp_x * tf.exp(-1.j * phi) * tf.exp(1.j * self.phi_common)
        lcp_x_mo = lcp_x * tf.exp(1.j * phi) * tf.exp(1.j * self.phi_common)

        return tf.stack([rcp_x_mo, lcp_x_mo], axis=1)


class BinarizedMO(MO):
    """パラメータが2値の状態しかない磁気光学効果

    Args:
        output_dim(tuple, list): 入力画像の画素
        theta(float): 最大ファラデー回転角(残留磁化でのファラデー回転角)
        eta(float): 最大ファラデー楕円率(残留磁化でのファラデー楕円率)
        sign_approximation(str): 符号化関数の近似法。今の所"signswish"(https://doi.org/10.48550/arXiv.1812.11800)しか無い。Noneにした場合は勾配を1とする。
        kernel_regularizer(tensorflow.keras.regularizers.Regularizer): 正則化をする場合の正則化関数

    """
    def __init__(self, output_dim, theta=0.0, eta=0.0, sign_approximation="signswish", kernel_regularizer=None, **kwargs):
        """コンストラクタ

        Args:
            output_dim(tuple, list): 入力画像の画素
            theta(float): 最大ファラデー回転角(残留磁化でのファラデー回転角)
            eta(float): 最大ファラデー楕円率(残留磁化でのファラデー楕円率)
            sign_approximation(str): 符号化関数の近似法。今の所"signswish"(https://doi.org/10.48550/arXiv.1812.11800)しか無い。Noneにした場合は勾配を1とする。
            kernel_regularizer(tensorflow.keras.regularizers.Regularizer): 正則化をする場合の正則化関数
            **kwargs (dict):その他
        """
        self.sign_approximation = sign_approximation
        if sign_approximation == "signswish":
            assert "beta" in kwargs
            self.beta = kwargs["beta"]

        super(BinarizedMO, self).__init__(
            output_dim=output_dim,
            limitation=None,
            theta=theta,
            eta=eta,
            kernel_regularizer=kernel_regularizer
        )

    @tf.custom_gradient
    def no_op(self, x):
        """符号化関数

        純伝搬のときは符号化関数を使い、逆伝搬の時の勾配は常に1になる関数。

        Args:
            x(float, array): 2値化する値

        Returns: 
            float, tensor:2値化された値

        """
        def grad(upstream):
            return upstream * 1.

        y = tf.clip_by_value((x + 1.) / 2., 0, 1)
        z = 2. * tf.round(y) - 1.
        return z, grad

    @tf.custom_gradient
    def signswish(self, x):
        """signswish関数

        純伝搬のときは符号化関数を使い、逆伝搬の時のsignswish関数の微分を用いる。

        Args:
            x(float, array):  2値化する値

        Returns:
            float, tensor:2値化された値

        """
        beta = self.beta

        def grad(upstream):
            return upstream * beta * (2. - beta * x * tf.tanh(beta * x / 2.)) / (1. + tf.cosh(beta * x))

        y = tf.clip_by_value((x + 1.) / 2., 0, 1)
        z = 2. * tf.round(y) - 1.
        return z, grad

    @tf.function
    def binaryzation(self, x):
        """2値化関数

        sign_approximationに従って2値化を行う。

        Args:
            x (float, array): 2値化する値

        Returns:
            float, tensor:2値化された値

        """
        if self.sign_approximation == "signswish":
            return self.signswish(x)

        return self.no_op(x)

    @tf.function
    def get_b_kernel(self):
        """2値パラメータの取得

        Returns:
            tensor:2値状態のパラメータ

        """
        return self.binaryzation(self.mag)

    def get_binarized_complex_faraday(self):
        """2値化された状態の複素ファラデー回転角の取得

        Returns:
            tensor:2値化された複素ファラデー回転角

        """
        b_kernel = self.binaryzation(self.mag)
        theta = self.theta * b_kernel
        alpha = self.alpha * b_kernel
        return tf.complex(theta, alpha)

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """

        config = super().get_config()
        config.update({
            "sign_approximation": self.sign_approximation
        })
        if self.sign_approximation == "signswish":
            config.update({
                "beta": self.beta
            })
        return config

    def build(self, input_shape):
        """パラメータの設定

        MOクラスを継承しているので、MOクラスのbuidを読んでパラメータを定義する。

        Args:
            input_shape: 入力画像の画素数

        """
        super(BinarizedMO, self).build(input_shape)
    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        2値の磁化による磁気光学効果の計算

        Args:
            x:左右円偏光の複素電場分布

        Returns:
            tensor:磁気光学効果によって位相変調された左右円偏光の電場分布

        """
        phi = self.get_binarized_complex_faraday()

        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, :, :])(x)

        rcp_x_mo = rcp_x * tf.exp(-1.j * phi) * tf.exp(1.j * self.phi_common)
        lcp_x_mo = lcp_x * tf.exp(1.j * phi) * tf.exp(1.j * self.phi_common)

        return tf.stack([rcp_x_mo, lcp_x_mo], axis=1)


class MNISTDetector(tf.keras.layers.Layer):
    """MNIST(10クラス)用のディテクター

    Args:
        output_dim(tuple, list): 出力画像の画素数
        inverse(bool): 符号の反転を行うかどうか
        activation(str): 活性化関数 "softmax"とすれば出力にsoftmax関数を適応する。
        normalization(str): 正規化 "minmax"とすれば最小値を-1、最大値を1にする
        mode(str): フィルターの作り方。"v1"か"v2"
        width(int): 検出器の幅の画素数
        height(int): 検出器の高さの画素数
        pad_w(int): 両脇の余白の画素数
        pad_h(int): 上下の余白の画素数
    """

    def __init__(self, output_dim, inverse=False, activation=None, normalization=None, mode="v2", width=None, height=None, pad_w=0, pad_h=0, **kwargs):
        """コンストラクタ

        Args:
            output_dim(tuple, list): 出力画像の画素数
            inverse(bool): 符号の反転を行うかどうか
            activation(str): 活性化関数 "softmax"とすれば出力にsoftmax関数を適応する。
            normalization(str): 正規化 "minmax"とすれば最小値を-1、最大値を1にする
            mode(str): フィルターの作り方。"v1"か"v2"
            width(int): 検出器の幅の画素数
            height(int): 検出器の高さの画素数
            pad_w(int): 両脇の余白の画素数
            pad_h(int): 上下の余白の画素数
            **kwargs(dict): その他
        """
        super(MNISTDetector, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.inverse = inverse
        self.activation = activation
        self.normalization = normalization
        self.mode = mode
        self.width = width
        self.height = height
        self.pad_w = pad_w
        self.pad_h = pad_h

    @tf.function
    def get_photo_mask(self):
        """フォトマスクの作成

        Returns:
            tensor:10クラス分の領域だけ穴があいた行列

        """
        return tf.reduce_sum(self.filters, axis=0)

    @staticmethod
    def make_filters_v2(shape, width=None, height=None, pad_w=0, pad_h=0):
        """10クラス分のフィルターを作る

        v2仕様

        Args:
            shape(tuple, list):縦横の画素数
            width(int): 検出器の幅の画素数
            height(int): 検出器の縦の画素数
            pad_w(int): 両端の余白の画素数
            pad_h(int): 上下の余白の画素数

        Returns: 
            tensor:10クラス分の0と1で校正された行列

        """
        # dw = detector width
        # dh = detector height

        clipped_shape = (shape[0] - pad_h * 2, shape[1] - pad_w * 2)
        dw = width if width is not None else min(int(tf.floor(clipped_shape[1] / 9.0)), int(tf.floor(clipped_shape[0] / 7.0)))
        dh = height if height is not None else min(int(tf.floor(clipped_shape[1] / 9.0)), int(tf.floor(clipped_shape[0] / 7.0)))

        # 1行目
        # tdh : height of detector's top
        tdh = round(shape[0] / 2 - dh * 5 / 2)

        # 2行目の1番目と2番目の間
        left = round(pad_w + (clipped_shape[1] - dw * 4) * 3 / 10 + dw / 2.0)
        w0 = np.zeros(shape, dtype='float32')
        w0[tdh:tdh + dh, left: left + dw] = 1.0
        # w0 = w0 / np.sum(w0)
        w0 = tf.constant(w0)

        left = round(pad_w + (clipped_shape[1] / 2.0 - dw / 2.0))
        w1 = np.zeros(shape, dtype='float32')
        w1[tdh:tdh + dh, left: left + dw] = 1.0
        # w1 = w1 / np.sum(w1)
        w1 = tf.constant(w1)

        # 2行目の3番目と4番目の間
        left = round(pad_w + (clipped_shape[1] - dw * 4) * 7 / 10 + dw * 5 / 2.0)
        w2 = np.zeros(shape, dtype='float32')
        w2[tdh:tdh + dh, left: left + dw] = 1.0
        # w2 = w2 / np.sum(w2)
        w2 = tf.constant(w2)

        # 2行目
        tdh = int(tf.floor((shape[0] - dh) / 2))

        left = round(pad_w + (clipped_shape[1] - dw * 4) / 5)
        w3 = np.zeros(shape, dtype='float32')
        w3[tdh:tdh + dh, left: left + dw] = 1.0
        # w3 = w3 / np.sum(w3)
        w3 = tf.constant(w3)

        left = round(pad_w + (clipped_shape[1] - dw * 4) / 5 * 2 + dw)
        w4 = np.zeros(shape, dtype='float32')
        w4[tdh:tdh + dh, left: left + dw] = 1.0
        # w4 = w4 / np.sum(w4)
        w4 = tf.constant(w4)

        left = round(pad_w + (clipped_shape[1] - dw * 4) / 5 * 3 + dw * 2)
        w5 = np.zeros(shape, dtype='float32')
        w5[tdh:tdh + dh, left: left + dw] = 1.0
        # w5 = w5 / np.sum(w5)
        w5 = tf.constant(w5)

        left = round(pad_w + (clipped_shape[1] - dw * 4) / 5 * 4 + dw * 3)
        w6 = np.zeros(shape, dtype='float32')
        w6[tdh:tdh + dh, left: left + dw] = 1.0
        # w6 = w6 / np.sum(w6)
        w6 = tf.constant(w6)

        # 3行目
        tdh = round(shape[0] / 2 + dh * 3 / 2)

        # 2行目の1番目と2番目の間
        left = round(pad_w + (clipped_shape[1] - dw * 4) * 3 / 10 + dw / 2.0)
        w7 = np.zeros(shape, dtype='float32')
        w7[tdh:tdh + dh, left: left + dw] = 1.0
        # w7 = w7 / np.sum(w7)
        w7 = tf.constant(w7)

        left = round(pad_w + (clipped_shape[1] / 2.0 - dw / 2.0))
        w8 = np.zeros(shape, dtype='float32')
        w8[tdh:tdh + dh, left: left + dw] = 1.0
        # w8 = w8 / np.sum(w8)
        w8 = tf.constant(w8)

        # 2行目の3番目と4番目の間
        left = round(pad_w + (clipped_shape[1] - dw * 4) * 7 / 10 + dw * 5 / 2.0)
        w9 = np.zeros(shape, dtype='float32')
        w9[tdh:tdh + dh, left: left + dw] = 1.0
        # w9 = w9 / np.sum(w9)
        w9 = tf.constant(w9)

        return tf.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], axis=0)

    @staticmethod
    def make_filters_v1(shape):
        width = min(int(tf.floor(shape[1] / 9.0)), int(tf.floor(shape[0] / 7.0)))
        height = min(int(tf.floor(shape[1] / 9.0)), int(tf.floor(shape[0] / 7.0)))

        w0 = np.zeros(shape, dtype='float32')
        w0[2 * height:3 * height, width:2 * width] = 1.0
        w0 = tf.constant(w0)

        w1 = np.zeros(shape, dtype='float32')
        w1[2 * height:3 * height, 4 * width:5 * width] = 1.0
        w1 = tf.constant(w1)

        w2 = np.zeros(shape, dtype='float32')
        w2[2 * height:3 * height, 7 * width:8 * width] = 1.0
        w2 = tf.constant(w2)

        w3 = np.zeros(shape, dtype='float32')
        w3[4 * height:5 * height, 1 * width:2 * width] = 1.0
        w3 = tf.constant(w3)

        w4 = np.zeros(shape, dtype='float32')
        w4[4 * height:5 * height, 3 * width:4 * width] = 1.0
        w4 = tf.constant(w4)

        w5 = np.zeros(shape, dtype='float32')
        w5[4 * height:5 * height, 5 * width:6 * width] = 1.0
        w5 = tf.constant(w5)

        w6 = np.zeros(shape, dtype='float32')
        w6[4 * height:5 * height, 7 * width:8 * width] = 1.0
        w6 = tf.constant(w6)

        w7 = np.zeros(shape, dtype='float32')
        w7[6 * height:7 * height, width:2 * width] = 1.0
        w7 = tf.constant(w7)

        w8 = np.zeros(shape, dtype='float32')
        w8[6 * height:7 * height, 4 * width:5 * width] = 1.0
        w8 = tf.constant(w8)

        w9 = np.zeros(shape, dtype='float32')
        w9[6 * height:7 * height, 7 * width:8 * width] = 1.0
        w9 = tf.constant(w9)

        return tf.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], axis=0)

    @staticmethod
    def plot(shape, width=None, height=None, pad_w=0, pad_h=0, ax=None):
        """可視化

        検出器の可視化用

        Args:
            shape(tuple, list):縦横の画素数
            width(int): 検出器の幅の画素数
            height(int): 検出器の縦の画素数
            pad_w(int): 両端の余白の画素数
            pad_h(int): 上下の余白の画素数
            ax(matplotlib.pyploy.Axes): プロットするときのAxes。指定しなくても良い。

        """
        image = tf.reduce_sum(MNISTDetector.make_filters_v2(shape, width, height, pad_w, pad_h), axis=0)
        if ax:
            ax.imshow(image.numpy())
        else:
            fig = plt.figure()
            _ax = fig.add_subplot()
            _ax.imshow(image.numpy())

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "inverse": self.inverse,
            "activation": self.activation,
            "normalization": self.normalization,
            "mode": self.mode,
            "width": self.width,
            "height": self.height,
            "pad_w": self.pad_w,
            "pad_h": self.pad_h
        })
        return config

    @classmethod
    def from_config(cls, config):
        """クラスの初期化

        get_config()で得られる辞書方配列からクラスのインスタンスを作り直す

        Args:
            config (dict):  コンストラクタの引数を鍵に持つ辞書型配列

        Returns: 
            object: configで指定した属性を持つこのクラスのインスタンス

        """
        return cls(**config)

    def build(self, input_shape):
        """行列の設定

        modeに従って検出器用の行列を作る。

        Args:
            input_shape(tuple, list): 入力画像の画素数

        """
        self.input_dim = input_shape

        if self.mode == "v2":
            if self.inverse:
                self.filters = -self.make_filters_v2([self.input_dim[-2], self.input_dim[-1]], self.width, self.height, self.pad_w, self.pad_h)
            else:
                self.filters = self.make_filters_v2([self.input_dim[-2], self.input_dim[-1]], self.width, self.height, self.pad_w, self.pad_h)
        else:
            if self.inverse:
                self.filters = -self.make_filters_v1([self.input_dim[-2], self.input_dim[-1]])
            else:
                self.filters = self.make_filters_v1([self.input_dim[-2], self.input_dim[-1]])

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        出力層から各ディテクターのエリアでの総和を計算する。

        Args:
            x(tensor): ミニバッチにおける出力層の分布

        Returns: 
            tensor:各検出器でのシグナル

        """
        y = tf.tensordot(x, self.filters, axes=[[1, 2], [1, 2]])

        if self.normalization == 'minmax':
            maximum = tf.reduce_max(y)
            minimum = tf.reduce_min(y)
            y = (y - minimum) / (maximum - minimum)

        if self.activation == 'softmax':
            y = tf.nn.softmax(y)

        return y


class CircleOnCircumferenceDetector(tf.keras.layers.Layer):
    """円周上にある円形検出器

    Args:
        output_dim(int): クラスの数(検出器の数)
        r1(float): 円周の半径
        r2(float): 検出器の半径
        activation: 活性化関数
        normalization: 正規化
        name: レイヤーの名前

    """
    def __init__(self, output_dim, r1, r2, activation=None, normalization=None, name="circle_on_circumference_detector", **kwargs):
        """コンストラクタ

        Args:
            output_dim(int): クラスの数(検出器の数)
            r1(float): 円周の半径
            r2(float): 検出器の半径
            activation: 活性化関数
            normalization: 正規化
            name: レイヤーの名前
            **kwargs:
        """
        super(CircleOnCircumferenceDetector, self).__init__(name=name, **kwargs)
        assert 0 < r1
        assert 0 < r2
        self.output_dim = output_dim
        self.r1 = r1
        self.r2 = r2
        self.activation = activation
        self.normalization = normalization

    @staticmethod
    def make_filters(shape, r1, r2, class_num):
        """各クラスに対応する行列の作成

        各クラスに対応する検出器の出力を計算するための行列をクラスの数分作成する。

        Args:
            shape(tuple, list):入力画像の画素数
            r1(float): 円周の半径
            r2(float): 検出器の半径
            class_num(int): クラスの数

        Returns: 
            tensor:各検出器のシグナルを計算する行列

        """
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
            _filter = np.where((xx - p) ** 2 + (yy - q) ** 2 <= r2 ** 2, 1, 0)
            f_list.append(_filter / np.sum(_filter))
        return tf.constant(np.array(f_list), dtype=tf.float32)

    @staticmethod
    def plot(shape, r1, r2, class_num, ax=None):
        """可視化

        検出器の位置、大きさの確認のための関数

        Args:
            shape(tuple, list):入力画像の画素数
            r1(float): 円周の半径
            r2(float): 検出器の半径
            class_num(int): クラスの数
            ax(matplotlib.pyplot.Axes): プロットエリア

        """
        filters = CircleOnCircumferenceDetector.make_filters(shape, r1, r2, class_num)
        sum_image = tf.reduce_sum(filters, axis=0)
        if ax:
            ax.imshow(sum_image.numpy())
        else:
            fig = plt.figure()
            _ax = fig.add_subplot()
            _ax.imshow(sum_image.numpy())

    @tf.function
    def get_photo_mask(self):
        """フォトマスクの作成

        Returns:
            tensor:全ての検出器を含むフォトマスクの行列

        """
        return tf.reduce_sum(self.filters, axis=0)

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "r1": self.r1,
            "r2": self.r2,
            "activation": self.activation,
            "normalization": self.normalization
        })
        return config

    def build(self, input_dim):
        """検出器の行列の決定

        input_dimから得られる入力の画素数からmake_filtersにて角検出器の行列を作成する。

        Args:
            input_dim(tuple, list):入力画像の画素数

        """
        self.input_dim = input_dim

        self.filters = self.make_filters([self.input_dim[-2], self.input_dim[-1]], self.r1, self.r2, self.output_dim)

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        角検出器のシグナルを計算する。

        Args:
            x:ミニバッチにおける出力層の分布

        Returns:
            tensor:角検出器のシグナル

        """
        y = tf.tensordot(x, self.filters, axes=[[1, 2], [1, 2]])

        if self.normalization == 'minmax':
            maximum = tf.reduce_max(y)
            minimum = tf.reduce_min(y)
            y = (y - minimum) / (maximum - minimum)

        if self.activation == 'softmax':
            y = tf.nn.softmax(y)

        return y


# daaa
class PhotoMask(tf.keras.layers.Layer):
    """フォトマスク

    四角い穴が行列形式で並んでいるフォトマスクを作成する。

    """
    def __init__(self, output_dim, row=1, col=1, width=None, height=None, pad_w=0, pad_h=0, **kwargs):
        """コンストラクタ

        Args:
            output_dim(tuple, list): 入力画像の画素数
            row(int): 穴の行数
            col(int): 穴の列数
            width(int): 穴の幅のピクセル数
            height(int): 穴の縦のピクセル数
            pad_w(int): 左右の余白のピクセル数
            pad_h(int):　上下の余白のピクセル数
            **kwargs(dict):その他
        """
        super(PhotoMask, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.row = row
        self.col = col
        self.width = width
        self.height = height
        self.pad_w = pad_w
        self.pad_h = pad_h

    @tf.function
    def get_photo_mask(self):
        """フォトマスクの取得

        自身の属性に基づいて作成したフォトマスクの行列を返す

        Returns:
            tensor:フォトマスクの行列

        """
        return self.make_photo_mask(self.input_dim, self.row, self.col, self.width, self.height, self.pad_w, self.pad_h)

    @staticmethod
    def make_photo_mask(shape, row, col, width=None, height=None, pad_w=0, pad_h=0):
        """フォトマスクの作成

        widh*heightの穴の穴がrow行col列並んでいるフォトマスクを作成する。光が通り抜ける場所は1その他は0で構成される行列である。

        Args:
            shape(tuple, list): フォトマスク全体のピクセル数
            row(int): 穴の列数
            col(int): 穴の行数
            width(int): 穴の幅のピクセル数
            height(int): 穴の縦のピクセル数
            pad_w(int): 左右の余白のピクセル数
            pad_h(int):　上下の余白のピクセル数

        Returns: 
            tensor:フォトマスクの行列

        """
        # dw = detector width
        # dh = detector height

        clipped_shape = (shape[0] - pad_h * 2, shape[1] - pad_w * 2)
        dw = width if width is not None else int(clipped_shape[1] / (2 * col + 1))
        dh = height if height is not None else int(clipped_shape[0] / (2 * row + 1))

        mask = np.zeros(shape, np.float64)
        for r in range(row):
            for c in range(col):
                interval_w = (clipped_shape[1] - col * dw) / (col + 1)
                interval_h = (clipped_shape[0] - row * dh) / (row + 1)
                up_left_col = pad_w + int(interval_w * (c + 1) + dw * c)
                up_left_row = pad_h + int(interval_h * (r + 1) + dh * r)
                mask[up_left_row:up_left_row + dh, up_left_col:up_left_col + dw] = 1.

        return mask

    @staticmethod
    def plot(shape, row, col, width=None, height=None, pad_w=0, pad_h=0, ax=None):
        """フォトマスクの可視化

        フォトマスクの形状を確認するための可視化関数。

        Args:
            shape(tuple, list): フォトマスク全体のピクセル数
            row(int): 穴の列数
            col(int): 穴の行数
            width(int): 穴の幅のピクセル数
            height(int): 穴の縦のピクセル数
            pad_w(int): 左右の余白のピクセル数
            pad_h(int):　上下の余白のピクセル数
            ax(matplitlib.pyplot.Axes, optional): プロットエリア

        """
        image = PhotoMask.make_photo_mask(shape, row, col, width, height, pad_w, pad_h)
        if ax:
            ax.imshow(image.numpy())
        else:
            fig = plt.figure()
            _ax = fig.add_subplot()
            _ax.imshow(image.numpy())

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "row": self.row,
            "col": self.col,
            "width": self.width,
            "height": self.height,
            "pad_w": self.pad_w,
            "pad_h": self.pad_h
        })
        return config

    @classmethod
    def from_config(cls, config):
        """クラスの初期化

        get_config()で得られる辞書方配列からクラスのインスタンスを作り直す

        Args:
            config (dict):  コンストラクタの引数を鍵に持つ辞書型配列

        Returns: 
            object:configで指定した属性を持つこのクラスのインスタンス

        """
        return cls(**config)

    def build(self, input_shape):
        """フォトマスクの決定

        input_shapeから、入力される画素数がわかるので、フォトマスクの形状を決定する。

        Args:
            input_shape(tuple, list): 入力画像の画素数

        """
        self.input_dim = input_shape
        _mask = self.make_photo_mask([self.input_dim[-2], self.input_dim[-1]], self.row, self.col, self.width, self.height, self.pad_w, self.pad_h)
        self.mask = tf.cast(tf.complex(_mask, 0. * _mask), dtype=tf.complex64)

    def call(self, x, **kwargs):
        """純伝搬の計算時に呼ばれる関数

        フォトマスクの透過光の計算

        Args:
            x(tensor): 左右円偏光の電場分布もしくは光強度分布
            **kwargs(dict): その他

        Returns:
            tensor:フォトマスク透過光

        """
        y = x * self.mask
        return y


class FaradayRotationByStokes(tf.keras.layers.Layer):
    """ストークスパラメータ法による偏光面の角度の検出

    左右円偏光の電場からストークスパラメータ法によって偏光面の角度を検出する。基本的には出漁層にて使うクラスである。

    Args:
        output_dim(tuple, list): 入力画像の画素数
        normalization(str): "minmax"なら最小値を-1、最大値を1にする
        eps(float): 0除算回避のための微少量

    """
    def __init__(self, output_dim, normalization=None, eps=1.0e-20):
        """コンストラクタ

        Args:
            output_dim(tuple, list): 入力画像の画素数
            normalization(str): "minmax"なら最小値を-1、最大値を1にする
            eps(float): 0除算回避のための微少量
        """
        super(FaradayRotationByStokes, self).__init__()
        self.output_dim = output_dim
        self.normalization = normalization
        self.eps = eps

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "normalization": self.normalization,
            "eps": self.eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        """クラスの初期化

        get_config()で得られる辞書方配列からクラスのインスタンスを作り直す

        Args:
            config (dict):  コンストラクタの引数を鍵に持つ辞書型配列

        Returns: 
            object:configで指定した属性を持つこのクラスのインスタンス

        """

        return cls(**config)

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        ストークスパラメータ法によって偏光面の角度を求める。

        Args:
            x(tensor): 左右円偏光の電場分布

        Returns: 
            tensor:偏光面の角度分布

        """
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(x)
        rcp_y = 1.0j * rcp_x
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, :, :])(x)
        lcp_y = -1.0j * lcp_x

        E0 = rcp_x + lcp_x
        I0 = tf.abs(E0) ** 2 / 2.0
        E90 = rcp_y + lcp_y
        I90 = tf.abs(E90) ** 2 / 2.0
        E45_x = (rcp_x + rcp_y + lcp_x + lcp_y) / 2.0
        E45_y = (rcp_x + rcp_y + lcp_x + lcp_y) / 2.0
        I45 = tf.abs(E45_x) ** 2 / 2 + tf.abs(E45_y) ** 2 / 2.0
        E135_x = (rcp_x - rcp_y + lcp_x - lcp_y) / 2.0
        E135_y = (-rcp_x + rcp_y - lcp_x + lcp_y) / 2.0
        I135 = tf.abs(E135_x) ** 2 / 2 + tf.abs(E135_y) ** 2 / 2.0

        S1 = I0 - I90
        S2 = I45 - I135

        # theta = tf.where(S1**2 > self.eps, tf.atan(S2*S1 / S1**2) / 2.0, tf.atan(S2*S1 / self.eps) / 2.0,)
        theta = tf.atan(S2 * S1 / (S1 ** 2 + self.eps)) / 2.0

        if self.normalization == 'minmax':
            minimum = tf.reduce_min(theta, axis=[1, 2], keepdims=True)
            maximum = tf.reduce_max(theta, axis=[1, 2], keepdims=True)
            theta = (theta - minimum) / (maximum - minimum)

        return theta


class FaradayRotationByPhaseDifference(tf.keras.layers.Layer):
    """位相差から偏光面の角度を検出する。

    左右円偏光の位相差から偏光面の角度を検出する。基本的には検出層にて使うクラスである。

    """
    def __init__(self):
        """コンストラクタ

        """
        super(FaradayRotationByPhaseDifference, self).__init__()

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        左右円偏光の偏角を計算し、それらの差の半分より、偏光面の角度を計算する。

        Args:
            x:

        Returns:

        """
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, :, :])(x)
        phase_rcp = tf.math.log(rcp_x)
        phase_lcp = tf.math.log(lcp_x)
        return tf.math.imag(-phase_rcp + phase_lcp) / 2


class Polarizer(tf.keras.layers.Layer):
    """偏光子、検光子

    """
    def __init__(self, output_dim, phi=0.0, trainable=False):
        """コンストラクタ

        Args:
            output_dim(tuple, list): 入力画像の画素数
            phi(float): 偏光子、検光子の角度
            trainable(bool): 偏光子、検光子の角度を学習するかどうか
        """
        super(Polarizer, self).__init__()
        self.output_dim = output_dim
        self.phi = tf.Variable(phi, name="phi", trainable=trainable)
        self.trainable = trainable

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "phi": float(self.phi.numpy()),
            "trainable": self.trainable
        })
        return config

    @classmethod
    def from_config(cls, config):
        """クラスの初期化

        get_config()で得られる辞書方配列からクラスのインスタンスを作り直す

        Args:
            config (dict):  コンストラクタの引数を鍵に持つ辞書型配列

        Returns:
            object:configで指定した属性を持つこのクラスのインスタンス

        """
        return cls(**config)

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        Args:
            x(tensor):左右円偏光の電場分布

        Returns:偏光子、検光子を通した後の左右円偏光の電場分布

        """
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(x)
        rcp_y = 1.0j * rcp_x
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, :, :])(x)
        lcp_y = -1.0j * lcp_x

        # Ercp = T@Ercp

        # Tr00 = Tr[0,0]
        tr00 = tf.complex(tf.cos(self.phi) ** 2, -tf.sin(2 * self.phi) / 2) / 2.0
        tr01 = tf.complex(tf.sin(2 * self.phi) / 2, -tf.sin(self.phi) ** 2) / 2.0

        tl00 = tf.complex(tf.cos(self.phi) ** 2, tf.sin(2 * self.phi) / 2) / 2.0
        tl01 = tf.complex(tf.sin(2 * self.phi) / 2, tf.sin(self.phi) ** 2) / 2.0

        rcp_x_pol = tr00 * (rcp_x + lcp_x) + tr01 * (rcp_y + lcp_y)
        lcp_x_pol = tl00 * (rcp_x + lcp_x) + tl01 * (rcp_y + lcp_y)

        return tf.stack([rcp_x_pol, lcp_x_pol], axis=1)


class PhaseToPeriodic(tf.keras.layers.Layer):
    """位相の周期化

    -piとpiを連続値で繋ぐためのクラス

    """
    def __init__(self, output_dim):
        """コンストラクタ

        Args:
            output_dim(tuple, list): 入力画像の画素数
        """
        super(PhaseToPeriodic, self).__init__()
        self.output_dim = output_dim

    def get_config(self):
        """設定の取得

        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        """クラスの初期化

        get_config()で得られる辞書方配列からクラスのインスタンスを作り直す

        Args:
            config (dict):  コンストラクタの引数を鍵に持つ辞書型配列

        Returns:
            object:configで指定した属性を持つこのクラスのインスタンス

        """
        return cls(**config)

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        Args:
            x(tensor):位相分布

        Returns: 周期化した位相分布

        """
        return tf.sin(x)


class MNISTDifferentialDetector(tf.keras.layers.Layer):
    """差分検出器

    差分検出法による検出

    """
    def __init__(self, output_dim, activation=None, normalization=None, **kwargs):
        """コンストラクタ

        Args:
            output_dim(tuple, list): 入力画像の画素数
            activation(str):活性化関数
            normalization(str):正規化方法
            **kwargs:
        """
        super(MNISTDifferentialDetector, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation
        self.normalization = normalization

    def get_config(self):
        """設定の取得
        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "inverse": self.inverse,
            "activation": self.activation,
            "normalization": self.normalization
        })
        return config

    @classmethod
    def from_config(cls, config):
        """クラスの初期化
        get_config()で得られる辞書方配列からクラスのインスタンスを作り直す

        Args:
            config (dict):  コンストラクタの引数を鍵に持つ辞書型配列

        Returns: configで指定した属性を持つこのクラスのインスタンス

        """
        return cls(**config)

    @staticmethod
    def make_positive_filter(input_shape):
        """正の検出器

        差分における引かれる法の検出器

        Args:
            input_shape(tuple, list):出力層の画素数

        Returns: 正の検出器の行列

        """
        width = int(input_shape[-1] / 10)
        height = min(int(input_shape[-2] / 2 / 5), width)
        pad_width = int(np.round(width / 2))
        pad_height = int(np.round((input_shape[-1] / 2 - height * 3) / 2))

        w0 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w0[pad_height:pad_height + height, pad_width:pad_width + width] = 1.0
        w0 = tf.constant(w0)

        w1 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w1[pad_height:pad_height + height, pad_width + width * 2:pad_width + width * 3] = 1.0
        w1 = tf.constant(w1)

        w2 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w2[pad_height:pad_height + height, pad_width + width * 4:pad_width + width * 5] = 1.0
        w2 = tf.constant(w2)

        w3 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w3[pad_height:pad_height + height, pad_width + width * 6:pad_width + width * 7] = 1.0
        w3 = tf.constant(w3)

        w4 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w4[pad_height:pad_height + height, pad_width + width * 8:pad_width + width * 9] = 1.0
        w4 = tf.constant(w4)

        w5 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w5[pad_height + height * 2:pad_height + height * 3, pad_width:pad_width + width] = 1.0
        w5 = tf.constant(w5)

        w6 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w6[pad_height + height * 2:pad_height + height * 3, pad_width + width * 2:pad_width + width * 3] = 1.0
        w6 = tf.constant(w6)

        w7 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w7[pad_height + height * 2:pad_height + height * 3, pad_width + width * 4:pad_width + width * 5] = 1.0
        w7 = tf.constant(w7)

        w8 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w8[pad_height + height * 2:pad_height + height * 3, pad_width + width * 6:pad_width + width * 7] = 1.0
        w8 = tf.constant(w8)

        w9 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w9[pad_height + height * 2:pad_height + height * 3, pad_width + width * 8:pad_width + width * 9] = 1.0
        w9 = tf.constant(w9)

        return tf.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], axis=-1)

    @staticmethod
    def make_negative_filter(input_shape):
        """負の検出器

        差分検出における引く方

        Args:
            input_shape(tuple, list): 出力層の画素数

        Returns: 負の検出器の行列

        """
        width = int(input_shape[-1] / 10)
        height = min(int(input_shape[-2] / 2 / 5), width)

        margin_height = int(input_shape[-2] / 2)

        pad_width = int(np.round(width / 2))
        pad_height = int(np.round((input_shape[-1] / 2 - height * 3) / 2)) + margin_height

        w0 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w0[pad_height:pad_height + height, pad_width:pad_width + width] = 1.0
        w0 = tf.constant(w0)

        w1 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w1[pad_height:pad_height + height, pad_width + width * 2:pad_width + width * 3] = 1.0
        w1 = tf.constant(w1)

        w2 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w2[pad_height:pad_height + height, pad_width + width * 4:pad_width + width * 5] = 1.0
        w2 = tf.constant(w2)

        w3 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w3[pad_height:pad_height + height, pad_width + width * 6:pad_width + width * 7] = 1.0
        w3 = tf.constant(w3)

        w4 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w4[pad_height:pad_height + height, pad_width + width * 8:pad_width + width * 9] = 1.0
        w4 = tf.constant(w4)

        w5 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w5[pad_height + height * 2:pad_height + height * 3, pad_width:pad_width + width] = 1.0
        w5 = tf.constant(w5)

        w6 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w6[pad_height + height * 2:pad_height + height * 3, pad_width + width * 2:pad_width + width * 3] = 1.0
        w6 = tf.constant(w6)

        w7 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w7[pad_height + height * 2:pad_height + height * 3, pad_width + width * 4:pad_width + width * 5] = 1.0
        w7 = tf.constant(w7)

        w8 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w8[pad_height + height * 2:pad_height + height * 3, pad_width + width * 6:pad_width + width * 7] = 1.0
        w8 = tf.constant(w8)

        w9 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w9[pad_height + height * 2:pad_height + height * 3, pad_width + width * 8:pad_width + width * 9] = 1.0
        w9 = tf.constant(w9)

        return tf.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], axis=-1)

    @staticmethod
    def plot(input_shape):
        """可視化

        可視化テスト用

        Args:
            input_shape(tuple, list): 出力層の画素数

        """
        positive = MNISTDifferentialDetector.make_positive_filter(input_shape)
        negative = MNISTDifferentialDetector.make_negative_filter(input_shape)
        image = tf.reduce_sum(positive, axis=-1) + -1 * tf.reduce_sum(negative, axis=-1)
        plt.imshow(image)

    def build(self, input_shape):
        """検出器の決定

        input_shapeから画素数がわかるので、正の検出器と負の検出器の行列を作成する。

        Args:
            input_shape(tuple, list): 出力面での画素数

        """
        self.input_dim = input_shape
        self.positive_filter = self.make_positive_filter(input_shape)
        self.negative_filter = self.make_negative_filter(input_shape)

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数
        差分検出法による検出を行う。

        Args:
            x(tensor):出力層の分布

        Returns:差分検出によるシグナル
            tensor:

        """
        y_positive = tf.tensordot(x, self.positive_filter, axes=[[1, 2], [0, 1]])
        y_negative = tf.tensordot(x, self.negative_filter, axes=[[1, 2], [0, 1]])

        y = y_positive - y_negative

        if self.normalization == 'minmax':
            maximum = tf.reduce_max(y)
            minimum = tf.reduce_min(y)
            y = (y - minimum) / (maximum - minimum)

        if self.activation == 'softmax':
            y = tf.nn.softmax(y)

        return y


class RandomPolarization(tf.keras.layers.Layer):
    """ランダム偏光

    ランダム位相を導入するためのクラス

    """
    def __init__(self, input_dist=None, name=None, dtype=tf.float32, **kwargs):
        """コンストラクタ

        Args:
            input_dist(tensor): 光強度分布
            name(str): レイヤーの名前
            dtype(tensorflow.dtype): データ型
            **kwargs:
        """
        self.input_dist = None  # input light distribution. if it is None, uniform distribution
        super(RandomPolarization, self).__init__(
            trainable=False,
            name=name,
            dtype=dtype,
            **kwargs
        )

    def build(self, input_dim):
        """初期の左右円偏光の決定

        偏光面の角度が0で光強度分布がinput_distになるような左右円偏光の電場を作成する。

        Args:
            input_dim(tuple, list):入力画像の画素数

        """
        E0 = tf.constant(self.input_dist) if self.input_dist is not None else tf.ones((input_dim[-2], input_dim[-1]))
        self.rcp_x = tf.complex(tf.sqrt(E0 / 2.0), 0.0 * E0)
        self.lcp_x = tf.complex(tf.sqrt(E0 / 2.0), 0.0 * E0)
        super(RandomPolarization, self).build(input_dim)

    def call(self, x):
        """純伝搬の計算時に呼ばれる関数

        乱数を受け取って、その分の偏光面を回転させる。

        Args:
            x(tensor): 乱数

        Returns:
            tensor:ランダム偏光の左右円偏光

        """
        phi = tf.complex(x, 0.0 * x)

        rcp_x_mo = self.rcp_x * tf.exp(-1.j * phi)
        lcp_x_mo = self.lcp_x * tf.exp(1.j * phi)
        return tf.stack([rcp_x_mo, lcp_x_mo], axis=1)
