import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Custom', name='symmetric_shift_l1')
class SymmetricShiftL1Regularizer(tf.keras.regularizers.Regularizer):
    """マンハッタン距離の正則化

    2値化を助長するように正則化を加えるためのクラス

    Args:
        l1(float):正則化の強さ
        shift(float):罰則が弱くなるところ

    """

    def __init__(self, l1=0., shift=0.):
        """コンストラクタ

        Args:
            l1: 正則化の強さ
            shift: 罰則が低くなるところ
        """
        self.l1 = l1
        self.shift = shift

    def __call__(self, x):
        """正則化の時に呼ばれる関数

        Args:
            x: 学習パラメータ

        Returns:
            float: 罰則の大きさ

        """
        return self.l1 * tf.math.reduce_sum(tf.math.abs(self.shift - tf.math.abs(x)))

    def get_config(self):
        """設定の取得
        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """

        return {'l1': float(self.l1), 'shift': float(self.shift)}


@tf.keras.utils.register_keras_serializable(package='Custom', name='shift_l2')
class SymmetricShiftL2Regularizer(tf.keras.regularizers.Regularizer):
    """ユークリッド距離の正則化

    2値化を助長するように正則化を加えるためのクラス

    Args:
        l2(float):正則化の強さ
        shift(float):罰則が弱くなるところ


    """
    def __init__(self, l2=0., shift=0.):
        """コンストラクタ

        Args:
            l2: 罰則の強さ
            shift: 罰則が低くなるところ
        """
        self.l1 = l2
        self.shift = shift

    def __call__(self, x):
        """正則化の時に呼ばれる関数

        Args:
            x: 学習パラメータ

        Returns:
            float:罰則の大きさ

        """
        return self.l1 * tf.math.reduce_sum(tf.math.abs((self.shift**2 - x**2)))

    def get_config(self):
        """設定の取得
        コンストラクタに渡した引数を取り出す

        Returns:
            dict: インスタンスが持つ属性の辞書型配列
        """

        return {'l2': float(self.l1), 'shift': float(self.shift)}
