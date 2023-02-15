"""光学チャートモジュール

* 一般的なフォトマスクの作製をする

"""

import numpy as np

def char1(size, inverse=False):
    """
    char1関数は、0と1で構成される画像を生成します。
    inverse引数はオプションで、これが真の場合、配列内のすべての値が反転されます（白から黒、またはその逆）。


    Args:
        size (list): チャートのピクセルサイズ
        inverse (bool): 0,1の反転

    Returns:
        numpy.array: 光学チャートの2次元配列
    """

    chart = np.zeros(size)
    w1 = int(size[1] / 10)
    chart[:, 0:w1] = 1
    chart[:, w1 * 2:w1 * 3] = 1
    chart[:, w1 * 4:w1 * 5] = 1

    w2 = int(size[0] / 10)
    chart[0: w2, w1 * 5:] = 1
    chart[w2 * 2: w2 * 3, w1 * 5:] = 1
    chart[w2 * 4: w2 * 5, w1 * 5:] = 1

    w3 = int(size[1] / 20)
    chart[w2 * 5:, w1 * 5 + w3 * 1:w1 * 5 + w3 * 2] = 1
    chart[w2 * 5:, w1 * 5 + w3 * 3:w1 * 5 + w3 * 4] = 1

    w4 = int(size[0] / 20)
    chart[w2 * 5 + w4: w2 * 5 + w4 * 2, w1 * 5 + w3 * 4:] = 1
    chart[w2 * 5 + w4 * 3: w2 * 5 + w4 * 4, w1 * 5 + w3 * 4:] = 1

    w5 = int(size[1] / 40)
    chart[w2 * 5 + w4 * 4:, w1 * 5 + w3 * 4 + w5: w1 * 5 + w3 * 4 + w5 * 2] = 1
    chart[w2 * 5 + w4 * 4:, w1 * 5 + w3 * 4 + w5 * 3: w1 * 5 + w3 * 4 + w5 * 4] = 1
    chart[w2 * 5 + w4 * 4:, w1 * 5 + w3 * 4 + w5 * 5: w1 * 5 + w3 * 4 + w5 * 6] = 1

    w6 = int(size[0] / 40)
    chart[w2 * 5 + w4 * 4 + w6: w2 * 5 + w4 * 4 + w6 * 2, w1 * 5 + w3 * 4 + w5 * 6:] = 1
    chart[w2 * 5 + w4 * 4 + w6 * 3: w2 * 5 + w4 * 4 + w6 * 4, w1 * 5 + w3 * 4 + w5 * 6:] = 1
    chart[w2 * 5 + w4 * 4 + w6 * 5: w2 * 5 + w4 * 4 + w6 * 6, w1 * 5 + w3 * 4 + w5 * 6:] = 1

    if inverse:
        chart = (chart - 1) * -1

    return chart
