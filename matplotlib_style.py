import matplotlib.pyplot as plt

def main():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18  # 適当に必要なサイズに
    plt.rcParams['xtick.direction'] = 'in'  # in or out
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.xmargin'] = 0.01
    plt.rcParams['axes.ymargin'] = 0.01
    plt.rcParams["legend.fancybox"] = False  # 丸角OFF
    plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更


if __name__ == "__main__":
    main()