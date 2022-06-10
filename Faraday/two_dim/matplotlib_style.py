import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


colors1 = ['#4158D0','#C850C0','#FFCC70']
colors2 = ['#D9AFD9','#97D9E1']
colors3 = ["#c03991", "#9f71d5", "#579efd", "#00c3ff", "#05e1f6"]
colors4 = ["#5d3798", "#7f4199", "#9b4f9a", "#b25f9c", "#c6719f"]
colors5 = ["#c6719f", "#b7639c", "#a7569b", "#944b9a", "#7f4199", "#6549a2", "#4551a7", "#0056a9", "#0065a6", "#007098", "#007787", "#287c77"]
colors6 = ["#3B376B", "#BC6092", "#E7C8FF"]
colors7 = ["#5c55a6", "#7758a6", "#8d5ca5", "#a161a4", "#b267a2", "#bc70a9", "#c579b0", "#cf82b7", "#d58fc9", "#d99cdb", "#ddaaed", "#e0b8ff"]
colors8 = ["#3B376B", "#E7C8FF"]
colors9 = ["#EE0000", "#BC6092", "#E7C8FF"]

cmap1 = LinearSegmentedColormap.from_list('custom', colors1)
cmap2 = LinearSegmentedColormap.from_list('custom', colors2)
cmap3 = LinearSegmentedColormap.from_list('custom', colors3)
cmap4 = LinearSegmentedColormap.from_list('custom', colors4)
cmap5 = LinearSegmentedColormap.from_list('custom', colors5)
cmap6 = LinearSegmentedColormap.from_list('custom', colors6)
cmap7 = LinearSegmentedColormap.from_list('custom', colors7)
cmap8 = LinearSegmentedColormap.from_list('custom', colors8)
cmap9 = LinearSegmentedColormap.from_list('custom', colors9)
cmap10 = LinearSegmentedColormap.from_list('custom', ["#597EC2", "#5B2C17", "#C69CE7"])
cmap11 = LinearSegmentedColormap.from_list('custom', ["#2A5772", "#4790BB", "#B0D7D5"])


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