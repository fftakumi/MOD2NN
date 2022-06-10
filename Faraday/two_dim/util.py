from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.size"] = 18

z = np.linspace(0, 3*np.pi, 100)
E = np.exp(-1.0j * z)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(E.imag, z, E.real, color="red")
ax.set_xlabel("image")
ax.set_zlabel("real")
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.show()