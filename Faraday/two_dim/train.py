import time
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from module.lib.layers import *
from module.lib import regularizer

tf.random.set_seed(1)

print("TensorFlow:", tf.__version__)
print("Python:", sys.version)

plt.rcParams['font.size'] = 18

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

#@title デフォルトのタイトル テキスト
wavelength = 532.0e-9 #@param {type:"number"}
d = 1.0e-6 #@param {type:"number"}
n = 1.5 #@param {type:"number"}

def create_model(**kwargs):
    shape = (100, 100)
    inputs = tf.keras.Input((28, 28))
    theta = -2.8/2 * np.pi/180
    eta = np.arctan(1.24/2 * np.pi/180)
    l1=kwargs["l1"]
    print(l1)
    x = ImageResizing(shape)(inputs)
    x = ImageBinarization(0.5, 0.0, 1.0)(x)
    x = IntensityToElectricField(shape)(x)
    x = MO(shape, limitation='sin', theta=theta, eta=eta, kernel_regularizer=regularizer.ShiftL1Regularizer(l1, np.pi/2))(x)
    x = AngularSpectrum(shape, wavelength=wavelength, z=0.7e-3, d=d, n=1.51, method='expand')(x)
    x = MO(shape, limitation='sin', theta=theta, eta=eta, kernel_regularizer=regularizer.ShiftL1Regularizer(l1, np.pi/2))(x)
    x = AngularSpectrum(shape, wavelength=wavelength, z=0.7e-3, d=d, n=1.0, method='expand')(x)
    x = FaradayRotation(shape)(x)
    # x = Polarizer(shape)(x)
    #x =ElectricFieldToIntensity(shape)(x)
    # x = Argument(shape)(x)
    x = MNISTDetector(10)(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs, x)
    return model


model = create_model(l1=0.1)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,  # category: sparse_categorical_crossentropy
              metrics=['accuracy'])

epochs = 50
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    min_delta=0.05,
    patience=2,
)

model_name = "202206027_3"
checkpoint_path = "checkpoint/" + model_name + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# チェックポイントコールバックを作る
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

logdir = os.path.join("logs", model_name)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

result = model.fit(x_train,
                   y_train,
                   batch_size=64,
                   epochs=epochs,
                   validation_data=(x_test, y_test),
                   callbacks=[cp_callback, tensorboard_callback]
                   )

path = "trained_model/"+ model_name
model.save(path)

df = pd.DataFrame(result.history)
df.to_csv(path + "/history.csv")