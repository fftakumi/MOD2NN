import tensorflow as tf
import numpy as np
import math


class AngularSpectrum(tf.keras.layers.Layer):
    def __init__(self,output_dim, wavelength, z=0.0, d=1.0e-6, n=1, normalization=None, method=None):
        super(AngularSpectrum, self).__init__()
        self.output_dim = output_dim
        self.wavelength = wavelength / n
        self.k = 2 * np.pi / self.wavelength
        self.z = z
        self.d = d
        self.n = n
        self.normalization = normalization
        self.method = method


    def build(self, input_dim):
        self.input_dim = input_dim

        width = self.input_dim[-1]
        height = self.input_dim[-2]
        u = np.fft.fftfreq(width, d=self.d)
        v = np.fft.fftfreq(height, d=self.d)
        UU, VV = np.meshgrid(u, v)
        w = np.where(UU ** 2 + VV ** 2 <= 1 / self.wavelength ** 2, tf.sqrt(1 / self.wavelength**2 - UU**2 - VV**2), 0).astype('float64')
        h = np.exp(1.0j * 2 * np.pi * w * self.z)

        if self.method == 'band_limited':
            du = 1/(2*width * self.d)
            dv = 1/(2*height * self.d)
            u_limit = 1/(np.sqrt((2 * du * self.z)**2 + 1)) / self.wavelength
            v_limit = 1/(np.sqrt((2 * dv * self.z)**2 + 1)) / self.wavelength
            u_filter = np.where(np.abs(UU)/(2*u_limit) <= 1/2, 1, 0)
            v_filter = np.where(np.abs(VV)/(2*v_limit) <= 1/2, 1, 0)
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
            u_limit = 1 / (np.sqrt((2 * du * self.z) ** 2 + 1)) / self.wavelength
            v_limit = 1 / (np.sqrt((2 * dv * self.z) ** 2 + 1)) / self.wavelength
            UU, VV = np.meshgrid(u, v)

            u_filter = np.where(np.abs(UU) <= u_limit, 1, 0)
            v_filter = np.where(np.abs(VV) <= v_limit, 1, 0)

            w = np.where(UU ** 2 + VV ** 2 <= 1 / self.wavelength ** 2, tf.sqrt(1 / self.wavelength ** 2 - UU ** 2 - VV ** 2), 0).astype('float64')
            h = np.exp(1.0j * 2 * np.pi * w * self.z)
            h = h * u_filter * v_filter

        self.res = tf.cast(tf.complex(h.real, h.imag), dtype=tf.complex64)

    @tf.function
    def propagation(self, cximages):
        if self.method=='band_limited':
            images_fft = tf.signal.fft2d(cximages)
            return tf.signal.ifft2d(images_fft * self.res)
        elif self.method=='expand':
            padding = [[0,0],[self.pad_upper, self.pad_upper],[self.pad_left, self.pad_left]]
            images_pad = tf.pad(cximages, paddings=padding)
            images_pad_fft = tf.signal.fft2d(images_pad)
            u_images_pad = tf.signal.ifft2d(images_pad_fft * self.res)
            u_images = tf.keras.layers.Lambda(lambda x:x[:, self.pad_upper:self.pad_upper + self.input_dim[-2], self.pad_left:self.pad_left + self.input_dim[-1]])(u_images_pad)
            return u_images
        else:
            images_fft = tf.signal.fft2d(cximages)
            return tf.signal.ifft2d(images_fft * self.res)

    @tf.function
    def call(self, x):
        rcp_x = tf.keras.layers.Lambda(lambda x:x[:,0,0,:,:])(x)
        rcp_y = tf.keras.layers.Lambda(lambda x:x[:,0,1,:,:])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x:x[:,1,0,:,:])(x)
        lcp_y = tf.keras.layers.Lambda(lambda x:x[:,1,1,:,:])(x)

        u_rcp_x = self.propagation(rcp_x)
        u_rcp_y = self.propagation(rcp_y)
        u_lcp_x = self.propagation(lcp_x)
        u_lcp_y = self.propagation(lcp_y)

        rcp = tf.stack([u_rcp_x, u_rcp_y], axis=1)
        lcp = tf.stack([u_lcp_x, u_lcp_y], axis=1)

        rl = tf.stack([rcp, lcp], axis=1)

        if self.normalization == 'max':
            maximum = tf.reduce_max(tf.abs(rl))
            rl = rl / tf.complex(maximum, 0.0*maximum)

        return rl


class CxTest(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(CxTest, self).__init__()
        self.output_dim = output_dim

    def call(self, inputs, **kwargs):

        return inputs**2


class ImageResizing(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(ImageResizing, self).__init__()
        self.output_dim = output_dim

    def call(self, x):
        x_expnad = tf.image.resize(tf.expand_dims(x, -1), self.output_dim)
        x_expnad = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0])(x_expnad)
        return x_expnad


class ImageBinarization(tf.keras.layers.Layer):
    def __init__(self, threshold, minimum, maximum):
        super(ImageBinarization, self).__init__()
        self.threshold = threshold
        self.minimum = minimum
        self.maximum = maximum

    def call(self, x):
        return tf.where(x >= self.threshold, self.maximum, self.minimum)


class ImageToElectricField(tf.keras.layers.Layer):
    def __init__(self, output_dim, a=0.5, b=0.5):
        super(ImageToElectricField, self).__init__()
        self.output_dim = output_dim
        self.a = a
        self.b = b

    @tf.function
    def call(self, x):
        rcp_x = tf.complex(tf.sqrt(x/2), 0.0*x)
        rcp_y = 1.0j * tf.complex(tf.sqrt(x/2), 0.0*x)
        lcp_x = tf.complex(tf.sqrt(x/2), 0.0*x)
        lcp_y = -1.0j * tf.complex(tf.sqrt(x/2), 0.0*x)
        rcp = tf.stack([rcp_x, rcp_y], axis=1)
        lcp = tf.stack([lcp_x, lcp_y], axis=1)
        return tf.stack([rcp, lcp], axis=1)



class CxD2NNIntensity(tf.keras.layers.Layer):
    def __init__(self, output_dim, normalization=None):
        super(CxD2NNIntensity, self).__init__()
        self.output_dim = output_dim
        self.normalization = normalization

    def call(self, x):
        rcp_x = tf.keras.layers.Lambda(lambda x:x[:,0,0,:,:])(x)
        rcp_y = tf.keras.layers.Lambda(lambda x:x[:,0,1,:,:])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x:x[:,1,0,:,:])(x)
        lcp_y = tf.keras.layers.Lambda(lambda x:x[:,1,1,:,:])(x)

        tot_x = rcp_x + lcp_x
        tot_y = rcp_y + lcp_y

        intensity = tf.abs(tot_x)**2 / 2.0 + tf.abs(tot_y)**2 / 2.0

        if self.normalization == 'max':
            intensity = intensity / tf.reduce_max(intensity)

        return intensity


class CxMO(tf.keras.layers.Layer):
    def __init__(self, output_dim, limitation=None, limitation_num=1):
        super(CxMO, self).__init__()
        self.output_dim = output_dim
        self.limitation = limitation
        self.limitation_num = limitation_num

    def build(self, input_dim):
        self.input_dim = input_dim
        self.phi = self.add_weight("phi",
                                   shape=[int(input_dim[-2]),
                                          int(input_dim[-1])])
        super(CxMO, self).build(input_dim)

    def call(self, x):
        if self.limitation == 'tanh':
            phi_lim = self.limitation_num * tf.tanh(self.phi)
        elif self.limitation == 'sin':
            phi_lim = self.limitation_num * tf.sin(self.phi)
        else:
            phi_lim = self.phi

        phi_rcp = tf.complex(tf.cos(-phi_lim), tf.sin(-phi_lim))
        phi_lcp = tf.complex(tf.cos(phi_lim), tf.sin(phi_lim))
        # phi_rcp = tf.exp(1.0j * -phi_lim)
        #
        # phi_lcp = tf.exp(1.0j * phi_lim)
        rcp_x = tf.keras.layers.Lambda(lambda x:x[:,0,0,:,:])(x)
        rcp_y = tf.keras.layers.Lambda(lambda x:x[:,0,1,:,:])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x:x[:,1,0,:,:])(x)
        lcp_y = tf.keras.layers.Lambda(lambda x:x[:,1,1,:,:])(x)

        rcp_x = rcp_x * phi_rcp
        rcp_y = rcp_y * phi_rcp
        lcp_x = lcp_x * phi_lcp
        lcp_y = lcp_y * phi_lcp

        rcp = tf.stack([rcp_x, rcp_y], axis=1)
        lcp = tf.stack([lcp_x, lcp_y], axis=1)
        return tf.stack([rcp, lcp], axis=1)


class D2NNMNISTDetector(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None, normalization=None, avoid_zero=False, **kwargs):
        super(D2NNMNISTDetector, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation
        self.normalization = normalization
        self.avoid_zero = avoid_zero

    def build(self, input_shape):
        self.input_dim = input_shape
        width = min(int(tf.floor(self.input_dim[2] / 9.0)), int(tf.floor(self.input_dim[1] / 7.0)))
        height = min(int(tf.floor(self.input_dim[2] / 9.0)), int(tf.floor(self.input_dim[1] / 7.0)))

        w0 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w0[2 * height:3 * height, width:2 * width] = 1.0
        w0 = tf.constant(w0)

        w1 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w1[2 * height:3 * height, 4 * width:5 * width] = 1.0
        w1 = tf.constant(w1)

        w2 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w2[2 * height:3 * height, 7 * width:8 * width] = 1.0
        w2 = tf.constant(w2)

        w3 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w3[4 * height:5 * height, 1 * width:2 * width] = 1.0
        w3 = tf.constant(w3)

        w4 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w4[4 * height:5 * height, 3 * width:4 * width] = 1.0
        w4 = tf.constant(w4)

        w5 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w5[4 * height:5 * height, 5 * width:6 * width] = 1.0
        w5 = tf.constant(w5)

        w6 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w6[4 * height:5 * height, 7 * width:8 * width] = 1.0
        w6 = tf.constant(w6)

        w7 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w7[6 * height:7 * height, width:2 * width] = 1.0
        w7 = tf.constant(w7)

        w8 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w8[6 * height:7 * height, 4 * width:5 * width] = 1.0
        w8 = tf.constant(w8)

        w9 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w9[6 * height:7 * height, 7 * width:8 * width] = 1.0
        w9 = tf.constant(w9)

        self.filter = tf.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], axis=-1)

    def call(self, x, **kwargs):
        y = tf.tensordot(x, self.filter, axes=[[1, 2], [0, 1]])

        if self.normalization == 'minmax':
            maximum = tf.reduce_max(y)
            minimum = tf.reduce_min(y)
            y = (y - minimum)/(maximum - minimum)

        if self.activation == 'softmax':
            y = tf.nn.softmax(y)

        if self.avoid_zero:
            y = y + 1.0e-12

        return y


class D2NNMNISTFilter(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        super(D2NNMNISTFilter, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        self.input_dim = input_shape
        width = min(int(tf.floor(self.input_dim[2] / 9.0)), int(tf.floor(self.input_dim[1] / 7.0)))
        height = min(int(tf.floor(self.input_dim[2] / 9.0)), int(tf.floor(self.input_dim[1] / 7.0)))

        filter = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        filter[2 * height:3 * height, width:2 * width] = 1.0
        filter[2 * height:3 * height, 4 * width:5 * width] = 1.0
        filter[2 * height:3 * height, 7 * width:8 * width] = 1.0
        filter[4 * height:5 * height, 1 * width:2 * width] = 1.0
        filter[4 * height:5 * height, 3 * width:4 * width] = 1.0
        filter[4 * height:5 * height, 5 * width:6 * width] = 1.0
        filter[4 * height:5 * height, 7 * width:8 * width] = 1.0
        filter[6 * height:7 * height, width:2 * width] = 1.0
        filter[6 * height:7 * height, 4 * width:5 * width] = 1.0
        filter[6 * height:7 * height, 7 * width:8 * width] = 1.0

        self.filter = tf.constant(filter)

    def call(self, x, **kwargs):
        return tf.multiply(x, self.filter)


class CxD2NNFaradayRotation(tf.keras.layers.Layer):
    def __init__(self, output_dim, normalization=None, activation=None):
        super(CxD2NNFaradayRotation, self).__init__()
        self.output_dim = output_dim
        self.normalization = normalization
        self.activation = activation

    def call(self, x, **kwargs):
        rcp_x = tf.keras.layers.Lambda(lambda x:x[:,0,0,:,:])(x)
        rcp_y = tf.keras.layers.Lambda(lambda x:x[:,0,1,:,:])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x:x[:,1,0,:,:])(x)
        lcp_y = tf.keras.layers.Lambda(lambda x:x[:,1,1,:,:])(x)

        E0 = rcp_x + lcp_x
        I0 = tf.abs(E0)**2 / 2.0
        E90 = rcp_y + lcp_y
        I90 = tf.abs(E90)**2 / 2.0
        E45_x = (rcp_x + rcp_y + lcp_x + lcp_y) / 2.0
        E45_y = (rcp_x + rcp_y + lcp_x + lcp_y) / 2.0
        I45 = tf.abs(E45_x)**2/2+tf.abs(E45_y)**2 / 2.0
        E135_x = (rcp_x - rcp_y + lcp_x - lcp_y) / 2.0
        E135_y = (-rcp_x + rcp_y - lcp_x + lcp_y) / 2.0
        I135 = tf.abs(E135_x)**2/2 + tf.abs(E135_y)**2 / 2.0

        S1 = I0 - I90
        S2 = I45 - I135

        theta = tf.atan((S2 / S1)) / 2.0


        if self.normalization == 'minmax':
            minimum = tf.reduce_min(theta)
            maximum = tf.reduce_max(theta)
            theta = (theta - minimum) / (maximum - minimum)

        if self.activation == 'softmax':
            theta = tf.nn.softmax(theta)

        return theta


class Polarizer(tf.keras.layers.Layer):
    def __init__(self, output_dim, phi=0.0, trainable=False):
        super(Polarizer, self).__init__()
        self.output_dim = output_dim
        self.phi = self.add_weight("phi",shape=[1], initializer=tf.initializers.Constant(value=phi))
        self.trainable = trainable

    def call(self, x):
        rcp_x = tf.keras.layers.Lambda(lambda x:x[:,0,0,:,:])(x)
        rcp_y = tf.keras.layers.Lambda(lambda x:x[:,0,1,:,:])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x:x[:,1,0,:,:])(x)
        lcp_y = tf.keras.layers.Lambda(lambda x:x[:,1,1,:,:])(x)

        p00 = tf.complex(tf.cos(self.phi[0])**2, 0.0)
        p01 = tf.complex(tf.sin(2 * self.phi[0]) / 2, 0.0)
        p10 = p01
        p11 = tf.complex(tf.sin(self.phi[0])**2, 0.0)

        rcp_x_pol = p00 * rcp_x + p01 * rcp_y
        rcp_y_pol = p10 * rcp_x + p11 * rcp_y

        lcp_x_pol = p00 * lcp_x + p01 * lcp_y
        lcp_y_pol = p10 * lcp_x + p11 * lcp_y

        rcp = tf.stack([rcp_x_pol, rcp_y_pol], axis=1)
        lcp = tf.stack([lcp_x_pol, lcp_y_pol], axis=1)

        rl = tf.stack([rcp, lcp], axis=1)

        return rl


class Dielectric(tf.keras.layers.Layer):
    def __init__(self, tensor):
        super(Dielectric, self).__init__()
        self.tensor = tf.complex(tensor, 0.0 * tensor)

    def call(self, x):
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, 0, :, :])(x)
        rcp_y = tf.keras.layers.Lambda(lambda x: x[:, 0, 1, :, :])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, 0, :, :])(x)
        lcp_y = tf.keras.layers.Lambda(lambda x: x[:, 1, 1, :, :])(x)
        

class GGG(AngularSpectrum):
    def __init__(self, output_dim, wavelength, z=0.0, d=1.0e-6, normalization=None, method=None):
        super(GGG, self).__init__(output_dim, wavelength, z=z, d=d, n=2.0, normalization=normalization, method=method)
