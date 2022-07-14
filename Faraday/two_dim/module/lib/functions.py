import numpy as np
import math


def angular_spectrum(g0, wavelength, z, d, n=1.0):
    """
    light propagation using 4x expand and band-limited angular spectrum method.
    Args:
        g0 (2d array): input image
        wavelength (num): wavelength
        z (num): pixel size
        d (num): pixel size
        n (num): reflective index

    Returns:
        2d array: propagated image
    """
    wavelength_eff = wavelength / n
    pad_width = math.ceil(g0.shape[1] / 2)
    pad_height = math.ceil(g0.shape[0] / 2)
    padded = np.pad(g0, [[pad_height, pad_height], [pad_width, pad_width]])
    padded_width = padded.shape[1]
    padded_height = padded.shape[0]
    fft_image = np.fft.fft2(padded)
    u = np.fft.fftfreq(padded_width, d)
    v = np.fft.fftfreq(padded_height, d)
    UU, VV = np.meshgrid(u, v)
    w = np.where(UU ** 2 + VV ** 2 <= 1 / wavelength_eff ** 2, np.sqrt(1 / wavelength_eff ** 2 - UU ** 2 - VV ** 2), 0)
    h = np.exp(1.0j * 2.0 * np.pi * w * z)
    du = 1.0 / (padded_width * d)
    dv = 1.0 / (padded_height * d)
    u_lim = 1.0 / (wavelength_eff * np.sqrt((2.0 * du * z) ** 2 + 1.0))
    v_lim = 1.0 / (wavelength_eff * np.sqrt((2.0 * dv * z) ** 2 + 1.0))
    u_filter = np.where(np.abs(UU / (2 * u_lim)) < 1 / 2, 1, 0)
    v_filter = np.where(np.abs(VV / (2 * v_lim)) < 1 / 2, 1, 0)
    h_lim = h * u_filter * v_filter
    gz = np.fft.ifft2(fft_image * h_lim)
    g_crop = gz[pad_height:pad_height + g0.shape[0], pad_width:pad_width + g0.shape[1]]
    return g_crop


if __name__ == "__main__":
    pass
