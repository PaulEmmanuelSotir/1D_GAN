# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from math import factorial

__all__ = ['tf_config', 'load_timeserie', 'leaky_relu', 'savitzky_golay', 'visualize_kernel']

def tf_config(allow_growth=True, **kwargs):
    config = tf.ConfigProto(**kwargs)
    config.gpu_options.allow_growth = allow_growth
    return config

def load_timeserie(path, window=-1, dtype=np.float32):
    # Load data
    timeserie = np.genfromtxt(path, delimiter=',', dtype=dtype)
    timeserie = timeserie[1:, 1:]
    # Normalize data
    timeserie = timeserie / timeserie.max()
    # Delete zero pad data and slice data according to window size
    if window > 0:
        n = (len(timeserie) // window) * window
        timeserie = timeserie[:n]
        timeserie = np.reshape(timeserie, (-1, window, timeserie.shape[-1]))
    return timeserie

def leaky_relu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def visualize_kernel(kernel, name):
    # Reshape tensor to image format [batch_size, height, width, channels]
    # TODO: do this in a cleanner way!
    if len(kernel.shape) == 4:
        image = tf.transpose(kernel, [2, 0, 3, 1])
        image = tf.reshape(image, [1, -1, image.shape[2].value * image.shape[3].value, 1])
    elif len(kernel.shape) == 3:
        image = tf.transpose(kernel, [1, 0, 2])
        image = tf.reshape(image, [1, -1, image.shape[2].value, 1])
    else:
        image = tf.reshape(kernel, [1, kernel.shape[0].value,  kernel.shape[1].value, 1])
    # this will display random 3 filters from the 64 in conv1
    tf.summary.image(name, image, max_outputs=3)
