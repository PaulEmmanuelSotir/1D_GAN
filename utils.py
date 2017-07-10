# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

__all__ = ['load_timeserie', 'leaky_relu']

def load_timeserie(path, window=-1):
    # Load data
    timeserie = np.genfromtxt(path, delimiter=',', dtype=np.float32)
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
