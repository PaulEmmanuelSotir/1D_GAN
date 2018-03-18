# -*- coding: utf-8 -*-
import math
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.python.client import timeline

__all__ = ['tf_config', 'load_timeserie', 'warm_restart', 'add_summary_values',
           'xavier_init', 'leaky_relu', 'savitzky_golay', 'visualize_kernel', 'TensorflowProfiler']


def tf_config(allow_growth=True, **kwargs):
    config = tf.ConfigProto(**kwargs)
    config.gpu_options.allow_growth = allow_growth
    return config


def load_timeserie(path, window=-1, dtype=np.float32):
    # Load data
    timeserie = np.genfromtxt(path, delimiter=',', dtype=dtype)
    timeserie = timeserie[:, 1:]
    # Normalize data
    timeserie = preprocessing.StandardScaler().fit_transform(timeserie)
    # Delete zero pad data and slice data according to window size
    # TODO: make overlaping windows!
    if window > 0:
        n = (len(timeserie) // window) * window
        timeserie = timeserie[:n]
        timeserie = np.reshape(timeserie, (-1, window, timeserie.shape[-1]))
    return timeserie


def _cosine_annealing(x):
    return (np.cos(np.pi * x) + 1.) / 2.


def _log_cosine_annealing(x):
    log = np.log((np.exp(2) - np.exp(0)) * x + np.exp(0)) / 2.
    return (np.cos(np.pi * log) + 1.) / 2.


def warm_restart(epoch, t_0, max_lr, min_lr=1e-8, t_mult=2, annealing_fn=_log_cosine_annealing):
    """ Stochastic gradient descent with warm restarts of learning rate (see https://arxiv.org/pdf/1608.03983.pdf) """
    def _cycle_length(c): return t_0 * t_mult ** c
    cycle = int(np.floor(np.log(1 - epoch / t_0 * (1 - t_mult)) / np.log(t_mult)))
    cycle_begining = np.sum([_cycle_length(c) for c in range(0, cycle)]) if cycle > 0 else 0.
    x = (epoch - cycle_begining) / _cycle_length(cycle)
    lr = min_lr + (max_lr - min_lr) * annealing_fn(x)
    return lr, x == 0.


def add_summary_values(summary_writer, global_step=None, **values):
    if len(values) > 0:
        summary = tf.Summary()
        for name, value in values.items():
            summary.value.add(tag=name, simple_value=value)
        summary_writer.add_summary(summary, global_step=global_step)


def xavier_init(scale=None, mode='fan_avg'):
    """
    Xavier initialization
    """
    # TODO: make sure this is the correct scale for tanh (some sources say ~1.32, others 4., but 1. seems to give better results)
    return tf.variance_scaling_initializer(2. if scale == 'relu' else 1., mode=mode)


def leaky_relu(x, leak=0.2, name='leaky_relu'):
    with tf.variable_scope(name):
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
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


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


class TensorflowProfiler:
    # TODO: test it
    """ Inspired from stackoverflow answer: https://stackoverflow.com/a/37774470/5323273
    Example usage:
        ops = ...
        summary_writer = ...
        with tf.Session() as sess:
            with TensorflowProfiler(summary_writer, 'run_timeline.json') as profiler:
                summary, res = sess.run([summary_op, ops], **profiler.options_kwargs)
                summary_writer.write(summary)
        # Then you can open 'run_timeline.json' file in chrome from 'chrome://tracing' to visualize profiling or see profiling from Tensorboard graph
    """

    def __init__(self, summary_writer=None, timeline_file=None, global_step=None):
        self._timeline_file = timeline_file
        self._summary_writer = summary_writer
        self._step = global_step
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        self.options_kwargs = {'options': self.run_options, 'run_metadata': self.run_metadata}

    def __enter__(self):
        return self

    def __exit__(self, exp_type, exp_value, exp_traceback):
        if exp_type is None:
            if self._summary_writer is not None:
                self._summary_writer.add_run_metadata(self.run_metadata, global_step=self._step)
            if self._timeline_file is not None:
                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(self.run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(self._timeline_file, 'w') as file:
                    file.write(ctf)
