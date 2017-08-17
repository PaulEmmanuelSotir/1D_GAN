#!/usr/bin/python
# -*- coding: utf-8 -*-
"""1D GAN

Simple tensorflow implementation of 1D Generative Adversarial Network.

TODO:
    * fight mode dropping problem!
    * make a conditionnal version of 1D GAN
    * try preprocessing timeserie to train on DCT or wavelet channels
    * implement InfoGAN version of 1D GAN
    * allow completion of missing 1D data using similar technique as used in http://www.gitxiv.com/posts/7x3yumLjzfeMZwo6k/semantic-image-inpainting-with-perceptual-and-contextual (could be usefull for timeserie forecasting for example) (see also http://www.gitxiv.com/posts/3TNjqk2DBJHo35q9g/context-encoders-feature-learning-by-inpainting)
    * use dropout?
    * Figure out whether if the fact that generator does not generate data of the exact same dimension as real data due to convolution VALID padding is a problem
    * train on classification task (classes could be like UP, DOWN, STILL, RISKY_UP and GENTLE_DOWN)
    * compare results with ARMA models, markov-chains and real-valued recurrent conditionnal GAN (https://arxiv.org/pdf/1706.02633.pdf)
    * learn about ByteNet and if it could be used for timeseries

.. See https://github.com/PaulEmmanuelSotir/1D_GAN

"""
import os
import io
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 

import utils

__all__ = ['restore', 'generate', 'main']

# TODO: use layer normalization instead of batch normalization?

# Hyper parameters
params = {
    'activation_function': utils.leaky_relu,
    'lr': 1e-4,
    #'lambda': 10, #TODO: WGAN Gradient penalty hyperparameter
    'window': 366,
    'epoch_num': 4000,
    'batch_size': 64,
    'latent_dim': 30,
    'n_discriminator': 40, # TODO: remove or re-implement it
    'checkpoint_period': 2000,
    'progress_check_period': 1,
    'allow_gpu_mem_growth': True,
    'dtype': tf.float32 # TODO: take this parameter into account
}

capacity = 8 # TODO: remove it, default value = 16

def sample_z(batch_size, latent_dim):
    return np.float32(np.random.normal(size=[batch_size, latent_dim]))

def discriminator(x, activation_function, reuse=None):
    """Model function of 1D GAN discriminator"""
    init = lambda: tf.contrib.layers.xavier_initializer()
    # Convolutional layers
    conv1 = tf.layers.conv1d(inputs=x, filters=2*capacity, kernel_size=4, strides=2, activation=activation_function, kernel_initializer=init(), padding='valid', name='conv1', reuse=reuse)
    conv2 = tf.layers.conv1d(inputs=conv1, filters=4*capacity, kernel_size=4, strides=2, activation=activation_function, kernel_initializer=init(), padding='valid', name='conv2', reuse=reuse)
    conv3 = tf.layers.conv1d(inputs=conv2, filters=8*capacity, kernel_size=4, strides=2, activation=activation_function, kernel_initializer=init(), padding='valid', name='conv3', reuse=reuse)

    # Dense layer
    conv_flat = tf.reshape(conv3, shape=[conv3.shape[0].value, -1])
    dense = tf.layers.dense(inputs=conv_flat, units=1024, activation=activation_function, name='dense1', kernel_initializer=init(), reuse=reuse)

    # Last discrimination layer
    return tf.layers.dense(inputs=dense, units=1, name='dense2', reuse=reuse, kernel_initializer=init())

def generator(z, activation_function, window, num_channels, reuse=None):
    """Model function of 1D GAN generator"""
    init = lambda: tf.contrib.layers.xavier_initializer()
    # Find dense feature vector size according to generated window size and convolution strides (note that if you change convolution padding or the number of convolution layers, you will have to change this value too)
    stride = 2
    kernel_size = 4
    get_upconv_output_dim = lambda in_dim: (in_dim - kernel_size) // stride + 1 # Transposed convolution with VALID padding
    dense_window = get_upconv_output_dim(get_upconv_output_dim(get_upconv_output_dim(window))) # We find the dimension of output after 3 convolutions on 1D window

    # Fully connected layers
    dense1 = tf.layers.dense(inputs=z, units=1024, name='dense1', activation=activation_function, kernel_initializer=init(), reuse=reuse)
    dense1_bn = tf.layers.batch_normalization(dense1, name='dense1_bn', reuse=reuse)
    dense2 = tf.layers.dense(inputs=dense1_bn, units=dense_window*8*capacity, name='dense2', activation=activation_function, kernel_initializer=init(), reuse=reuse)
    dense2_bn = tf.layers.batch_normalization(dense2, name='dense2_bn', reuse=reuse)
    dense_features = tf.reshape(dense2_bn, shape=[-1, dense_window, 1, 8*capacity])
    
    # Deconvolution layers (We use tf.nn.conv2d_transpose as there is no implementation of conv1d_transpose in tensorflow for now)
    upconv1 = tf.layers.conv2d_transpose(inputs=dense_features, filters=4*capacity, kernel_size=(kernel_size, 1), strides=(stride, 1), padding='valid', name='upconv1', activation=activation_function, kernel_initializer=init(), reuse=reuse)
    upconv1_bn = tf.layers.batch_normalization(upconv1, name='upconv1_bn', reuse=reuse)
    upconv2 = tf.layers.conv2d_transpose(inputs=upconv1_bn, filters=2*capacity, kernel_size=(kernel_size, 1), strides=(stride, 1), padding='valid', name='upconv2', activation=activation_function, kernel_initializer=init(), reuse=reuse)
    upconv2_bn = tf.layers.batch_normalization(upconv2, name='upconv2_bn', reuse=reuse)
    upconv3 = tf.layers.conv2d_transpose(inputs=upconv2_bn, filters=num_channels, kernel_size=(kernel_size, 1), strides=(stride, 1), padding='valid', name='upconv3', activation=tf.nn.sigmoid, kernel_initializer=init(), reuse=reuse)
    return tf.squeeze(upconv3, axis=2, name='output')

def em_loss(y_coefficients, y_pred):
    """ Earth mover distance (wasserstein loss) """
    return tf.reduce_mean(tf.multiply(y_coefficients, y_pred))

def gan_losses(z, x, activation_function, window):
    with tf.variable_scope('generator'):
        g_sample = generator(z, activation_function, window, num_channels=x.shape[-1].value)
    # Get interpolates for gradient penalty (improved WGAN)
    epsilon = tf.random_uniform([], 0.0, 1.0)
    x_hat = epsilon * x + (1.0 - epsilon) * g_sample
    # Apply discriminator on real, fake and interpolated data
    with tf.variable_scope('discriminator'):
        d_real = discriminator(x, activation_function)
        d_fake = discriminator(g_sample, activation_function, reuse=True)
        d_hat = discriminator(x_hat, activation_function, reuse=True)
    # Process gradient penalty
    gradients = tf.gradients(d_hat, x_hat)
    gradient_penalty = 10.0 * tf.square(tf.norm(gradients[0], ord=2) - 1.0) # TODO: Use lambda hyperparameter here
    # Losses
    d_loss = em_loss(tf.ones(z.shape[0].value), d_fake) - em_loss(tf.ones(z.shape[0].value), d_real) + gradient_penalty
    g_loss = tf.reduce_mean(d_fake)
    # Log losses and gradients to summary
    tf.summary.histogram('gradients', gradients)
    tf.summary.scalar('gradient_penalty', gradient_penalty)
    tf.summary.scalar('generator_loss', g_loss)
    tf.summary.scalar('discriminator_loss', d_loss)
    return d_loss, g_loss

def gan_optimizers(d_loss, g_loss, lr):
    for v in tf.trainable_variables():
        if 'kernel' in v.name:
            utils.visualize_kernel(v, v.name)
        else:
            tf.summary.histogram(v.name, v)
    disc_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # Nescessary for batch normalization to update its mean and variance
    with tf.control_dependencies(extra_update_ops):
        d_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=disc_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=gen_vars)
    return d_optimizer, g_optimizer

def restore(sess, checkpoint_dir):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    saver = tf.train.import_meta_graph(latest + '.meta')
    saver.restore(sess, latest)

def generate(sess, count=1):
    graph = tf.get_default_graph()
    z = graph.get_tensor_by_name('input/z:0')
    gen = graph.get_tensor_by_name('generator/output:0')
    return sess.run(gen, feed_dict={z: sample_z(count, params['latent_dim'])})

def generate_curve_plots(sess):
    data = generate(sess, 64)
    # Plot first generated curves to byte buffer
    d = pd.DataFrame([t for t in data[0]], columns=['price', 'volume'])
    buffer = io.BytesIO()
    d.plot().get_figure().savefig(buffer, format='png', dpi=250)
    buffer.seek(0)
    return buffer.getvalue()

def summarize_generated_curves():
    # Decode generated curve plot byte buffer and save it to summary
    image = tf.placeholder(tf.string)
    decoded_im = tf.image.decode_png(image, channels=4)
    decoded_im = tf.expand_dims(decoded_im, 0)
    tf.summary.image('generated_curve', decoded_im)
    return image

def main(_=None):    
    train_dir = '/output/models/' if tf.flags.FLAGS.floyd_job else './models/'
    data_path = '/input/data.csv' if tf.flags.FLAGS.floyd_job else './data/data.csv'

    # Set log level to debug
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load time serie data
    timeserie = utils.load_timeserie(data_path, params['window'])

    with tf.name_scope('input'):
        # Latent input placeholder
        z = tf.placeholder(tf.float32, [params['batch_size'], params['latent_dim']], name='z')
        # Preloaded data input
        data_initializer = tf.placeholder(dtype=timeserie.dtype, shape=timeserie.shape, name='x')
        input_data = tf.Variable(data_initializer, trainable=False, collections=[])
        sample = tf.train.slice_input_producer([input_data], num_epochs=params['epoch_num'])
        samples = tf.train.batch(sample, batch_size=params['batch_size'])
        # TODO: temp, remove it
        with tf.name_scope('sample'):
            for s in sample:
                tf.summary.histogram(str(s), s)
        with tf.name_scope('samples'):
            tf.summary.histogram(str(samples), samples)

    # Create optimizers
    d_loss, g_loss = gan_losses(z, samples, params['activation_function'], params['window'])
    d_optimizer, g_optimizer = gan_optimizers(d_loss, g_loss, params['lr'])

    # Define generated curve plots summarization
    image = summarize_generated_curves()

    # Create model saver
    saver = tf.train.Saver()

    # Create variable initialization op
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session(config=utils.tf_config(params['allow_gpu_mem_growth'])) as sess:
        # TODO: restore previous training session if any with saver and tf.gfile.Exists(...)

        # Intialize variables
        sess.run(init_op)
        sess.run(input_data.initializer, feed_dict={data_initializer: timeserie})

        # Create summary utils
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Train GAN model
        try:
            step = 1
            while not coord.should_stop():
                latent = lambda: sample_z(params['batch_size'], params['latent_dim'])
                # Train discriminator
                for _ in range(params['n_discriminator']):
                    _ = sess.run(d_optimizer, feed_dict={z: latent()})
                # Train generator
                _ = sess.run(g_optimizer, feed_dict={z: latent()})
                # Show progress and append results to summary
                if step % params['progress_check_period'] == 0:
                    # Plot samples curves to tensorboard summary
                    im = generate_curve_plots(sess)
                    summary, d_loss_curr, g_loss_curr = sess.run([summary_op, d_loss, g_loss], feed_dict={z: latent(), image: im})
                    summary_writer.add_summary(summary, step)
                    print('STEP=%d\tdisc_loss=%.4f\tgen_loss=%.4f' % (step, d_loss_curr, g_loss_curr))
                # Save a checkpoint periodically
                if step % params['checkpoint_period'] == 0:
                    print('Saving checkpoint...')
                    saver.save(sess, os.path.join(train_dir, 'gan1d'), global_step=step)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Saving...')
            saver.save(sess, os.path.join(train_dir, 'gan1d'), global_step=step)
        finally:
            coord.request_stop()
        # Wait for threads to finish
        coord.join(threads)

if __name__ == '__main__':
    tf.flags.DEFINE_bool('floyd-job', False, 'Change working directories for training on Floyd.')
    tf.app.run()
