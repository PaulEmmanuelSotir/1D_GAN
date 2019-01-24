#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 1D GAN
Tensorflow implementation of 1D Generative Adversarial Network (improved training of WGAN variant).

TODO:
    * make a conditionnal version of 1D GAN
    * try preprocessing timeserie to train on DCT or wavelet channels
    * implement InfoGAN version of 1D GAN
    * allow completion of missing 1D data using similar technique as used in http://www.gitxiv.com/posts/7x3yumLjzfeMZwo6k/semantic-image-inpainting-with-perceptual-and-contextual (could be usefull for timeserie forecasting for example) (see also http://www.gitxiv.com/posts/3TNjqk2DBJHo35q9g/context-encoders-feature-learning-by-inpainting)
    * use dropout?
    * try to use layer normalization instead of batch normalization?
    * train on classification task (classes could be like UP, DOWN, STILL, RISKY_UP and GENTLE_DOWN)
    * compare results with ARMA models, markov-chains and real-valued recurrent conditionnal GAN (https://arxiv.org/pdf/1706.02633.pdf)
    * learn about ByteNet and if it could be used for timeseries

.. See https://github.com/PaulEmmanuelSotir/1D_GAN

"""
import os
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

__all__ = ['restore', 'generate', 'main']


# Hyper parameters
hp = {'activation_fn': utils.leaky_relu,  # Don't forget to change Xavier initialization accordingly when changing activation function
      'lr': 1e-4,
      #   'warm_resart_lr': {
      #       'initial_cycle_length': 20,
      #       'lr_cycle_growth': 1.5,
      #       'minimal_lr': 5e-8},
      'window': 366,
      'epochs': 900,
      'latent_dim': 32,
      'batch_size': 16,
      'n_generator': 1,
      'n_discriminator': 1,
      'grad_penalty_lambda': 10.}

ALLOW_GPU_MEM_GROWTH = True
CHECKPOINT_PERIOD = 20
EVALUATE_PERIOD = 1
CAPACITY = 16  # TODO: remove it, default value = 16


def sample_z(batch_size, latent_dim):
    return np.float32(np.random.normal(size=[batch_size, latent_dim]))


def discriminator(x, activation_fn, reuse=None, scope=None):
    """ Model function of 1D GAN discriminator """
    # Convolutional layers
    conv = tf.layers.conv1d(inputs=x, filters=2 * CAPACITY, kernel_size=4, strides=2, activation=activation_fn,
                            kernel_initializer=utils.xavier_init('relu'), padding='valid', name='conv_1', reuse=reuse)
    conv = tf.layers.conv1d(inputs=conv, filters=4 * CAPACITY, kernel_size=4, strides=2, activation=activation_fn,
                            kernel_initializer=utils.xavier_init('relu'), padding='valid', name='conv_2', reuse=reuse)
    conv = tf.layers.conv1d(inputs=conv, filters=8 * CAPACITY, kernel_size=4, strides=2, activation=activation_fn,
                            kernel_initializer=utils.xavier_init('relu'), padding='valid', name='conv_3', reuse=reuse)
    conv = tf.layers.conv1d(inputs=conv, filters=8 * CAPACITY, kernel_size=4, strides=2, activation=activation_fn,
                            kernel_initializer=utils.xavier_init('relu'), padding='valid', name='conv_4', reuse=reuse)
    conv = tf.reshape(conv, shape=[-1, np.prod([dim.value for dim in conv.shape[1:]])])

    # Dense layers
    dense = tf.layers.dense(inputs=conv, units=1024, activation=activation_fn, name='dense_1', kernel_initializer=utils.xavier_init(), reuse=reuse)
    return tf.layers.dense(inputs=dense, units=1, activation=tf.nn.sigmoid, name='dense_2', reuse=reuse, kernel_initializer=utils.xavier_init())


def generator(z, activation_fn, window, num_channels, training=False, reuse=None):
    """ Model function of 1D GAN generator """
    # Find dense feature vector size according to generated window size and convolution strides (note that if you change convolution padding or the number of convolution layers, you will have to change this value too)
    stride = 2
    kernel_size = 4

    # We find the dimension of output after 4 convolutions on 1D window
    def get_upconv_output_dim(in_dim): return (in_dim - kernel_size) // stride + 1  # Transposed convolution with VALID padding
    dense_window_size = get_upconv_output_dim(get_upconv_output_dim(get_upconv_output_dim(get_upconv_output_dim(window))))

    # Fully connected layers
    dense = tf.layers.dense(inputs=z, units=1024, name='dense1', kernel_initializer=utils.xavier_init('relu'), activation=activation_fn, reuse=reuse)

    dense = tf.layers.dense(inputs=dense, units=dense_window_size * 8 * CAPACITY, name='dense2', kernel_initializer=utils.xavier_init('relu'), reuse=reuse)
    dense = tf.layers.batch_normalization(dense, name='dense2_bn', training=training, reuse=reuse)
    dense = activation_fn(dense)

    dense = tf.reshape(dense, shape=[-1, dense_window_size, 1, 8 * CAPACITY])

    # Deconvolution layers (We use tf.nn.conv2d_transpose as there is no implementation of conv1d_transpose in tensorflow for now)
    upconv = tf.layers.conv2d_transpose(inputs=dense, filters=8 * CAPACITY, kernel_size=(kernel_size, 1), strides=(stride, 1),
                                        padding='valid', name='upconv0', kernel_initializer=utils.xavier_init('relu'), reuse=reuse)
    upconv = tf.layers.batch_normalization(upconv, name='upconv0_bn', training=training, reuse=reuse)
    upconv = activation_fn(upconv)

    upconv = tf.layers.conv2d_transpose(inputs=upconv, filters=4 * CAPACITY, kernel_size=(kernel_size, 1), strides=(stride, 1),
                                        padding='valid', name='upconv1', kernel_initializer=utils.xavier_init('relu'), reuse=reuse)
    upconv = tf.layers.batch_normalization(upconv, name='upconv1_bn', training=training, reuse=reuse)
    upconv = activation_fn(upconv)

    upconv = tf.layers.conv2d_transpose(inputs=upconv, filters=2 * CAPACITY, kernel_size=(kernel_size, 1), strides=(stride, 1),
                                        padding='valid', name='upconv2', kernel_initializer=utils.xavier_init('relu'), reuse=reuse)
    upconv = tf.layers.batch_normalization(upconv, name='upconv2_bn', training=training, reuse=reuse)
    upconv = activation_fn(upconv)

    upconv = tf.layers.conv2d_transpose(inputs=upconv, filters=num_channels, kernel_size=(kernel_size, 1), strides=(stride, 1),
                                        padding='valid', name='upconv3', kernel_initializer=utils.xavier_init(), reuse=reuse)
    upconv = tf.layers.batch_normalization(upconv, name='upconv3_bn', training=training, reuse=reuse)
    return tf.squeeze(upconv, axis=2, name='output')


def gan_losses(z, x, activation_fn, window, grad_penalty_lambda, gan_training):
    with tf.variable_scope('generator'):
        g_sample = generator(z, activation_fn, window, num_channels=x.shape[-1].value, training=gan_training)
    if grad_penalty_lambda is not None:
        # Get interpolates for gradient penalty (improved WGAN)
        with tf.variable_scope('gradient_penalty'):
            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * x + (1.0 - epsilon) * g_sample
    # Apply discriminator on real, fake and interpolated data
    with tf.variable_scope('discriminator'):
        d_real = discriminator(x, activation_fn)
        d_fake = discriminator(g_sample, activation_fn, reuse=True)
        if grad_penalty_lambda is not None:
            d_hat = discriminator(x_hat, activation_fn, reuse=True)
    # Process gradient penalty
    gradient_penalty = 0.
    if grad_penalty_lambda is not None:
        with tf.variable_scope('gradient_penalty'):
            gradients = tf.gradients(d_hat, x_hat)[0]
            assert len(gradients.shape) == 3, 'Bad gradient rank'
            flat_grad_dim = np.prod([dim.value for dim in gradients.shape[1:]])
            gradient_penalty = grad_penalty_lambda * tf.reduce_mean(tf.square(tf.norm(tf.reshape(gradients, shape=[-1, flat_grad_dim]), ord=2) - 1.0))
    # Losses
    with tf.variable_scope('loss'):
        g_loss = tf.reduce_mean(d_fake)
        d_loss = tf.reduce_mean(d_real) - g_loss + gradient_penalty
    return d_loss, g_loss


def gan_optimizers(d_loss, g_loss, lr):
    # TODO: uncomment this ad put summarization back
    """for v in tf.trainable_variables():
        if 'kernel' in v.name:
            utils.visualize_kernel(v, v.name)
        else:
            tf.summary.histogram(v.name, v)"""
    disc_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Nescessary for batch normalization to update its mean and variance
    with tf.control_dependencies(extra_update_ops):
        d_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=disc_vars, name='disc_opt')
        g_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=gen_vars, name='gen_opt')
    return tf.tuple([d_loss], control_inputs=[d_optimizer]), tf.tuple([g_loss], control_inputs=[g_optimizer])


def restore(sess, checkpoint_dir):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    saver = tf.train.import_meta_graph(latest + '.meta')
    saver.restore(sess, latest)


def generate(sess, count=1):
    graph = tf.get_default_graph()
    z = graph.get_tensor_by_name('input/z:0')
    gen = graph.get_tensor_by_name('generator/output:0')
    return sess.run(gen, feed_dict={z: sample_z(count, hp['latent_dim'])})


def generate_curve_plots(sess):
    data = generate(sess)
    # Plot first generated curves to byte buffer
    buffer = io.BytesIO()
    fig = pd.DataFrame(data[0]).plot().get_figure()
    fig.savefig(buffer, format='png', dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def summarize_generated_curves():
    # Decode generated curve plot byte buffer and save it to summary
    with tf.variable_scope('curve_summarization'):
        image = tf.placeholder(tf.string, [], name='curve')
        decoded_im = tf.image.decode_png(image, channels=4)
        decoded_im = tf.expand_dims(decoded_im, 0)
        tf.summary.image('generated_curve', decoded_im)
    return image


def train(dataset, hp, sample_shape, train_dir):
    wr_hp = hp.get('warm_resart_lr')

    # Create input placeholders
    with tf.variable_scope('input'):
        gan_training = tf.placeholder_with_default(False, [], name='training')
        z = tf.placeholder(tf.float32, [None, hp['latent_dim']], name='z')
        X = tf.placeholder(tf.float32, [None, *sample_shape], name='X')
        lr = tf.placeholder(tf.float32, [], name='learning_rate')
        # Summarization of generated sample (plot image)
        image = summarize_generated_curves()

    # Create optimizers
    d_loss, g_loss = gan_losses(z, X, hp['activation_fn'], hp['window'], hp['grad_penalty_lambda'], gan_training)
    d_optimizer, g_optimizer = gan_optimizers(d_loss, g_loss, lr)

    # Create model saver
    saver = tf.train.Saver()

    # Create variable initialization op
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session(config=utils.tf_config(ALLOW_GPU_MEM_GROWTH)) as sess:
        # TODO: restore previous training session if any with saver and tf.gfile.Exists(...)

        # Intialize variables
        sess.run(init_op)

        # Create summary utils
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        # Train GAN model
        learning_rate = hp['lr']
        batch_per_epoch = int(np.ceil(len(dataset) / hp['batch_size']))
        for epoch in range(hp['epochs']):
            # Shuffle dataset
            dataset = dataset[np.random.permutation(len(dataset))]
            # Train GAN on minibatches
            mean_d_loss, mean_g_loss = 0., 0.
            for step, range_min in zip(range(batch_per_epoch), range(0, len(dataset) - 1, hp['batch_size'])):
                range_max = min(range_min + hp['batch_size'], len(dataset))
                # Train discriminator
                latent = partial(sample_z, range_max - range_min, hp['latent_dim'])
                d_loss, = sess.run(d_optimizer, feed_dict={z: latent(), lr: learning_rate, X: dataset[range_min:range_max]})
                mean_d_loss += (range_max - range_min) * d_loss / len(dataset)
                # Train generator
                if step % hp['n_discriminator'] == 0:  # TODO: make it more accurate
                    for _ in range(hp['n_generator']):
                        g_loss, = sess.run(g_optimizer, feed_dict={z: latent(), lr: learning_rate, gan_training: True})
                        mean_g_loss += (range_max - range_min) * g_loss / (len(dataset) *
                                                                           hp['n_generator'] / hp['n_discriminator'])  # TODO: make it more accurate
                if wr_hp is not None:
                    learning_rate, _ = utils.warm_restart(epoch + step / batch_per_epoch, t_0=wr_hp['initial_cycle_length'],
                                                          max_lr=hp['lr'], min_lr=wr_hp['minimal_lr'], t_mult=wr_hp['lr_cycle_growth'])
            # Show progress and append results to summary
            utils.add_summary_values(summary_writer, global_step=epoch, g_loss=mean_g_loss, d_loss=mean_d_loss, lr=learning_rate)
            summary = sess.run(summary_op, feed_dict={z: sample_z(1, hp['latent_dim']), image: generate_curve_plots(sess)})
            summary_writer.add_summary(summary, epoch)
            print('EPOCH=%d\t G_LOSS=%f\t D_LOSS=%f\t' % (epoch, mean_g_loss, mean_d_loss))
            # Save a checkpoint periodically
            if epoch % CHECKPOINT_PERIOD == 0:
                print('Saving checkpoint...')
                saver.save(sess, os.path.join(train_dir, 'gan1d'), global_step=epoch)
        print('Training done, saving...')
        saver.save(sess, os.path.join(train_dir, 'gan1d'), global_step=epoch)


def main(_=None):
    train_dir = '/output/models/sinus_test5/' if tf.flags.FLAGS.floyd_job else './models/sinus_test5/'
    data_path = '/input/data.csv' if tf.flags.FLAGS.floyd_job else './data/data.csv'

    # Set log level to debug
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load time serie data
    timeserie = utils.load_timeserie(data_path, hp['window'])
    sample_shape = timeserie.shape[1:]

    # Train 1D GAN
    train(timeserie, hp, sample_shape, train_dir)


if __name__ == '__main__':
    tf.flags.DEFINE_bool('floyd-job', False, 'Change working directories for training on Floyd.')
    tf.app.run()
