# -*- coding: utf-8 -*-
"""1D GAN

Simple tensorflow implementation of 1D Generative Adversarial Network.

TODO:
    * make a conditionnal version of 1D GAN
    * try preprocessing timeserie to train on DCT or wavelet channels
    * implement InfoGAN version of 1D GAN
    * allow completion of missing 1D data using similar technique as used in http://www.gitxiv.com/posts/7x3yumLjzfeMZwo6k/semantic-image-inpainting-with-perceptual-and-contextual (could be usefull for timeserie forecasting for example) (see also http://www.gitxiv.com/posts/3TNjqk2DBJHo35q9g/context-encoders-feature-learning-by-inpainting)
    * use dropout?

.. See https://github.com/PaulEmmanuelSotir/1D_GAN

"""
import time
import numpy as np
import tensorflow as tf

import utils

from tensorflow.python.ops import array_ops

__all__ = ['restore', 'generate', 'main']

# Hyper parameters
params = {
    'activation_function': utils.leaky_relu,
    'lr': 0.01,
    'epoch_num': 300,
    'batch_size': 32,
    'latent_dim': 30,
    'window': 8 * 48, # 8 = 2*2*2 = 3 convolution layers of stride 2
    'checkpoint_period': 1000,
    'train_dir': 'train/',
    'data_path': 'data/data.csv',
    'allow_gpu_mem_growth': True,
    'dtype': tf.float32 # TODO: take this parameter into account
}

def sample_z(batch_size, latent_dim):
    return np.float32(np.random.normal(size=[batch_size, latent_dim]))

def discriminator(x, activation_function, reuse=None):
    """Model function of 1D GAN discriminator"""
    # TODO: use batch normalization for discriminator?

    # Convolutional layers
    conv1 = tf.layers.conv1d(inputs=x, filters=32, kernel_size=4, strides=2, activation=activation_function, padding='same', name='conv1', reuse=reuse)
    conv2 = tf.layers.conv1d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=activation_function, padding='same', name='conv2', reuse=reuse)
    conv3 = tf.layers.conv1d(inputs=conv2, filters=128, kernel_size=4, strides=2, activation=activation_function, padding='same', name='conv3', reuse=reuse)

    # Dense layer
    conv_flat = tf.reshape(conv3, shape=[conv3.shape[0].value, -1])
    dense = tf.layers.dense(inputs=conv_flat, units=1024, activation=activation_function, name='dense1', reuse=reuse)

    # Last discrimination layer
    logits = tf.layers.dense(inputs=dense, units=1, name='dense2', reuse=reuse)
    output = tf.nn.sigmoid(logits, name='output')
    return output, logits

def generator(z, activation_function, window, num_channels, reuse=None):
    """Model function of 1D GAN generator"""
    # Find dense feature vector size according to generated window size and convolution strides (note that if you change convolution padding or the number of convolution layers, you will have to change this value too)
    stride = 2
    dense_window = window // (stride*stride*stride)

    # Fully connected layers
    bn_z = tf.layers.batch_normalization(inputs=z, name='batch_norm1') # TODO: think 2sec and figure out whether if BN is usefull on z
    dense1 = tf.layers.dense(inputs=bn_z, units=1024, name='dense1', activation=activation_function, reuse=reuse)
    bn_dense1 = tf.layers.batch_normalization(inputs=dense1, name='batch_norm2')
    dense2 = tf.layers.dense(inputs=bn_dense1, units=dense_window*128, name='dense2', activation=activation_function, reuse=reuse)
    bn_dense2 = tf.layers.batch_normalization(inputs=dense2, name='batch_norm3')
    dense_features = tf.reshape(bn_dense2, shape=[bn_dense2.shape[0].value, -1, 1, 128])

    # Deconvolution layers (We use tf.nn.conv2d_transpose as there is no implementation of conv1d_transpose in tensorflow for now)
    upconv1 = tf.layers.conv2d_transpose(inputs=dense_features, filters=64, kernel_size=(4, 1), strides=(stride, 1), padding='same', name='upconv1', activation=activation_function, reuse=reuse)
    bn_upconv1 = tf.layers.batch_normalization(inputs=upconv1, name='batch_norm4')
    upconv2 = tf.layers.conv2d_transpose(inputs=bn_upconv1, filters=32, kernel_size=(4, 1), strides=(stride, 1), padding='same', name='upconv2', activation=activation_function, reuse=reuse)
    upconv3 = tf.layers.conv2d_transpose(inputs=upconv2, filters=num_channels, kernel_size=(4, 1), strides=(stride, 1), padding='same', name='upconv3', activation=tf.nn.sigmoid, reuse=reuse)
    return tf.squeeze(upconv3, axis=2, name='output')

def gan_loss(z, x, activation_function, window):
    with tf.variable_scope('generator'):
        g_sample = generator(z, activation_function, window, num_channels=x.shape[-1].value)
    with tf.variable_scope('discriminator'):
        d_real, d_logit_real = discriminator(x, activation_function)
        d_fake, d_logit_fake = discriminator(g_sample, activation_function, reuse=True)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
    # Log losses to summary
    tf.summary.scalar('generator_loss', g_loss)
    tf.summary.scalar('discriminator_fake_loss', d_loss_fake)
    tf.summary.scalar('discriminator_real_loss', d_loss_real)
    tf.summary.scalar('discriminator_loss', d_loss_fake + d_loss_real)
    return d_loss_fake + d_loss_real, g_loss

def gan_optimizers(d_loss, g_loss, lr):
    disc_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(d_loss, var_list=disc_vars)
    g_optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(g_loss, var_list=gen_vars)
    return d_optimizer, g_optimizer

def train_step(sess, z, batch_size, latent_dim, d_opt, g_opt, d_loss, g_loss, summary_op):
    """Executes a GAN training step. Note that two input batches are used in one step"""
    # Train discriminator
    latent = sample_z(batch_size, latent_dim)
    _ = sess.run(d_opt, feed_dict={z: latent})
    # Train generator
    latent = sample_z(batch_size, latent_dim)
    _ = sess.run(g_opt, feed_dict={z: latent})
    # Return losses and summary
    return sess.run([summary_op, d_loss, g_loss], feed_dict={z: sample_z(batch_size, latent_dim)})

def restore(sess, checkpoint_dir=params['train_dir']):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    saver = tf.train.import_meta_graph(latest + '.meta')
    saver.restore(sess, latest)

def generate(sess, count=1):
    graph = tf.get_default_graph()
    z = graph.get_tensor_by_name('input/z:0')
    gen = graph.get_tensor_by_name('generator/output:0')
    result = sess.run([gen], feed_dict={z: sample_z(count, params['latent_dim'])})
    return result

def main(_=None):
    # Set log level to debug
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load timeserie data
    timeserie = utils.load_timeserie(params['data_path'], params['window'])

    with tf.name_scope('input'):
        # Latent input placeholder
        z = tf.placeholder(tf.float32, [params['batch_size'], params['latent_dim']], name='z')
        # Preloaded data input
        data_initializer = tf.placeholder(dtype=timeserie.dtype, shape=timeserie.shape, name='x')
        input_data = tf.Variable(data_initializer, trainable=False, collections=[])
        sample = tf.train.slice_input_producer([input_data], num_epochs=params['epoch_num'])
        samples = tf.train.batch(sample, batch_size=params['batch_size'])

    # Create optimizers
    d_loss, g_loss = gan_loss(z, samples, params['activation_function'], params['window'])
    d_optimizer, g_optimizer = gan_optimizers(d_loss, g_loss, params['lr'])

    # Create model saver
    saver = tf.train.Saver()

    # Create the op for initializing variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = params['allow_gpu_mem_growth']
    with tf.Session(config=config) as sess:
        # TODO: restore previous training session if any with saver and tf.gfile.Exists(...)

        # Intialize variables
        sess.run(init_op)
        sess.run(input_data.initializer, feed_dict={data_initializer: timeserie})

        # Create summary utils
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(params['train_dir'], sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Train GAN model
        try:
            step = 1 # TODO: use global_step = tf.train.get_or_create_global_step() instead?
            while not coord.should_stop():
                t0 = time.time()
                summary, d_loss_curr, g_loss_curr = train_step(sess, z, params['batch_size'], params['latent_dim'], d_optimizer, g_optimizer, d_loss, g_loss, summary_op)
                summary_writer.add_summary(summary, step)
                t = time.time() - t0
                print('STEP=%d\tDt=%.2f\tdisc_loss=%.4f\tgen_loss=%.4f' % (step, t, d_loss_curr, g_loss_curr))
                # Save a checkpoint periodically
                if step % params['checkpoint_period'] == 0:
                    print('Saving checkpoint...')
                    saver.save(sess, params['train_dir'] + 'gan1d', global_step=step)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Saving...')
            saver.save(sess, params['train_dir'] + 'gan1d', global_step=step)
        finally:
            coord.request_stop()
        # Wait for threads to finish
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
