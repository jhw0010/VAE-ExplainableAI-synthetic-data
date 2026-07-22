import numpy as np
import tensorflow as tf
from typing import Tuple

def log_normal_pdf(sample: tf.Tensor, mean: tf.Tensor, logvar: tf.Tensor, raxis: int = 1) -> tf.Tensor:
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis,
    )

def mean_squared_error(logit: tf.Tensor, ground_truth: tf.Tensor) -> tf.Tensor:
    error = ground_truth - logit
    return tf.reduce_mean(tf.square(error), axis=1)

def compute_loss(model: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    x_reshaped = tf.reshape(x, tf.shape(x_logit))

    mse = mean_squared_error(logit=x_logit, ground_truth=x_reshaped)
    logpx_z = -tf.reduce_sum(mse, axis=0)

    logpz = log_normal_pdf(z, tf.zeros_like(z), tf.zeros_like(z))
    logqz_x = log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model: tf.keras.Model, x: tf.Tensor) -> Tuple[list, tf.Tensor]:
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients, loss
