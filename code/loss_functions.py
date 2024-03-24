import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow import keras


@keras.utils.register_keras_serializable(package="own_loss_functions")
def gaussian_loss(y_true, y_pred):
    """
    Calculates the negative log-likelihood loss for a Gaussian distribution.

    Parameters:
    y_true (Tensor): The actual data points.
    y_pred (Tensor): Predictions of the parameters of the Gaussian distribution, assuming
                     y_pred is in the form [mu, sigma].

    Returns:
    Tensor: Negative log-likelihood loss.
    """
    mu_pred = y_pred[:, :3600]
    sigma_pred = y_pred[:, 3600:]
    sigma_loss = tf.math.softplus(sigma_pred)

    likelihood = -0.5 * tf.math.log(2 * tf.constant(np.pi, dtype=tf.float32) * tf.square(sigma_loss)) - \
                 tf.square(y_true - mu_pred) / (2 * tf.square(sigma_loss))
    return -tf.reduce_mean(likelihood)


@keras.utils.register_keras_serializable(package="own_loss_functions")
def gaussian_loss_15s(y_true, y_pred):
    """
    Calculates the negative log-likelihood loss for a Gaussian distribution.

    Parameters:
    y_true (Tensor): The actual data points.
    y_pred (Tensor): Predictions of the parameters of the Gaussian distribution, assuming
                     y_pred is in the form [mu, sigma].

    Returns:
    Tensor: Negative log-likelihood loss.
    """
    mu_pred = y_pred[:, :240]
    sigma_pred = y_pred[:, 240:]
    sigma_loss = tf.math.softplus(sigma_pred)

    likelihood = -0.5 * tf.math.log(2 * tf.constant(np.pi, dtype=tf.float32) * tf.square(sigma_loss)) - \
                 tf.square(y_true - mu_pred) / (2 * tf.square(sigma_loss))
    return -tf.reduce_mean(likelihood)


@keras.utils.register_keras_serializable(package="own_loss_functions")
def student_loss(y_true, y_pred):
    """
    Calculates the negative log-likelihood loss for a Student's T-distribution.

    Parameters:
    y_true (Tensor): The actual data points.
    y_pred (Tensor): Predictions of the parameters of the Student's T-distribution, assuming
                     y_pred is in the form [mu, sigma, v].

    Returns:
    Tensor: Negative log-likelihood loss.
    """
    mu, sigma, v = y_pred[:, :3600], y_pred[:, 3600:7200], y_pred[:, 7200:]
    sigma = tf.math.softplus(sigma)
    v = tf.math.softplus(v)

    z = tf.square((y_true - mu) / sigma)
    log_likelihood = tf.reduce_mean(
        tf.math.lgamma((v + 1) / 2) - tf.math.lgamma(v / 2) -
        0.5 * tf.math.log(v * tf.constant(np.pi, dtype=tf.float32)) -
        tf.math.log(sigma) -
        ((v + 1) / 2) * tf.math.log(1 + (z / v))
    )

    return -log_likelihood


@keras.utils.register_keras_serializable(package="own_loss_functions")
def cauchy_loss(y_true, y_pred):
    """
    Calculates the negative log-likelihood loss for a Cauchy distribution.

    Parameters:
    y_true (Tensor): The actual data points.
    y_pred (Tensor): Predictions of the parameters of the Cauchy distribution, assuming
                     y_pred is in the form [mu, gamma].

    Returns:
    Tensor: Negative log-likelihood loss.
    """
    mu_pred = y_pred[:, :3600]
    gamma_pred = y_pred[:, 3600:]
    gamma_loss = tf.math.softplus(gamma_pred)

    cauchy_loss = tf.math.log(tf.constant(np.pi, dtype=tf.float32)) + tf.math.log(gamma_loss) + \
                  tf.math.log(1 + tf.square((y_true - mu_pred) / gamma_loss))

    return tf.reduce_mean(cauchy_loss)


@keras.utils.register_keras_serializable(package="own_loss_functions")
def correlated_gaussian_loss(N, covariance_matrix):
    """
    Calculating the negative log-likelihood of the true labels under
    correlated gaussian process defined by the model predictions.

    Args:
        N (int): Number of Observations
        covariance_matrix (tf.Tensor): Pre-computed covariance matrix.

    Returns:
        Function: Calculated negative log likelihood loss.
    """

    def loss(y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        mean_pred = y_pred[:, :N]
        sigma = y_pred[:, N:]

       
        mean_pred = tf.nn.tanh(mean_pred)
        sigma = tf.nn.sigmoid(sigma)  
        sigma = tf.reshape(sigma, [-1, 1, 1])
        cov_matrix_scaled = sigma * covariance_matrix
        jitter = 1e-6  # Small constant
        cov_matrix_jittered = cov_matrix_scaled + jitter * tf.eye(N, batch_shape=[batch_size])

        # Defining the multivariate normal distribution with the adjusted covariance matrix
        mvn = tfp.distributions.MultivariateNormalFullCovariance(
            loc=mean_pred,
            covariance_matrix=cov_matrix_jittered
        )
        nll = -mvn.log_prob(y_true)

        return tf.reduce_mean(nll)

    return loss
