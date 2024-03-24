import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import multivariate_normal

tfk = tfp.math.psd_kernels


def prepare_covariance_matrix(N, kernel_type='RationalQuadratic'):
    """
    Calculating the covariance matrix using a specified kernel.

    Args:
        N (int): Number of observations/data points, also defines the shape of X.
        kernel_type (str, optional): Type of the kernel to be used for covariance matrix calculation. Defaults to 'RationalQuadratic'.
            Supported values: 'RationalQuadratic', 'ExponentiatedQuadratic', 'MaternOneHalf'

    Returns:
        covariance_matrix (tf.Tensor): Calculated covariance matrix based on X using the specified kernel.
    """
    X = tf.linspace(0.0, N - 1, N)
    X = tf.reshape(X, (N, 1))

    if kernel_type == 'RationalQuadratic':
        kernel = tfk.RationalQuadratic(amplitude=1., length_scale=1., scale_mixture_rate=1.)
    elif kernel_type == 'ExponentiatedQuadratic':
        kernel = tfk.ExponentiatedQuadratic(amplitude=1.0, length_scale=1.0)
    elif kernel_type == 'MaternOneHalf':
        kernel = tfk.MaternOneHalf(amplitude=1.0, length_scale=1.0)
    else:
        raise ValueError(
            "Unsupported kernel type. Please choose from 'RationalQuadratic', 'ExponentiatedQuadratic', "
            "or 'MaternOneHalf'.")

    covariance_matrix = kernel.matrix(X, X)
    return covariance_matrix


def sample_with_given_first_element(mu_empirical, Sigma_empirical, given_first_element):
    """
    Generates a sample from a multivariate normal distribution with a given first element.

    Parameters:
    mu_empirical (np.array): The mean vector of the distribution.
    Sigma_empirical (np.array): The covariance matrix of the distribution.
    given_first_element (float): The value of the first element of the generated sample.

    Returns:
    np.array: A sample from the conditional distribution with the given first element.
    """
    mu_1, mu_2 = mu_empirical[0], mu_empirical[1:]
    Sigma_11, Sigma_22 = Sigma_empirical[0, 0], Sigma_empirical[1:, 1:]
    Sigma_12, Sigma_21 = Sigma_empirical[0, 1:], Sigma_empirical[1:, 0]
    mu_conditional = mu_2 + np.dot(Sigma_21, (given_first_element - mu_1) / Sigma_11)
    Sigma_conditional = Sigma_22 - np.outer(Sigma_21, Sigma_12) / Sigma_11
    sample_conditional = multivariate_normal.rvs(mean=mu_conditional, cov=Sigma_conditional)
    sample_full = np.insert(sample_conditional, 0, given_first_element)

    return sample_full


def compute_conditional_stats(mu_empirical, Sigma_empirical, x_empirical):
    """
    Computes the mean and standard deviation of conditional distributions for each element
    (except the first) given the previous values, based on the empirical mean vector and
    empirical covariance matrix.

    Parameters:
    mu_empirical (np.array): The empirical mean vector.
    Sigma_empirical (np.array): The empirical covariance matrix.
    x_empirical (np.array): A realization of the first random variable.

    Returns:
    np.array: An array of dimension n x 2 containing means and standard deviations of the conditional distributions.
    """
    x_empirical = np.array(x_empirical).flatten()
    n = mu_empirical.size
    conditional_stats = np.zeros((n, 2))  
    conditional_stats[0, :] = [x_empirical[0], 0] 

    for i in range(1, n):
        mu_i = mu_empirical[i]
        Sigma_ii = Sigma_empirical[i, i]
        Sigma_i_past = Sigma_empirical[i, :i]
        Sigma_past_past = Sigma_empirical[:i, :i]
        conditional_mean = mu_i + Sigma_i_past @ np.linalg.solve(Sigma_past_past, x_empirical[:i] - mu_empirical[:i])
        conditional_variance = Sigma_ii - Sigma_i_past @ np.linalg.solve(Sigma_past_past, Sigma_i_past.T)
        conditional_stats[i, 0] = conditional_mean
        conditional_stats[i, 1] = np.sqrt(conditional_variance)

    return conditional_stats


def cyclic_coordinates(value, max_value):
    sine = np.sin(2 * np.pi * value / max_value)
    cosine = np.cos(2 * np.pi * value / max_value)
    return sine, cosine


def moving_average(data, window_size):
    extended_data = np.pad(data, (window_size // 2, window_size - 1 - window_size // 2), mode='edge')
    return np.convolve(extended_data, np.ones(window_size) / window_size, mode='valid')
