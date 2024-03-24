import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist
from scipy.stats import norm


def calculate_negative_log_likelihood(mus, sigmas, samples):
    """
    Calculates the negative log likelihood for given means (mus),
    standard deviations (sigmas), and samples.

    Parameters:
    - mus: matrix of means.
    - sigmas: matrix of standard deviations.
    - samples:  matrix of samples.
    Returns:
    - A numpy array containing the NLL values for each row.
    """

    nll = 0.5 * np.log(2 * np.pi * sigmas ** 2) + ((samples - mus) ** 2) / (2 * sigmas ** 2)
    nll_sum_per_row = np.sum(nll, axis=1)

    return nll_sum_per_row


def crps_norm(mus, sigmas, xs):
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for arrays of value of a univariate normal distribution.

    :param mus: Array of mean values of the normal distributions
    :param sigmas: Array of standard deviations of the normal distributions
    :param xs:  Array of observed values
    :return: Array of CRPS values
    """
    zs = (xs - mus) / sigmas
    crps_values = sigmas * (zs * (2 * norm.cdf(zs) - 1) + 2 * norm.pdf(zs) - 1 / np.sqrt(np.pi))
    return crps_values


def negative_log_likelihood_gaussian(y_true, y_pred, num):
    """
    Calculates the negative log-likelihood loss for a Gaussian distribution.

    Parameters:
    y_true (Tensor): The actual data points.
    y_pred (Tensor): Predictions of the parameters of the Gaussian distribution, assuming
                     y_pred is in the form [mu, sigma].
    num: number of observations

    Returns:
    Tensor: Negative log-likelihood loss.
    """
    mu_pred = y_pred[:, :num]
    sigma_pred = y_pred[:, num:]
    sigma_loss = tf.math.softplus(sigma_pred)

    likelihood = -0.5 * tf.math.log(2 * tf.constant(np.pi, dtype=tf.float32) * tf.square(sigma_loss)) - \
                 tf.square(y_true - mu_pred) / (2 * tf.square(sigma_loss))
    return -tf.reduce_mean(likelihood)


def histogram_of_realized_quantiles(y_true, mus, sigmas):
    """
    Creates a histogram of the realized quantile levels.

    Parameters:
    y_true (array): The true values.
    mus (array): The estimated means.
    sigmas (array): The estimated standard deviations.
    """
    quantiles = norm.cdf(y_true, loc=mus, scale=sigmas)
    plt.figure(figsize=(8, 6))
    plt.hist(quantiles, bins=50, density=True, alpha=0.7, color='skyblue')
    plt.title('Histogram of Realized Quantile Levels')
    plt.xlabel('Quantile Level')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()


def calibration_plot(y_true, mus, sigmas):
    """
    Creates a calibration plot.

    Parameters:
    y_true (array): The true values.
    mus (array): The estimated means.
    sigmas (array): The estimated standard deviations.
    """
    quantiles = norm.cdf(y_true, loc=mus, scale=sigmas)
    sorted_quantiles = np.sort(quantiles)
    cumulative_proportions = np.linspace(1 / len(sorted_quantiles), 1, len(sorted_quantiles))

    plt.figure(figsize=(8, 8))
    plt.plot(sorted_quantiles, cumulative_proportions, label='Realized Quantile Levels', color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.title('Calibration Plot')
    plt.xlabel('Theoretical Quantile Levels')
    plt.ylabel('Cumulative Proportion of Realized Quantile Levels')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


def calculate_negative_log_likelihood_multi_gaussian(y_true, mus, sigmas):
    """
    Calculates the negative log-likelihood for each observation.

    :param y_true: Array of actual values
    :param mus: Array of means
    :param sigmas: Array of covariance matrix
    :return: Array of negative log likelihood value
    """
    from scipy.stats import multivariate_normal
    nll_values = np.zeros(y_true.shape[0])

    for i in range(y_true.shape[0]):
        density = multivariate_normal.pdf(y_true[i], mean=mus[i], cov=sigmas[i])
        nll_values[i] = -np.log(density)

    return nll_values


def energy_score(y, mu_x, Cov_X, m=1000):
    """
    Calculate the energy score (ES) for a given point y, where X is sampled from a multivariate Gaussian distribution.


    Parameters:
    y (np.ndarray): a d-dimensional vector.
    mu_x (np.ndarray): the mean vector of the Gaussian distribution.
    Cov_X (np.ndarray): the covariance matrix of the Gaussian distribution.
    m (int): number of samples to draw from the Gaussian distribution.

    Returns:
    float: the energy score.
    """

    X = np.random.multivariate_normal(mu_x, Cov_X, m)
    term1 = np.mean(np.linalg.norm(X - y, axis=1))
    pairwise_distances = pdist(X, 'euclidean')
    term2 = np.mean(pairwise_distances)

    # Calculate the Energy Score
    ES = term1 - (1 / (2 * m)) * term2
    return ES


def energy_scores_for_multiple_ys_and_gaussians(ys, mu_xs, cov_xs, m):
    """
    Calculate the energy scores (ES) for multiple points y and corresponding Gaussian distributions.

    Parameters:
    ys (np.ndarray): an array of shape (n, d) where each row is a d-dimensional point y.
    mu_xs (np.ndarray): an array of shape (n, d) where each row is the mean vector of a Gaussian distribution.
    cov_xs (np.ndarray): a 3D array of shape (n, d, d) where each matrix along the first axis is the covariance matrix of a Gaussian distribution.
    m (int): number of samples to draw from each Gaussian distribution. 

    Returns:
    np.ndarray: an array of energy scores of length n.
    """
    n = ys.shape[0]  # number of data points
    energy_scores = np.zeros(n)

    for i in range(n):
        y = ys[i]
        mu_x = mu_xs[i]
        cov_x = cov_xs[i]

        energy_scores[i] = energy_score(y, mu_x, cov_x, m)

    return energy_scores

def save_histogram_of_realized_quantiles(y_true, mus, sigmas, filename):
    quantiles = norm.cdf(y_true, loc=mus, scale=sigmas)
    plt.figure(figsize=(8, 6))
    plt.hist(quantiles, bins=50, density=True, alpha=0.7, color='skyblue')
    plt.title('Histogram of Realised Quantile Levels')
    plt.xlabel('Quantile Level')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(f"{filename}_histogram.png")
    plt.close()

def save_calibration_plot(y_true, mus, sigmas, filename):
    quantiles = norm.cdf(y_true, loc=mus, scale=sigmas)
    sorted_quantiles = np.sort(quantiles)
    cumulative_proportions = np.linspace(1 / len(sorted_quantiles), 1, len(sorted_quantiles))

    plt.figure(figsize=(8, 8))
    plt.plot(sorted_quantiles, cumulative_proportions, label='Realised Quantile Levels', color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.title('Calibration Plot')
    plt.xlabel('Theoretical Quantile Levels')
    plt.ylabel('Cumulative Proportion of Realized Quantile Levels')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(f"{filename}_calibration.png")
    plt.close()
