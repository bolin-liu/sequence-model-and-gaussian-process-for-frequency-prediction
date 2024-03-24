import joblib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

from loss_functions import correlated_gaussian_loss
from proper_scoring_rule import calculate_negative_log_likelihood, calculate_negative_log_likelihood_multi_gaussian, \
    energy_scores_for_multiple_ys_and_gaussians
from utilities import prepare_covariance_matrix, compute_conditional_stats

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# import data
day_ahead_features_test = pd.read_pickle(r"..\data\day_ahead_features_test.pkl")
frequency_test = pd.read_pickle(r"..\data\frequency_test.pkl")
day_ahead_features_test_np = day_ahead_features_test.to_numpy()
frequency_test_np = frequency_test.to_numpy()
inputs_test = day_ahead_features_test_np
scaler = joblib.load(f"../trained_models/scaler.gz")
inputs_test = scaler.transform(inputs_test)

# Frequency Deviation
outputs_test = frequency_test_np - 50.0

# load sequence models
aggregation = 15
N = 3600 // aggregation
covariance_matrix_eq = prepare_covariance_matrix(N, kernel_type='ExponentiatedQuadratic')
covariance_matrix_rq = prepare_covariance_matrix(N, kernel_type='RationalQuadratic')
loss_eq = {"loss": correlated_gaussian_loss(N, covariance_matrix_eq)}
loss_rq = {"loss": correlated_gaussian_loss(N, covariance_matrix_rq)}

# standard with transformer
model_rational_quadratic = load_model(f"../trained_models/correlated_gaussian_rational_quadratic",
                                      custom_objects=loss_rq)
model_exponentiated_quadratic = load_model(f"../trained_models/correlated_gaussian_Exponentiated_Quadratic",
                                           custom_objects=loss_eq)

# model_rational_quadratic = load_model(f"../trained_models/correlated_gaussian_rational_quadratic_GRU",
#                                       custom_objects=loss_rq)
# model_exponentiated_quadratic = load_model(f"../trained_models/correlated_gaussian_Exponentiated_Quadratic_GRU",
#                                            custom_objects=loss_eq)
model_transformer_gaussian = load_model(f"../trained_models/transformer_gaussian_15s")

outputs_test_angular_frequency = outputs_test.astype(np.float32) * (2 * np.pi)

outputs_test_angular_frequency = outputs_test_angular_frequency[:, ::aggregation]

predictions_rational_quadratic = model_rational_quadratic.predict(inputs_test)
predictions_exponentiated_quadratic = model_exponentiated_quadratic.predict(inputs_test)

mu_rational_quadratic = 2 * np.pi * predictions_rational_quadratic[:, :N]
mu_exponentiated_quadratic = 2 * np.pi * predictions_exponentiated_quadratic[:, :N]

sigma2_rational_quadratic = (2 * np.pi) ** 2 * tf.math.sigmoid(predictions_rational_quadratic[:, N:])
sigma2_rational_quadratic = np.array(sigma2_rational_quadratic).reshape(-1, 1, 1)
sigma2_rational_quadratic = np.array(sigma2_rational_quadratic) * covariance_matrix_rq

sigma2_exponentiated_quadratic = (2 * np.pi) ** 2 * tf.math.sigmoid(predictions_exponentiated_quadratic[:, N:])
sigma2_exponentiated_quadratic = np.array(sigma2_exponentiated_quadratic).reshape(-1, 1, 1)
sigma2_exponentiated_quadratic = np.array(sigma2_exponentiated_quadratic) * covariance_matrix_eq

predictions_test_transformer = model_transformer_gaussian.predict(inputs_test)
mu_test_transformer = 2 * np.pi * predictions_test_transformer[:, :240]
sigma_test_transformer = 2 * np.pi * tf.math.softplus(predictions_test_transformer[:, 240:])

squared_sigmas = sigma_test_transformer ** 2
cov_matrices = np.zeros(
    (sigma_test_transformer.shape[0], sigma_test_transformer.shape[1], sigma_test_transformer.shape[1]))
np.einsum('ij,jk->ijk', squared_sigmas, np.eye(sigma_test_transformer.shape[1]), out=cov_matrices)

# ----------------------------------------------------------------------------------------------------------------------

# scoring rule
# negative log likelihood
nll_rq = calculate_negative_log_likelihood_multi_gaussian(outputs_test_angular_frequency, mu_rational_quadratic,
                                                          sigma2_rational_quadratic)
nll_eq = calculate_negative_log_likelihood_multi_gaussian(outputs_test_angular_frequency, mu_exponentiated_quadratic,
                                                          sigma2_exponentiated_quadratic)
nll_indp = calculate_negative_log_likelihood(mu_test_transformer, sigma_test_transformer,
                                             outputs_test_angular_frequency)

#
# energy score
e_rq = np.median(energy_scores_for_multiple_ys_and_gaussians(outputs_test_angular_frequency, mu_rational_quadratic,
                                                             sigma2_rational_quadratic, m=100))# m = 1000, 2000, 3000
e_eq = np.median(energy_scores_for_multiple_ys_and_gaussians(outputs_test_angular_frequency, mu_exponentiated_quadratic,
                                                             sigma2_exponentiated_quadratic, m=100))
e_indp = np.median(
    energy_scores_for_multiple_ys_and_gaussians(outputs_test_angular_frequency, mu_test_transformer, cov_matrices,
                                                m=100))

results = pd.DataFrame({
    'Model': ['Rational Quadratic', 'Exponentiated Quadratic', 'Independent'],
    'Negative Loglikelihood': [np.median(nll_rq), np.median(nll_eq), np.median(nll_indp)],
    'Energy Score': [np.median(e_rq), np.median(e_eq), np.median(e_indp)]
})



# -----------------------------------------------------------------------------------------------------------------------
# fig
# conditional prediction
prediction_sample_time = []
prediction_sample_time.append(day_ahead_features_test.index.get_loc('2019-11-20 01:00:00'))
prediction_sample_time.append(day_ahead_features_test.index.get_loc('2019-03-06 18:00:00'))

feature_input = []
true_output = []
prediction_rational_quadratic = []
prediction_exponentiated_quadratic = []
mu_rational_quadratic = []
sigma2_rational_quadratic = []

mu_exponentiated_quadratic = []
sigma2_exponentiated_quadratic = []

len_outputs = outputs_test_angular_frequency.shape[1]
for i in range(len(prediction_sample_time)):
    position = prediction_sample_time[i]
    feature_input.append(inputs_test[position, :].reshape(1, 14))
    true_output.append(outputs_test_angular_frequency[position, :].reshape(len_outputs, 1))

    prediction_rational_quadratic.append(model_rational_quadratic.predict(feature_input[i]))
    mu_rational_quadratic.append(np.array(2 * np.pi * prediction_rational_quadratic[i][0, :N].reshape(N, 1)).ravel())
    sigma2_rational_quadratic.append(np.array(np.array(
        (2 * np.pi) ** 2 * tf.math.sigmoid(prediction_rational_quadratic[i][0, N:])).ravel() * covariance_matrix_rq))

    prediction_exponentiated_quadratic.append(model_exponentiated_quadratic.predict(feature_input[i]))
    mu_exponentiated_quadratic.append(
        np.array(2 * np.pi * prediction_exponentiated_quadratic[i][0, :N].reshape(N, 1)).ravel())
    sigma2_exponentiated_quadratic.append(np.array(np.array(
        (2 * np.pi) ** 2 * tf.math.sigmoid(
            prediction_exponentiated_quadratic[i][0, N:])).ravel() * covariance_matrix_eq))

time_steps = np.arange(240)

N = len(time_steps)
fig, axs = plt.subplots(2, 2, figsize=(10, 3))
fig.suptitle('Time Step = 15 s', fontsize=10)

labels = ['(a)', '(b)', '(c)', '(d)']

for i in range(2):
    for j in range(2):
        trans = mtransforms.ScaledTranslation(0, 0.15, fig.dpi_scale_trans)
        axs[i, j].text(0.0, 1.0, labels[i * 2 + j], transform=axs[i, j].transAxes + trans,
                       fontsize=10, verticalalignment='top')
    # Calculation for rational_quadratic
    syn_data_conditional_rational = compute_conditional_stats(mu_rational_quadratic[i], sigma2_rational_quadratic[i],
                                                              true_output[i])
    axs[i, 0].plot(time_steps, true_output[i].reshape(N, 1), label='Actual Data', color='green', linewidth=2)
    axs[i, 0].plot(time_steps, syn_data_conditional_rational[:, 0], label='Rational Quadratic', color='blue',
                   linewidth=2)
    axs[i, 0].fill_between(time_steps, syn_data_conditional_rational[:, 0] - syn_data_conditional_rational[:, 1],
                           syn_data_conditional_rational[:, 0] + syn_data_conditional_rational[:, 1], color='blue',
                           alpha=0.2, linewidth=2)
    axs[i, 0].set_xlabel('Time')
    axs[i, 0].set_ylabel(r'$\omega\, (\text{rad/s})$')
    # Calculation for exponentiated_quadratic
    syn_data_conditional_exponential = compute_conditional_stats(mu_exponentiated_quadratic[i],
                                                                 sigma2_exponentiated_quadratic[i], true_output[i])
    axs[i, 1].plot(time_steps, true_output[i].reshape(N, 1), label='Actual Data', color='green', linewidth=2)
    axs[i, 1].plot(time_steps, syn_data_conditional_exponential[:, 0], label='Exponentiated Quadratic', color='orange',
                   linewidth=2)
    axs[i, 1].fill_between(time_steps, syn_data_conditional_exponential[:, 0] - syn_data_conditional_exponential[:, 1],
                           syn_data_conditional_exponential[:, 0] + syn_data_conditional_exponential[:, 1],
                           color='orange', alpha=0.2, linewidth=2)
    axs[i, 1].set_xlabel('Time')
    axs[i, 1].set_ylabel(r'$\omega\, (\text{rad/s})$')

handles1, labels1 = axs[0, 0].get_legend_handles_labels()
handles2, labels2 = axs[0, 1].get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2
unique = dict(zip(labels, handles))
fig.legend(unique.values(), unique.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.12), fancybox=True, shadow=True,
           ncol=3)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()
