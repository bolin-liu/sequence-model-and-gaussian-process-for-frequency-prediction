import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from statsmodels.tsa.stattools import acf


from loss_functions import student_loss, cauchy_loss, correlated_gaussian_loss
from utilities import prepare_covariance_matrix

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 15

# import data
day_ahead_features_test = pd.read_pickle(r"..\data\day_ahead_features_test.pkl")
frequency_test = pd.read_pickle(r"..\data\frequency_test.pkl")
day_ahead_features_test_np = day_ahead_features_test.to_numpy()
frequency_test_np = frequency_test.to_numpy()
inputs_test = day_ahead_features_test_np

scaler = joblib.load(f"../trained_models/scaler.gz")
inputs_test = scaler.transform(inputs_test)
outputs_test = frequency_test_np - 50.0

# load models
loss_student = {"loss": student_loss}
loss_cauchy = {"loss": cauchy_loss}

model_transformer_gaussian = load_model(f"../trained_models/transformer_gaussian")
model_transformer_student = load_model(f"../trained_models/student", custom_objects=loss_student)
model_transformer_cauchy = load_model(f"../trained_models/cauchy", custom_objects=loss_cauchy)

# load sequence models
aggregation = 15
N = 3600 // aggregation
covariance_matrix_rq = prepare_covariance_matrix(N, kernel_type='RationalQuadratic')
loss_rq = {"loss": correlated_gaussian_loss(N, covariance_matrix_rq)}
model_rational_quadratic = load_model(f"../trained_models/correlated_gaussian_Rational_Quadratic_GRU",
                                      custom_objects=loss_rq)

# 2019 Januar
mask = (day_ahead_features_test.index >= '2019-01-01') & (day_ahead_features_test.index <= '2019-01-31')

matching_indices = day_ahead_features_test.index.where(mask).dropna()
row_numbers = [day_ahead_features_test.index.get_loc(index) for index in matching_indices]

# real data
input_test_01_2019 = inputs_test[row_numbers, :]
outputs_test_01_2019 = outputs_test[row_numbers, :]
outputs_test_angular_frequency_01_2019 = outputs_test_01_2019.astype(np.float32) * (2 * np.pi)

# gaussian independent
predictions_test_gaussian = model_transformer_gaussian.predict(input_test_01_2019)
mu_test_gaussian = np.array(2 * np.pi * predictions_test_gaussian[:, :3600]).flatten()
sigma_test_gaussian = np.array(2 * np.pi * tf.math.softplus(predictions_test_gaussian[:, 3600:])).flatten()

# student
predictions_test_student = model_transformer_student.predict(input_test_01_2019)
mu_student = 2 * np.pi * predictions_test_student[:, :3600].flatten()
scale_student = 2 * np.pi * np.array(tf.math.softplus(predictions_test_student[:, 3600:7200])).flatten()
v = np.array(tf.math.softplus(predictions_test_student[:, 7200:])).flatten()

# cauchy
predictions_test_cauchy = model_transformer_cauchy.predict(input_test_01_2019)
cauchy_median = 2 * np.pi * predictions_test_cauchy[:, 0:3600].flatten()
cauchy_s = 2 * np.pi * np.array(tf.math.softplus(predictions_test_cauchy[:, 3600:])).flatten()

# correlated model
predictions_rational_quadratic = model_rational_quadratic.predict(input_test_01_2019)
mu_rational_quadratic = 2 * np.pi * predictions_rational_quadratic[:, :N]
sigma2_rational_quadratic = (2 * np.pi) ** 2 * tf.math.sigmoid(predictions_rational_quadratic[:, N:])
sigma2_rational_quadratic = np.array(sigma2_rational_quadratic).reshape(-1, 1, 1)
sigma2_rational_quadratic = np.array(sigma2_rational_quadratic) * covariance_matrix_rq

# synthetic data
sample_gaussian = sigma_test_gaussian * np.random.randn(sigma_test_gaussian.size) + mu_test_gaussian
sample_student = scale_student * np.random.standard_t(v) + mu_student
sample_cauchy = cauchy_s * np.random.standard_cauchy(cauchy_s.size) + cauchy_median
samples_correlated_gaussian = np.array(
    [np.random.multivariate_normal(mu_rational_quadratic[i], sigma2_rational_quadratic[i], 1)[0] for i in
     range(len(mu_rational_quadratic))])
samples_correlated_gaussian = samples_correlated_gaussian.flatten()

samples_correlated_gaussian_fi = samples_correlated_gaussian[
    (samples_correlated_gaussian >= -0.6) & (samples_correlated_gaussian <= 0.6)]
sample_gaussian = sample_gaussian[(sample_gaussian >= -0.6) & (sample_gaussian <= 0.6)]
sample_student = sample_student[(sample_student >= -0.6) & (sample_student <= 0.6)]
sample_cauchy = sample_cauchy[(sample_cauchy >= -0.6) & (sample_cauchy <= 0.6)]
real_data = outputs_test_angular_frequency_01_2019.flatten()
real_data = real_data[(real_data >= -0.6) & (real_data <= 0.6)]

mu, std = norm.fit(outputs_test_angular_frequency_01_2019.flatten())
x = np.linspace(-0.6, 0.6, 100)
p = norm.pdf(x, mu, std)
# ------------------------------------------------------------------------------------------------------------------------
# 
# tail behavior

plt.figure(figsize=(10, 6))
sns.histplot(real_data, kde=True, stat="density", label="Real Data", bins=30, color="skyblue",linewidth=2.5)
sns.kdeplot(sample_gaussian, label="Independent Gaussian Transformer", linestyle="--", color="orange", linewidth=2.5)
sns.kdeplot(sample_student, label="Student's t Transformer", linestyle="--", linewidth=2.5)
sns.kdeplot(sample_cauchy, label="Cauchy Transformer", linestyle="--", linewidth=2.5)
plt.plot(x, p, label="Fitted Normal PDF", color="red", linewidth=2.5)
plt.legend()
plt.xlabel("Value")
plt.ylabel("Density")
plt.yscale('log')
plt.xlim(-0.6, 0.6)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# 
# ACF of Frequency and its increments

data_syn = samples_correlated_gaussian
data_real = outputs_test_angular_frequency_01_2019[:, ::aggregation].flatten()
data_real_diff = np.diff(data_real)
data_syn_diff = np.diff(data_syn)

acf_values_syn = acf(data_syn, nlags=240*2)
acf_values_real = acf(data_real, nlags=240*2)

acf_values_syn_diff = acf(data_syn_diff, nlags=20)
acf_values_real_diff = acf(data_real_diff, nlags=20)


fig, axs = plt.subplots(1, 2, figsize=(10, 5)) 

fig.suptitle('Autocorrelation with 1 lag = 15s')

axs[0].plot(range(len(acf_values_real)), acf_values_real, label='Real data')
axs[0].plot(range(len(acf_values_syn)), acf_values_syn, label='Synthetic Data')
axs[0].set_xlabel('Lag')
axs[0].set_ylabel('ACF of frequency deviations')
axs[0].legend()
axs[0].text(0.01, 1.1, '(a)', transform=axs[0].transAxes, verticalalignment='top', horizontalalignment='left')

axs[1].plot(range(len(acf_values_real_diff)), acf_values_real_diff, label='Real increments')
axs[1].plot(range(len(acf_values_syn_diff)), acf_values_syn_diff, label='Synthetic increments')
axs[1].set_xlabel('Lag')
axs[1].set_ylabel('ACF of frequency increments')
axs[1].legend()
axs[1].text(0.01, 1.1, '(b)', transform=axs[1].transAxes, verticalalignment='top', horizontalalignment='left')

for ax in axs:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
