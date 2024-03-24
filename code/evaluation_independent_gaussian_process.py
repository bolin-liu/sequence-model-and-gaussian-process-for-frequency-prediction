import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

from loss_functions import gaussian_loss, student_loss, cauchy_loss
from proper_scoring_rule import crps_norm, calculate_negative_log_likelihood, histogram_of_realized_quantiles, calibration_plot


plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# import test data

day_ahead_features_test = pd.read_pickle(r"..\data\day_ahead_features_test.pkl")

frequency_test = pd.read_pickle(r"..\data\frequency_test.pkl")

day_ahead_features_test_np = day_ahead_features_test.to_numpy()

frequency_test_np = frequency_test.to_numpy()

inputs_test = day_ahead_features_test_np

scaler = joblib.load(f"../trained_models/scaler.gz")
inputs_test = scaler.transform(inputs_test)

outputs_test = frequency_test_np - 50.0

# Time window for evaluation
#time_eval = 900 # the first 15 minutes
time_eval = 900 # the whole hour

# load models
loss = {"loss": gaussian_loss}
model_transformer_gaussian = load_model(f"../trained_models/transformer_gaussian")
model_gru_gaussian = load_model(f"../trained_models/gru_gaussian")
outputs_test_angular_frequency = outputs_test.astype(np.float32) * (2 * np.pi)
predictions_test_gru = model_gru_gaussian.predict(inputs_test)
predictions_test_transformer = model_transformer_gaussian.predict(inputs_test)

# gru independent
mu_test_gru = 2 * np.pi * predictions_test_gru[:, :time_eval]
sigma_test_gru = 2 * np.pi * tf.math.softplus(predictions_test_gru[:, 3600:3600+time_eval])

# attention independent
mu_test_transformer = 2 * np.pi * predictions_test_transformer[:, :time_eval]
sigma_test_transformer = 2 * np.pi * tf.math.softplus(predictions_test_transformer[:, 3600:3600+time_eval])
# ----------------------------------------------------------------------------------------------------------------------


# load baseline models
# y_true for the following models

y_true = np.load('..\data\y_true.npy')
# baseline day ahead
mu_test_da = np.load('..\data\day_ahead_ml_pred_mean.npy')
sigma_test_da = np.load('..\data\day_ahead_ml_pred_stddev.npy')

# baseline ex post
mu_test_ep = np.load('..\data\ex_post_ml_pred_mean.npy')
sigma_test_ep = np.load('..\data\ex_post_ml_pred_stddev.npy')

# constant 0
point_test_c0 = np.zeros(shape=y_true.shape)

# constant 1
# tmax = time_eval
if time_eval == 900:
    mu_test_c1 = np.load('..\data\c_pred_mean900.npy')
    sigma_test_c1 = np.load('..\data\c_pred_stddev900.npy')
elif time_eval == 3600:
    mu_test_c1 = np.load('..\data\c_pred_mean3600.npy')
    sigma_test_c1 = np.load('..\data\c_pred_stddev3600.npy')


# constant 2
point_test_c2 = np.ones(shape=y_true.shape) * outputs_test_angular_frequency[:, 0][:, np.newaxis]

# daily profile
if time_eval == 900:
    mu_test_dp = np.load('..\data\dp_pred_mean900.npy')
    sigma_test_dp = np.load('..\data\dp_pred_stddev900.npy')
elif time_eval == 3600:
    mu_test_dp = np.load('..\data\dp_pred_mean3600.npy')
    sigma_test_dp = np.load('..\data\dp_pred_stddev3600.npy')


# k nearst-neighbor
point_test_knn = np.load(r'..\data\knn_point_predictions.npy') * 2 * np.pi

point_predictions_baseline = {'c0': point_test_c0, 'mu c1': mu_test_c1,
                              'c2': point_test_c2, 'mu daily profile': mu_test_dp,
                              'k nearst neighbor': point_test_knn, 'mu da': mu_test_da, 'mu ep': mu_test_ep}

point_predictions_sequence_model = {'mean gru': mu_test_gru, 'mean transformer': mu_test_transformer}

# load models
loss_student = {"loss": student_loss}
loss_cauchy = {"loss": cauchy_loss}

model_transformer_student = load_model(f"../trained_models/student", custom_objects=loss_student)
model_transformer_cauchy = load_model(f"../trained_models/cauchy", custom_objects=loss_cauchy)

# student
predictions_test_student = model_transformer_student.predict(inputs_test)
mu_student = 2 * np.pi * predictions_test_student[:, :time_eval]
scale_student = 2 * np.pi * np.array(tf.math.softplus(predictions_test_student[:, 3600:3600+time_eval]))
v = np.array(tf.math.softplus(predictions_test_student[:, 7200:7200+time_eval]))

# cauchy
predictions_test_cauchy = model_transformer_cauchy.predict(inputs_test)
cauchy_median = 2 * np.pi * predictions_test_cauchy[:, 0:time_eval]

point_predictions_fat_tails = {'model transformer cauchy': cauchy_median,
                               'model transformer student': mu_student}


probabilistic_prediction_baseline = {'constant profile': {'mu': mu_test_c1, 'sigma': sigma_test_c1},
                                     'daily profile': {'mu': mu_test_dp, 'sigma': sigma_test_dp},
                                     'day ahead': {'mu': mu_test_da, 'sigma': sigma_test_da},
                                     'ex post': {'mu': mu_test_ep, 'sigma': sigma_test_ep}}

probabilistic_prediction_sequence = {'model gru gaussian': {'mu': mu_test_gru, 'sigma': sigma_test_gru},
                                     'model transformer gaussian': {'mu': mu_test_transformer,
                                                                    'sigma': sigma_test_transformer}}


# -----------------------------------------------------------------------------------------------------------------------

# measures for different point estimators

def evaluate_predictions(predictions, true_values, error_metric='MAE'):
    if error_metric == 'MAE':
        return mean_absolute_error(true_values, predictions)
    elif error_metric == 'MSE':
        return mean_squared_error(true_values, predictions)
    elif error_metric == 'RMSE':
        return mean_squared_error(true_values, predictions, squared=False)
    else:
        raise ValueError("Unknown Measure")


error_metrics = ['MAE', 'MSE', 'RMSE']  # Liste der Fehlermaße, die du auswerten möchtest

results_point_predictions = {}

for error_metric in error_metrics:
    results_point_predictions[error_metric] = {'baselines': {}, 'sequence_model': {},  'fat_tails': {}}

    for key in point_predictions_baseline:
        if key == 'k nearst neighbor':
            results_point_predictions[error_metric]['baselines'][key] = evaluate_predictions(
                point_predictions_baseline[key][:, 0:time_eval],
                outputs_test_angular_frequency[:, 0:time_eval],
                error_metric)
        else:
            print()
            results_point_predictions[error_metric]['baselines'][key] = evaluate_predictions(
                point_predictions_baseline[key][:, 0:time_eval],
                y_true[:, 0:time_eval],
                error_metric)

    for key in point_predictions_sequence_model:
        results_point_predictions[error_metric]['sequence_model'][key] = evaluate_predictions(
            point_predictions_sequence_model[key][:, 0:time_eval],
            outputs_test_angular_frequency[:, 0:time_eval],
            error_metric)

    for key in point_predictions_fat_tails:
        results_point_predictions[error_metric]['fat_tails'][key] = evaluate_predictions(
            point_predictions_fat_tails[key][:, 0:time_eval],
            outputs_test_angular_frequency[:, 0:time_eval],
            error_metric)



# proper scoring rules
results_psr = {}
for category, models in [('baseline', probabilistic_prediction_baseline),
                         ('sequence', probabilistic_prediction_sequence)]:
    results_psr[category] = {}
    for model_name, params in models.items():
        mu = params['mu']
        sigma = params['sigma']
        if category == 'baseline':
            # Use y_true for the baseline category
            result1 = np.median(calculate_negative_log_likelihood(mu[:, 0:time_eval], sigma[:, 0:time_eval], y_true[:, 0:time_eval]))
            result2 = np.mean(crps_norm(mu[:, 0:time_eval], sigma[:, 0:time_eval], y_true[:, 0:time_eval]))
        else:
            # Use outputs_test_angular_frequency for other models
            result1 = np.median(calculate_negative_log_likelihood(mu[:, 0:time_eval], sigma[:, 0:time_eval],
                                                                  outputs_test_angular_frequency[:, 0:time_eval]))
            result2 = np.mean(crps_norm(mu[:, 0:time_eval], sigma[:, 0:time_eval], outputs_test_angular_frequency[:, 0:time_eval]))
        results_psr[category][model_name] = {'negative_log_likelihood': result1, 'crps_norm': result2}

results_list_psr = []
for category, models in results_psr.items():
    for model_name, model_results in models.items():
        results_list_psr.append({
            'Category': category,
            'Model': model_name,
            'negative_log_likelihood': model_results['negative_log_likelihood'],
            'crps_norm': model_results['crps_norm']
        })
results_df = pd.DataFrame(results_list_psr)

# -----------------------------------------------------------------------------------------
# prediction sample
# Plot [] in paper
prediction_sample_time = []
prediction_sample_time.append(day_ahead_features_test.index.get_loc('2019-08-05 01:00:00'))
prediction_sample_time.append(day_ahead_features_test.index.get_loc('2019-11-20 01:00:00'))
prediction_sample_time.append(day_ahead_features_test.index.get_loc('2019-01-08 18:00:00'))
prediction_sample_time.append(day_ahead_features_test.index.get_loc('2019-03-06 18:00:00'))

feature_input = []
true_output = []
prediction_gru_gaussian = []
mu_gru_gaussian = []
sigma_gru_gaussian = []
prediction_transformer_gaussian = []
mu_transformer_gaussian = []
sigma_transformer_gaussian = []
len_outputs = outputs_test.shape[1]
for i in range(len(prediction_sample_time)):
    position = prediction_sample_time[i]
    feature_input.append(inputs_test[position, :].reshape(1, 14))
    true_output.append(2 * np.pi * outputs_test[position, :].reshape(len_outputs, 1))

    prediction_gru_gaussian.append(model_gru_gaussian.predict(feature_input[i]))
    mu_gru_gaussian.append(2 * np.pi * prediction_gru_gaussian[i][0, :3600])
    sigma_gru_gaussian.append(2 * np.pi * tf.math.softplus(prediction_gru_gaussian[i][0, 3600:]))

    prediction_transformer_gaussian.append(model_transformer_gaussian.predict(feature_input[i]))
    mu_transformer_gaussian.append(2 * np.pi * prediction_transformer_gaussian[i][0, :3600])
    sigma_transformer_gaussian.append(2 * np.pi * tf.math.softplus(prediction_transformer_gaussian[i][0, 3600:]))

baselines = []
baselines.append(pd.read_csv(r"..\data\prediction_example_0_0.csv").iloc[:, -5:])
baselines.append(pd.read_csv(r"..\data\prediction_example_0_1.csv").iloc[:, -5:])
baselines.append(pd.read_csv(r"..\data\prediction_example_1_0.csv").iloc[:, -5:])
baselines.append(pd.read_csv(r"..\data\prediction_example_1_1.csv").iloc[:, -5:])

fig, axs = plt.subplots(2, 2, figsize=(10, 3))
axs = axs.flatten()


colors = ['green', 'blue', 'purple', 'black', 'red']

for i, ax in enumerate(axs):
    ax.text(0.01, 0.95, f'({chr(97 + i)})', transform=ax.transAxes, fontsize=8, verticalalignment='top')
    # True_output
    ax.plot(true_output[i], label='True Output', color=colors[0])
    # mu_transformer_gaussian and its envelope
    ax.plot(mu_transformer_gaussian[i], label='Mu Transformer Gaussian', color=colors[1])
    ax.fill_between(range(len(mu_gru_gaussian[i])),
                    mu_gru_gaussian[i] - sigma_gru_gaussian[i],
                    mu_gru_gaussian[i] + sigma_gru_gaussian[i], alpha=0.2, color=colors[1])

    # Baselines and their envelopes
    baseline_mean = baselines[i]['y_ml_pred_mean']
    daily_mean = baselines[i]['y_dp_pred_mean']
    ax.plot(daily_mean, label='Daily Profile', color=colors[2])

    ax.plot(point_test_knn[i], label='True Output', color=colors[3])

    ax.plot(baseline_mean, label='Ex', color=colors[4])
    ax.fill_between(range(len(mu_gru_gaussian[i])),
                    baseline_mean + baselines[i]['y_ml_pred_std'],
                    baseline_mean - baselines[i]['y_ml_pred_std'], alpha=0.2, color=colors[4])

    ax.set_xlabel(r'Time ($s$)')
    ax.set_ylabel(r'$\omega\,(\text{rad}/s)$ ')

handles = [plt.Line2D([], [], color=color) for color in colors]
labels = ['Truth', 'Independent Gaussian GRU', 'Daily Profile', 'KNN', 'PIML Ex Post']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=5)
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# calibration plots
for profile, stats in probabilistic_prediction_baseline.items():
    mu_flat = np.array(stats['mu']).flatten()
    sigma_flat = np.array(stats['sigma']).flatten()

    print(f"{profile}")
    histogram_of_realized_quantiles(y_true.flatten(), mu_flat, sigma_flat)
    calibration_plot(y_true.flatten(), mu_flat, sigma_flat)

for model_name, stats in probabilistic_prediction_sequence.items():
    mu_flat = np.array(stats['mu']).flatten()
    sigma_flat = np.array(stats['sigma']).flatten()

    print(f"{model_name}")
    histogram_of_realized_quantiles(np.array(outputs_test_angular_frequency).flatten(), mu_flat, sigma_flat)
    calibration_plot(np.array(outputs_test_angular_frequency).flatten(), mu_flat, sigma_flat)