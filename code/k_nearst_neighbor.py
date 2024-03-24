from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# data import
day_ahead_features_train = pd.read_pickle(r"..\data\day_ahead_features_train.pkl")
day_ahead_features_test = pd.read_pickle(r"..\data\day_ahead_features_test.pkl")
frequency_train = pd.read_pickle(r"..\data\frequency_train.pkl")
frequency_test = pd.read_pickle(r"..\data\frequency_test.pkl")

day_ahead_features_train_np = day_ahead_features_train.to_numpy()
day_ahead_features_test_np = day_ahead_features_test.to_numpy()
frequency_train_np = frequency_train.to_numpy()
frequency_test_np = frequency_test.to_numpy()

inputs = day_ahead_features_train_np
inputs_test = day_ahead_features_test_np

scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)
inputs_test = scaler.transform(inputs_test)

outputs = frequency_train_np
outputs = outputs - 50.0  # Frequency Deviation
outputs_test = frequency_test_np - 50.0


k_range = range(1, 151, 10)


avg_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, inputs, outputs, cv=5, scoring='neg_mean_squared_error')
    avg_scores.append(-scores.mean())

best_k = k_range[np.argmin(avg_scores)]

best_knn = KNeighborsRegressor(n_neighbors=best_k)
best_knn.fit(inputs, outputs)
test_predictions = best_knn.predict(inputs_test)
np.save(r'..\data\knn_point_predictions.npy', test_predictions)


# Plot erstellen
plt.plot(k_range, avg_scores)
plt.xlabel('value of k')
plt.ylabel('MSE')
plt.title('kNN performance for different k')
plt.show()