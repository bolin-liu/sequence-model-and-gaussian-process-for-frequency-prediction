# sequence-model-and-gaussian-process-for-frequency-prediction
Code for the Paper: Predicting grid frequency short-term dynamics with Gaussian processes and sequence modeling (published)

- models.py contains the model structures of the customised transformer structure and gru structure.
- experiment.py script for training different models
- loss_functions.py contains different loss functions for model training.
- proper_scoring_rule.py methods for evaluating probabilistic predictions
- k_nearst_neighbor.py script for training a simple knn-model
- scripts for reproducing study results
  - evaluation_independent_gaussian_process.py
  - evaluation_correlated_gaussian_process.py
  - tail_behavior_ACF.py
- utilities.py helper functions

Training data (external features and frequency data) for the models are the training data used in [1]. To use our code, please follow the description in [1] to get the preprocessed training and test data and save it as frequency_train.pkl, frequency.test.pkl, day_ahead_features_train.pkl and day_ahead_features_test.pkl. 

In addition, the data set in [2] can be used to reproduce the figures and results of the paper.

[1]: https://github.com/johkruse/PIML-for-grid-frequency-modelling
[2]: https://zenodo.org/records/10866500

