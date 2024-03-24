# sequence-model-and-gaussian-process-for-frequency-prediction
Code for the Paper: Predicting grid frequency short-term dynamics with Gaussian processes and sequence modeling (submitted)

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


