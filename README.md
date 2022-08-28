# Reducing-Systemic-Risk-with-DDPG

We use the DDPG algorithm to reduce systemic risk in terms of the DebtRank Measure.

Initial hyperparameters for the DDPG agent and parameters for the complex network can be set in the train.py file.

To change the weights on the reward function uncomment the respective weights in _calculate_debtrank_difference() method in model_network.py
