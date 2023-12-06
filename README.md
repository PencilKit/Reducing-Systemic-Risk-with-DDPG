# Reducing Systemic Risk with DDPG

# Description
We use the DDPG algorithm to reduce systemic risk in terms of the DebtRank Measure. This repo contains the python code used to train and evaluate the DDPG agent to produce the results in Le and Ku (2022). Reducing the systemic risk of a financial network via re-organization of the interbank relationship was inspired by Diem et al. (2020). 

Initial hyperparameters for the DDPG agent and parameters for the complex network can be set in the train.py file.

References:

Diem, C., Pichler, A., & Thurner, S. (2020). What is the minimal systemic risk in financial exposure networks?. Journal of Economic Dynamics and Control, 116, 103900. https://doi.org/https://doi.org/10.1016/j.jedc.2020.103900

Le, R., & Ku, H. (2022). Reducing systemic risk in a multi-layer network using reinforcement learning. Physica A: Statistical Mechanics and its Applications, 605, 128029. https://doi.org/https://doi.org/10.1016/j.physa.2022.128029

# Notes
To change the weights on the reward function uncomment the respective weights in _calculate_debtrank_difference() method in model_network.py
