import os
import numpy as np
from datetime import datetime
from numpy import genfromtxt
from random import randrange

# For using R to generate the initial network.
class NetworkImporter(object):
    """
    To use: Need to generate random networks before hand using systemicrisk package in R and save in a folder.
    
    E.g. Generating 10000 random networks of size N=30, M=3 the folder name containing these networks must be

    R_initial_networks_30-2

    the network names in the folder need to be 
    
    R_liability_network_1
    R_liability_network_2
    ...
    R_liability_network_10000

    From this folder we sample a random network.

    """
    def __init__(self, parent_dir, liability_network_dir, networth_dir, n_nodes, num_layers, uniform=False, seed=None):
        self.liability_fname = "R_liability_network"
        self.networth_fname = "R_networth_values"

        if uniform == True:
            print("Sampling from uniform folder")
            parent_dir = os.path.join(parent_dir, "uniform")

        self.num_layers = num_layers
        self.liability_network_dir = []
        for alpha in range(3): # To generate all names and so you can reverse the list later
            self.liability_network_dir.append(os.path.join(parent_dir, liability_network_dir + "_" + str(n_nodes) + "-" + str(alpha)))
        self.liability_network_dir.reverse() # Reverses the list so that we start sampling from the highest (most value network)
        self.networth_dir = os.path.join(parent_dir, networth_dir + "_" + str(n_nodes))
        self.liability_network = self.sample_liability_network(seed=seed)
        # self.networth_values = self._get_networth_values()[1:, 1:]
        self.N_nodes = self._get_network_properties()
        if self.N_nodes != n_nodes:
            raise Exception("The N_nodes input does not match the R N_nodes value.")

    def _get_network_properties(self):
        # Get the properties of the network.
        return self.liability_network[0].shape[0]

    def sample_liability_network(self, seed=None):
        liability_network = []
        for alpha in range(self.num_layers):
            nfiles = len([name for name in os.listdir(self.liability_network_dir[alpha])])
            if seed is not None:
                mat_count = seed[alpha]
            else:
                mat_count = randrange(1, nfiles)+1
            liability_mat_name = self.liability_fname + "_" + str(mat_count) +".csv"
            mat = genfromtxt(os.path.join(self.liability_network_dir[alpha], liability_mat_name), delimiter=',')[1:, 1:].copy()
            liability_network.append(mat[:-1,:-1]) # Need to remove last row/column because Rstudio puts an extra row/column

        liability_network.reverse()

        return np.array(liability_network)

    def _get_networth_values(self):
        """ Get the networth values """

        return genfromtxt(
            os.path.join(self.networth_dir, self.networth_fname+".csv"),
            delimiter=','
        )

    def thetas(self, total_lending_amount):
        """ This function calculates the proportion of loans used"""

        total_liability_value = self.liability_network.sum()
        if total_lending_amount > total_liability_value:
            thetas = []
            for alpha in range(self.num_layers):
                thetas.append(self.liability_network[alpha].sum()/total_lending_amount)
    
        else:
            print("The total asset value is too small.")
            raise Exception

        return thetas

class ResultLogger(object):
    """ This class is used to log the results and save them """
    def __init__(self, parent_dir, initial_dir, result_dir, tensorboard_dir, print_dir, parameters):

        self.parent_dir = parent_dir
        self.initial_dir = initial_dir
        self.result_dir = result_dir
        self.tensorboard_dir = tensorboard_dir
        self.print_dir = print_dir
        self.parameters = parameters

    def make_log_and_print_dir(self, init_debtrank):
        """ Make the directory for the """

        folder_name = "eval_networks"
        init_debtrank = str(init_debtrank)[:5]
        self.log_result_path = os.path.join(self.parent_dir, self.result_dir, folder_name, init_debtrank)
        self.print_result_path = os.path.join(self.parent_dir, self.print_dir, folder_name, init_debtrank)

        if not os.path.exists(self.log_result_path):
            os.makedirs(self.log_result_path)
        if not os.path.exists(self.print_result_path):
            os.makedirs(self.print_result_path)

    def make_tensorboard_dir(self, initial_debtrank):
        """ Make the dir for the tensorboard results """

        date = str(datetime.today().strftime('%Y-%m-%d'))
        train_path = os.path.join(self.parent_dir, self.tensorboard_dir, initial_debtrank + "_" + date + "_train")
        eval_path = os.path.join(self.parent_dir, self.tensorboard_dir, initial_debtrank + "_" + date +  "_eval")

        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)

        return train_path, eval_path

    def make_policy_saver_dir(self, initial_debtrank):
        """ Make the dir for the policy to load """
        self.policy_path = os.path.join(self.parent_dir, "policy", str(initial_debtrank))
        if not os.path.exists(self.policy_path):
            os.makedirs(self.policy_path)

    def print_debtrank(self, complex_network):
        """ Used in the model network to save the log """
        debtrank_str = str(complex_network.curr_debtrank.sum())
        with open(os.path.join(self.log_result_path, debtrank_str + "_network.npy"), 'wb') as f:
            np.savez(
                f,
                network=complex_network.multi_adj,
                debtrank=complex_network.curr_debtrank,
                networth=[bank_node.net_worth for bank_node in  complex_network.bank_list],
                creditrisk=[bank_node.leverage_ratio for bank_node in  complex_network.bank_list],
                parameters=self.parameters
            )
    
    def update_policy_path(self, complex_network):
        """ Used to update the path of where we save the current policy """

        debtrank_str = str(complex_network.curr_debtrank.sum())
        self.curr_policy_dir = os.path.join(self.policy_path, debtrank_str)
