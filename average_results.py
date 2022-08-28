import os
from absl import logging
import tensorflow as tf
import numpy as np
import networkx as nx
import safety_layer as sl
from tf_agents.environments import tf_py_environment
from datetime import datetime

from environment import NetworkEnvironment
from network_gym.envs import model_network as network_model # Multi-layer methods

import importing_modules as im

import os

import cProfile


import pandas as pd
def jaccard_similarity(g, h):
    """ Takes the edge list of graph g and h """
    i = set(g).intersection(h)
    return round(len(i) / (len(g) + len(h) - len(i)),3)

def average_dr(N_NODES, NUM_LAYERS, NUM_EPISODES, LEVERAGE_EXP=False, CASE=None, POLICY=None, results=None):
    """ Case determines if we do a different type of initial network based on the 
    leverage ratio? """
    pr = cProfile.Profile()
    pr.enable()

    logging.set_verbosity(logging.DEBUG)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    np.random.seed(1)

    NETWORK_SEED = [2, 2, 2] # For regular experiments
    # NETWORK_SEED = [3, 3, 3] #N=30 for leverage 
    BETA = 0.18

    if LEVERAGE_EXP == True:
        UNIFORM = True
        GAMMA_NET = [0.2] # For making uniform DR distribution
        NETWORK_SEED = [2] # For N=30 uniform network

    else:
        UNIFORM = False
        GAMMA_NET = np.random.uniform(low=0.07, high=0.20, size=N_NODES) # new # Use this for main experiments


    if CASE == "uniform":
        if LEVERAGE_EXP == False:
            UNIFORM = True
            GAMMA_NET = [0.2]
            NETWORK_SEED = [3 ,3, 3]
        else:
            raise ValueError("ERROR: The LEVERAGE_EXP should be set to false")

    R = 0.9
    REW_LAMBDA = 1.0
    REW_RHO = 1.0

    # Action scale to scale actions \in [-1.0, 1.0]^d
    ACTION_SCALE = 100000


    MAX_EPISODE_STEPS = 50


    critic_learning_rate = 3e-4
    actor_learning_rate = 3e-5
    target_update_tau = 0.001
    target_update_period = 1.0
    gamma = 0.8
    reward_scale_factor = 1.0  

    gaussian_std = 0.15

    # Load the data
    path_base = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    root_str = os.path.join(path_base, "data")
    root_R_str = os.path.join(path_base, "data", "R folder", "Systemic Risk", "Data")

    NetworkImporter = im.NetworkImporter(
        root_R_str,
        "R_initial_networks",
        "R_networth",
        N_NODES,
        NUM_LAYERS,
        uniform=UNIFORM,
        seed=NETWORK_SEED
        )

    asset_scale = 2.22

    TOTAL_ASSETS = NetworkImporter.liability_network.sum() * asset_scale
    THETAS = NetworkImporter.thetas(TOTAL_ASSETS)
    print(THETAS)
    parameters = {
        "n_nodes": N_NODES,
        "num_layers": NUM_LAYERS,
        "beta": BETA,
        "gamma_net": GAMMA_NET,
        "rew_lambda": REW_LAMBDA,
        "rew_rho": REW_RHO,
        "asset_scale": asset_scale,
        "total_assets": TOTAL_ASSETS,
        "thetas": THETAS,
        "actor lr": actor_learning_rate,
        "critic lr": critic_learning_rate,
        "gamma": gamma,
        "reward_scale_factor": reward_scale_factor,
        "target_update_tau": target_update_tau,
        "target_update_period": target_update_period,
        "gaussian_std": gaussian_std,
        "network_seed": NETWORK_SEED
    }

    ResultLogger = im.ResultLogger(
        parent_dir=root_str,
        initial_dir="initial_network", # Where the initial network data is saved
        result_dir="network_py_data", # Where the result network data are saved
        tensorboard_dir="results", # Where the tensorboard results are saved
        print_dir="to_print", # Where the prints are saved
        parameters=parameters
        )

    eval_complex_network = network_model.Multigraph(
        N_nodes=N_NODES,
        num_layers=NUM_LAYERS,
        total_assets=TOTAL_ASSETS,
        thetas=THETAS,
        beta=BETA,
        gamma=GAMMA_NET,
        r=R,
        c_eps=None,
        rew_lambda=REW_LAMBDA,
        rew_rho=REW_RHO,
        network_importer=NetworkImporter,
        is_eval=True
    )
    print(eval_complex_network.init_debtrank.sum())

    training_complex_network = network_model.Multigraph(
        N_nodes=N_NODES,
        num_layers=NUM_LAYERS,
        total_assets=TOTAL_ASSETS,
        thetas=THETAS,
        beta=BETA,
        gamma=GAMMA_NET,
        r=R,
        c_eps=None,
        rew_lambda=REW_LAMBDA,
        rew_rho=REW_RHO,
        network_importer=NetworkImporter,
        is_eval=False
    )

    if LEVERAGE_EXP == True:
        half_N = int(N_NODES/2)
        gamma_diff = 0.05
        gamma_l = 0.07
        gamma_h = 0.99
        GAMMA_NET = np.concatenate(
            (np.random.uniform(low=gamma_h - gamma_diff, high=gamma_h, size=half_N),
            np.random.uniform(low=gamma_l, high=gamma_l + gamma_diff, size=half_N))
            )

        for eval_node, training_node, lev_val in zip(eval_complex_network.bank_list, training_complex_network.bank_list, GAMMA_NET):
            eval_node.leverage_ratio = lev_val
            training_node.leverage_ratio = lev_val

    print(training_complex_network.init_debtrank.sum())
    TrainingComplexNetworkEnvironment = NetworkEnvironment(training_complex_network, ResultLogger, MAX_EPISODE_STEPS, ACTION_SCALE)
    EvalComplexNetworkEnvironment = NetworkEnvironment(eval_complex_network, ResultLogger, MAX_EPISODE_STEPS, ACTION_SCALE)

    # Create the safety layer functions
    safety_layer = sl.SafetyLayer(
        env=TrainingComplexNetworkEnvironment
    )
    # Create the environment
    TrainingComplexNetworkEnvironment.add_safety_layer(safety_layer)
    EvalComplexNetworkEnvironment.add_safety_layer(safety_layer)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    train_env = tf_py_environment.TFPyEnvironment(TrainingComplexNetworkEnvironment)

    policy_path = os.path.join(root_str, "policy", POLICY)

    policy_list = np.array([float(dr) for dr in os.listdir(policy_path)])

    min_policy_folder_name = str(policy_list[0]) # The policy to use
    policy_path = os.path.join(policy_path, min_policy_folder_name)
    saved_policy = tf.saved_model.load(policy_path)


    def average_main_results():
        
        init_dr_list = []
        dr_list = []
        for _ in range(NUM_EPISODES):
            time_step = train_env.reset()
            init_dr_list.append(
                TrainingComplexNetworkEnvironment._complex_network._sum_debtranks(TrainingComplexNetworkEnvironment._complex_network.init_debtrank).sum()
            )

            while not time_step.is_last():
                action_step = saved_policy.action(time_step)
                time_step = train_env.step(action_step.action)
            dr_list.append(
                TrainingComplexNetworkEnvironment._complex_network._sum_debtranks(TrainingComplexNetworkEnvironment._complex_network.curr_debtrank).sum()
                )

        init_dr_list = np.array(init_dr_list)
        dr_list = np.array(dr_list)

        reduction_list = 100*np.abs(dr_list - init_dr_list)/init_dr_list

        initial_avg = np.mean(init_dr_list)
        initial_std = np.std(init_dr_list)
        opt_avg = np.mean(dr_list)
        opt_std = np.std(dr_list)

        reduction_avg = np.mean(reduction_list)
        reduction_std = np.std(reduction_list)

        return initial_avg, opt_avg, reduction_avg, initial_std, opt_std, reduction_std

    def average_statistics():
        n_eps=1


        init_density_dict = {
            0: [],
            1: [],
            2: []
        }

        opt_density_dict = {
            0: [],
            1: [],
            2: []
        }

        init_jaccard_mat = []
        opt_jaccard_mat = []

        init_cc = []
        opt_cc = []

        init_nn = []
        opt_nn = []

        init_dr_list = []
        dr_list = []
        for _ in range(NUM_EPISODES):
            time_step = train_env.reset()
            init_weight_mats = TrainingComplexNetworkEnvironment._complex_network.multi_adj

            init_dr_list.append(
                TrainingComplexNetworkEnvironment._complex_network._sum_debtranks(TrainingComplexNetworkEnvironment._complex_network.init_debtrank).sum()
            )

            while not time_step.is_last():
                action_step = saved_policy.action(time_step)
                time_step = train_env.step(action_step.action)

            dr_list.append(
                TrainingComplexNetworkEnvironment._complex_network._sum_debtranks(TrainingComplexNetworkEnvironment._complex_network.curr_debtrank).sum()
                )

            weight_mats = TrainingComplexNetworkEnvironment._complex_network.multi_adj

            init_graph_layers = [] # list of each layer of init network
            opt_graph_layers = [] # list of each layer of opt network
            for m in range(NUM_LAYERS):

                init_weight_mat = init_weight_mats[m]
                init_weight_mat[init_weight_mat < n_eps] = 0.
                weight_mat = weight_mats[m]
                weight_mat[weight_mat < n_eps] = 0.

                init_G = nx.from_numpy_array(init_weight_mat, parallel_edges=False, create_using=nx.DiGraph)
                opt_G = nx.from_numpy_array(weight_mat, parallel_edges=False, create_using=nx.DiGraph)
                
                # Calculate the density
                init_density_dict[m].append(nx.density(init_G))
                opt_density_dict[m].append(nx.density(opt_G))

                # Calculate the clustering coefficient
                init_cc.append(nx.clustering(init_G))
                opt_cc.append(nx.clustering(opt_G))

                # NOTE: Calculate neighbour degree (need to modify function to calculate eq. 69)
                init_nn.append(nx.average_neighbor_degree(init_G, source='in+out', target='in+out', weight='weight'))
                opt_nn.append(nx.average_neighbor_degree(opt_G, source='in+out', target='in+out', weight='weight'))

                # The neighbourhood degrees to plot
                init_graph_layers.append(init_G)
                opt_graph_layers.append(opt_G)

            temp_init_table = []
            temp_opt_table = []
            for n in range(NUM_LAYERS):
                init_row = []
                opt_row = []
                for m in range(NUM_LAYERS):
                    init_row.append(1-jaccard_similarity(init_graph_layers[n].to_undirected().edges(), init_graph_layers[m].to_undirected().edges()))
                    opt_row.append(1-jaccard_similarity(opt_graph_layers[n].to_undirected().edges(), opt_graph_layers[m].to_undirected().edges()))
                temp_init_table.append(init_row)
                temp_opt_table.append(opt_row)

            init_jaccard_mat.append(temp_init_table)
            opt_jaccard_mat.append(temp_opt_table)

        # Do the averaging
        avg_init_density_dict = {
            0: 0.0,
            1: 0.0,
            2: 0.0
        }
        std_init_density_dict = {
            0: 0.0,
            1: 0.0,
            2: 0.0
        }
        avg_opt_density_dict = {
            0: 0.0,
            1: 0.0,
            2: 0.0
        }
        std_opt_density_dict = {
            0: 0.0,
            1: 0.0,
            2: 0.0
        }
        for m in range(NUM_LAYERS):
            avg_init_density_dict[m] = np.array([dens for dens in init_density_dict[m]]).mean()
            avg_opt_density_dict[m] = np.array([dens for dens in opt_density_dict[m]]).mean()

            std_init_density_dict[m] = np.array([dens for dens in init_density_dict[m]]).std()
            std_opt_density_dict[m] = np.array([dens for dens in opt_density_dict[m]]).std()

        # Jaccard averaging
        avg_init_jaccard_mat = np.mean(init_jaccard_mat, axis=0)
        avg_opt_jaccard_mat = np.mean(opt_jaccard_mat, axis=0)

        std_init_jaccrd_mat = np.std(init_jaccard_mat, axis=0)
        std_opt_jaccard_mat = np.std(opt_jaccard_mat, axis=0)

        return avg_init_density_dict, avg_opt_density_dict, avg_init_jaccard_mat, avg_opt_jaccard_mat, std_init_density_dict, std_opt_density_dict, std_init_jaccrd_mat, std_opt_jaccard_mat


    def average_cc_nn():
        n_eps=1

        init_cc = {
            0: [],
            1: [],
            2: []
        }
        opt_cc = {
            0: [],
            1: [],
            2: []
        }

        init_nn = {
            0: [],
            1: [],
            2: []
        }
        opt_nn = {
            0: [],
            1: [],
            2: []
        }

        initial_debtrank = {
            0: [],
            1: [],
            2: []
        }
        optimized_debtrank = {
            0: [],
            1: [],
            2: []
        }
        for n_episode in range(NUM_EPISODES):
            print("NUM_EPISODE=",n_episode)

            time_step = train_env.reset()
            init_weight_mats = TrainingComplexNetworkEnvironment._complex_network.multi_adj

            init_dr = (
                TrainingComplexNetworkEnvironment._complex_network._sum_debtranks(TrainingComplexNetworkEnvironment._complex_network.init_debtrank)
            )

            while not time_step.is_last():
                action_step = saved_policy.action(time_step)
                time_step = train_env.step(action_step.action)

            opt_dr = (
                TrainingComplexNetworkEnvironment._complex_network._sum_debtranks(TrainingComplexNetworkEnvironment._complex_network.curr_debtrank)
                )

            weight_mats = TrainingComplexNetworkEnvironment._complex_network.multi_adj

            init_graph_layers = [] # list of each layer of init network
            opt_graph_layers = [] # list of each layer of opt network

            for m in range(NUM_LAYERS):
                init_weight_mat = init_weight_mats[m]
                init_weight_mat[init_weight_mat < n_eps] = 0.
                weight_mat = weight_mats[m]
                weight_mat[weight_mat < n_eps] = 0.

                init_G = nx.from_numpy_array(init_weight_mat, parallel_edges=False, create_using=nx.DiGraph)
                opt_G = nx.from_numpy_array(weight_mat, parallel_edges=False, create_using=nx.DiGraph)
           

                # Calculate the clustering coefficient
                init_cc[m].append(np.mean([val for val in nx.clustering(init_G).values()]))
                opt_cc[m].append(np.mean([val for val in nx.clustering(opt_G).values()]))

                # Calculate neighbour degree
                init_nn[m].append(np.sum([val for val in nx.average_neighbor_degree(init_G, source='in+out', target='in+out', weight='weight').values()]))
                opt_nn[m].append(np.sum([val for val in nx.average_neighbor_degree(opt_G, source='in+out', target='in+out', weight='weight').values()]))

                initial_debtrank[m].append(init_dr[m].sum())
                optimized_debtrank[m].append(opt_dr[m].sum())

                # The neighbourhood degrees to plot
                init_graph_layers.append(init_G)
                opt_graph_layers.append(opt_G)


        return initial_debtrank, optimized_debtrank, init_cc, opt_cc, init_nn, opt_nn


    def average_leverage():
        
        high_dr_list = []
        low_dr_list = []
        for N_epi in range(NUM_EPISODES):
            print("Episode N = ", str(N_epi))
            time_step = train_env.reset()
     
            while not time_step.is_last():
                action_step = saved_policy.action(time_step)
                time_step = train_env.step(action_step.action)
            
            debtrank = TrainingComplexNetworkEnvironment._complex_network._sum_debtranks(TrainingComplexNetworkEnvironment._complex_network.curr_debtrank)
            high_dr_list.append(np.sum(debtrank[0][:int(N_NODES/2)]))
            low_dr_list.append(np.sum(debtrank[0][int(N_NODES/2):]))


        avg_high_dr = np.mean(high_dr_list)
        avg_low_dr = np.mean(low_dr_list)

        std_high_dr = np.std(high_dr_list)
        std_low_dr = np.std(low_dr_list)

        return avg_high_dr, avg_low_dr, std_high_dr, std_low_dr

    if results == "main":
        return average_main_results()
    elif results == "stats":
        return average_statistics()
    elif results == "cc_nn":
        return average_cc_nn()
    elif results == "leverage":
        return average_leverage()


if __name__ == "__main__":

    policy_dict = {
        30: {
            1: "18.161872585900607",
            2: "26.31708381856567",
            3: "33.295454885318456"
        },
        20: {
            1: "13.872525203609701",
            2: "23.260422907686706",
            3: "30.402851902433095"
        },
        10: {
            1: "7.7408578246030135",
            2: "14.021826409799196",
            3: "21.37639441829311"
        }
    }

    policy_uniform_dict = {
        30: {
            1: "12.534112174999047",
            2: "11.848197437601694",
            3: "14.058939038527265"
        },
        20: {
            1: "12.275934293341955",
            2: "11.680745742070073",
            3: "14.629125199358128"
        },
        10: {
            1: "7.074051130008529",
            2: "11.373744396360374",
            3: "14.514386717235313"
        }
    }

    lev_policy_dict = {
        30: {"uniform": "12.665814697812916-lev-uniform-30",
        "linear": "12.665814697812916-lev-linear-30",
        "exponential1": "12.665814697812916-lev-exp1-30", 
        "exponential10": "12.665814697812916-lev-exp10-30"
        },
        20: {"uniform": "11.118127820283755-lev-uniform-20",
        "linear": "11.118127820283755-lev-linear-20",
        "exponential1": "11.118127820283755-lev-exp1-20", 
        "exponential10": "11.118127820283755-lev-exp10-20"
        },
        10: {"uniform": "7.846093132926077-lev-uniform-10",
        "linear": "7.846093132926077-lev-linear-10",
        "exponential1": "7.846093132926077-lev-exp1-10", 
        "exponential10": "7.846093132926077-lev-exp10-10"
        }
    }

    column_name = ["Initial DR", "Optimized DR", "% Reduction", "Init std", "Opt Std", "Red Std"]

    # Load the data
    path_base = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    out_path = os.path.join(path_base, "data", "prints", "table")

    # DR
    do_leverage_results = False
    number_of_episodes = 1
    CASE = "original" # or replcae with "original"

    if do_leverage_results == False:
        layer_ind = [1,2,3]
        table_list = []
        for case in [CASE]:
            if case == "uniform":
                policy_dict = policy_uniform_dict
                print(case)
            elif case=="original":
                policy_dict = policy_dict
            rows = []
            for n in [10, 20, 30]:
                for m in layer_ind:
                    print("WORKING ON: N=", n, " M=", m)
                    avg_initial_dr, avg_optimized_dr, avg_reduction, std_init, std_opt, std_red = average_dr(
                        n,
                        m,
                        number_of_episodes,
                        LEVERAGE_EXP=do_leverage_results,
                        CASE=case,
                        POLICY=policy_dict[n][m],
                        results="main"
                        )

                    rows.append([avg_initial_dr, avg_optimized_dr, avg_reduction, std_init, std_opt, std_red])
                    print("Completed row: ", [avg_initial_dr, avg_optimized_dr, avg_reduction])
            table_list.append(pd.DataFrame(rows, columns=column_name))

        layer_df = pd.DataFrame(layer_ind * 3, columns = ["M"])

        table_df = pd.concat(table_list, axis=1)

        table_df = pd.concat([layer_df,table_df], axis=1)

    elif do_leverage_results == True:
        case="uniform"
        column_name = column_name*len(lev_policy_dict[30].keys())
        n_ind = [10, 20, 30] # , 20, 30
        table_rows = []
        for n in n_ind:
            weight_rows = []
            for weight_func in lev_policy_dict[n].keys():
                print("WORKING ON: N=", n, " M=", 1, "Weight: ", weight_func)
                avg_initial_dr, avg_optimized_dr, avg_reduction, std_init, std_opt, std_red = average_dr(
                    n,
                    1,
                    number_of_episodes,
                    LEVERAGE_EXP=do_leverage_results,
                    POLICY=lev_policy_dict[n][weight_func],
                    results="main"
                    )

                weight_rows.extend([avg_initial_dr, avg_optimized_dr, avg_reduction, std_init, std_opt, std_red])
            
            print("Completed row: ", weight_rows)
            table_rows.append(weight_rows)
        table_list = pd.DataFrame(table_rows, columns=column_name)

        n_df = pd.DataFrame(n_ind, columns = ["N"])

        table_df = pd.concat([n_df, table_list], axis=1)

    table_df.to_csv(out_path + "/table_3-" + str(do_leverage_results) + "_" + str(number_of_episodes) + "_" + case + ".csv")

    print(table_df)
