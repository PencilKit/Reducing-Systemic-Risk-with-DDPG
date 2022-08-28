
from multiprocessing.sharedctypes import Value
import os
from absl import logging
import tensorflow as tf
import numpy as np

# Import the safety layer
import safety_layer as sl

from tf_agents.environments import tf_py_environment

from tf_agents.metrics import tf_metrics

from tf_agents.eval import metric_utils

from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tf_agents.policies import random_tf_policy
from tf_agents.policies import gaussian_policy

from tf_agents.utils import common

from tf_agents.policies import policy_saver

from tf_agents.agents.ddpg import ddpg_agent, actor_network, critic_network

from datetime import datetime

# Custom environment
from environment import NetworkEnvironment
from network_gym.envs import model_network as network_model # Multi-layer methods

import importing_modules as im

import os

import cProfile
import pstats
import io

def train_and_evaluate_agent_ddpg(tf_agent, use_tf_function=False, name='agent'):

    train_summary_writer = tf.summary.create_file_writer(
        train_dir, flush_millis=10000
    )
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.summary.create_file_writer(
        eval_dir, flush_millis=10000
    )

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_policy = tf_agent.policy
    tf_policy_saver = policy_saver.PolicySaver(eval_policy)

    collect_policy = gaussian_policy.GaussianPolicy(
        tf_agent.collect_policy,
        scale=gaussian_std,
        clip=True
    )

    ### Initialize replay buffer.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=ddpg_replay_buffer_capacity
    )
    replay_observer = [replay_buffer.add_batch]

    replay_sequence_len = 2

    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(),
        train_env.action_spec()
    )

    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        initial_collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=ddpg_initial_collect_steps
    )

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        collect_policy ,
        observers=replay_observer + train_metrics,
        num_steps=collect_steps_per_iteration
    )

    if use_tf_function == True:
        initial_collect_driver.run = common.function(initial_collect_driver.run)
        collect_driver.run = common.function(collect_driver.run)
        tf_agent.train = common.function(tf_agent.train)

    # Collect initial replay buffer data.
    logging.info(
        "Initializing replay buffer by collecting experience for %d steps with a random policy",
        ddpg_initial_collect_steps
    )
    initial_collect_driver.run()

    tf_agent.train_step_counter.assign(0)

    results = metric_utils.eager_compute(
        eval_metrics,
        eval_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics'
    )
    if eval_metrics_callback is not None:
        eval_metrics_callback(results, global_step.numpy())
    metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = collect_policy.get_initial_state(train_env.batch_size)

    # Dataset generates trajectories
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=replay_sequence_len,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    iterator = iter(dataset)

    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train(experience)

    if use_tf_function:
        train_step = common.function(train_step)

    for i in range(ddpg_num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer
        for _ in range(collect_steps_per_iteration):
            # Runs the environment for however many steps.
            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state
            )

        # Sample a batch of data from the buffer and update the agent's network.
        for _ in range(train_steps_per_iteration):
            train_loss = train_step()

        if global_step.numpy() % ddpg_log_interval == 0:
            logging.info("step = %d, loss %5f", global_step.numpy(), train_loss.loss)

        for train_metric in train_metrics:
            train_metric.tf_summaries(train_step=global_step, step_metrics=train_metrics[:2])


        if global_step.numpy() % ddpg_eval_interval == 0:
            results = metric_utils.eager_compute(
                eval_metrics,
                eval_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix="Metrics"
            )
            if eval_metrics_callback is not None:
                eval_metrics_callback(results, global_step.numpy())
            metric_utils.log_metrics(eval_metrics)
            tf_policy_saver.save(EvalComplexNetworkEnvironment.result_logger.curr_policy_dir)


    return train_loss

if __name__ == "__main__":

    pr = cProfile.Profile()
    pr.enable()

    logging.set_verbosity(logging.DEBUG)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    np.random.seed(1)

    # DR
    N_NODES = 30
    NUM_LAYERS = 3


    NETWORK_SEED = [2, 2, 2] # For regular experiments
    # NETWORK_SEED = [3, 3, 3] #N=30 for leverage 
    NETWORK_SEED = [20,20,20] #
    BETA = 0.18
    LEVERAGE_EXP = False
    CASE = "original" # can be set to either "original" or "uniform"


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

    batch_size = 256


    ddpg_num_iterations = 15000
    ddpg_initial_collect_steps = int(0.05*ddpg_num_iterations) - 500
    ddpg_eval_interval = 100
    ddpg_replay_buffer_capacity = int(0.05*ddpg_num_iterations) - 500
    ddpg_log_interval = 20
    ddpg_num_iterations = 8000

    MAX_EPISODE_STEPS = 50

    collect_steps_per_iteration = 1
    train_steps_per_iteration = 1

    critic_learning_rate = 3e-4 
    actor_learning_rate = 3e-5
    target_update_tau = 0.001
    target_update_period = 1.0
    gamma = 0.8
    reward_scale_factor = 1.0  

    gaussian_std = 0.15

    gradient_clipping = None 

    num_eval_episodes = 1

    current = np.datetime64(datetime.now().replace(microsecond=0).isoformat(' '))

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
        gamma_h = 1.0
        GAMMA_NET = np.concatenate(
            (np.random.uniform(low=gamma_h - gamma_diff, high=gamma_h, size=half_N),
            np.random.uniform(low=gamma_l, high=gamma_l + gamma_diff, size=half_N))
            ) # new

        for eval_node, training_node, lev_val in zip(eval_complex_network.bank_list, training_complex_network.bank_list, GAMMA_NET):
            eval_node.leverage_ratio = lev_val
            training_node.leverage_ratio = lev_val

        print("Done leverage assignment")


    train_dir, eval_dir = ResultLogger.make_tensorboard_dir(str(training_complex_network.init_debtrank.sum()))
    
    import networkx.algorithms.isomorphism as iso
    import networkx as nx
    em = iso.numerical_edge_match('loans', 1)
    debtrank_str = str(eval_complex_network.init_debtrank.sum())
    with open(os.path.join(root_str,"network_py_data", "initial_network")  + "\\" + debtrank_str + '_network.npy', 'wb') as f:
        np.savez(
            f,
            network=eval_complex_network.multi_adj,
            debtrank=eval_complex_network.init_debtrank,
            networth=[bank_node.net_worth for bank_node in  eval_complex_network.bank_list],
            creditrisk=[bank_node.leverage_ratio for bank_node in  eval_complex_network.bank_list],
            c_eps=eval_complex_network.c_eps,
            parameters=parameters
            )

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
    eval_env = tf_py_environment.TFPyEnvironment(EvalComplexNetworkEnvironment)

    layer_param_num = 256
    actor_fc_layer_params = (layer_param_num, layer_param_num, layer_param_num)
    critic_joint_fc_layer_params = (layer_param_num, layer_param_num, layer_param_num)

    OTHER_ASSETS = np.array([node.other_assets for node in TrainingComplexNetworkEnvironment._complex_network.bank_list])

    eval_metrics_callback = None

    from tf_agents.agents.ddpg import ddpg_agent, actor_network, critic_network
    from tf_agents.train.utils import strategy_utils

    use_gpu = True
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

    with strategy.scope():
        actor_net = actor_network.ActorNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=actor_fc_layer_params)

        critic_net = critic_network.CriticNetwork(
            (train_env.observation_spec(),  train_env.action_spec()),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params)

        global_step = tf.compat.v2.Variable(0, dtype=tf.int64)
        tf_agent = ddpg_agent.DdpgAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,  
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            train_step_counter=global_step,
            ou_stddev=0.0,
            ou_damping=0.0
            )
        tf_agent.initialize()

    returns = train_and_evaluate_agent_ddpg(tf_agent, use_tf_function=False, name='ddpg')

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('profiler_results.txt', 'w+') as f:
        f.write(s.getvalue())
