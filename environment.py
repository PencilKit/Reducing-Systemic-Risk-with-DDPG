import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class NetworkEnvironment(py_environment.PyEnvironment):
    
    def __init__(self, complex_network, result_logger, max_episode_steps, action_scale=1.0):

        # Basis action scheme
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(complex_network.num_layers, complex_network.null_space_dim,), dtype=np.float64, minimum=-1.0, maximum=1.0, name='action'
        )

        self._action_modifier = None
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(complex_network.num_layers*(complex_network.N_nodes ** 2),), dtype=np.float64, minimum=0, maximum=None, name='observation'
        )
        self._action_scale = action_scale
        self._episode_ended = False
        self._episode_step_counter = 0
        self._max_episode_steps = max_episode_steps
        self._complex_network = complex_network
        self.result_logger = result_logger
        self.result_logger.make_log_and_print_dir(complex_network.init_debtrank.sum())
        self.result_logger.make_policy_saver_dir(complex_network.init_debtrank.sum())


        self.constraint_violation_counter = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._episode_step_counter = 0
        self._complex_network.reset()

        self.constraint_violation_counter = 0

        return ts.restart(self._complex_network._get_obs())

    def add_safety_layer(self, safety_layer):
        # Function to call 
        self._action_modifier = safety_layer.get_safe_action_cvxpy_delta_direct_basis

    def _step(self, action):

        def combine_tri(action, env=None, flat_array=None):
            lower_tri = np.concatenate((np.tril(action, -1), np.zeros((action.shape[0], 1))), axis=1)
            upper_tri = np.concatenate((np.zeros((action.shape[0], 1)), np.triu(action)), axis=1)

            return lower_tri + upper_tri

        # If we are done, reset the network
        if self._episode_ended:
            return self.reset()

        action_h = action * self._action_scale

        if self._action_modifier: # not none?
  
            action_h = self._action_modifier(
                self._complex_network.multi_adj,
                action_h
                )

            action_h = self._complex_network.basis @ action_h
            action_h = action_h.reshape((self._complex_network.num_layers, self._complex_network.N_nodes, self._complex_network.N_nodes-1))

        actor_action = []
        for alpha in range(self._complex_network.num_layers):
            actor_action.append(combine_tri(action_h[alpha]))
        actor_action = np.array(actor_action)


        self._complex_network._world_step(actor_action)
        self._episode_step_counter += 1
    
        self._complex_network._calculate_debtrank_difference(
            multi_debtrank=True
        )

        # The environment response is dependent on the debtrank difference.
        reward = self._complex_network.reward()

        self._episode_ended = self._complex_network._get_done()

        if self._episode_step_counter == self._max_episode_steps:
            self._episode_ended = True # Don't delete this, it resets the environment.
            if self._complex_network.is_eval == True:
                self.result_logger.print_debtrank(self._complex_network)
                self.result_logger.update_policy_path(self._complex_network)
            return ts.termination(
                observation=self._complex_network._get_obs(),
                reward=reward
                )
        elif self._episode_ended == True:
            if self._complex_network.is_eval == True:
                self.result_logger.print_debtrank(self._complex_network)
                self.result_logger.update_policy_path(self._complex_network)

            return ts.termination(
                observation=self._complex_network._get_obs(),
                reward=reward
            )
        elif self._episode_ended == False:
            return ts.transition(
                observation=self._complex_network._get_obs(),
                reward=reward
            )

        raise ValueError("Incorrect ActionResult response.")
