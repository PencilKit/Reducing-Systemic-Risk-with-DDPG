from logging import exception
import numpy as np
import cvxpy as cp
import cplex

class SafetyLayer:
    """
    init_bound for previous implmentation was 0.003
    learning rate for the adam was 0.0001
    """
    def __init__(self, env):
        self._env = env
        self._eval_global_step = 0


    def get_safe_action_cvxpy_delta_direct_basis(self, multi_adj, action):
        # We need the basis matrix. 
        basis_matrix = self._env._complex_network.basis

        safe_action = cp.Variable(shape=(self._env._complex_network.null_space_dim, 1), name="safe_action")

        action_return = []
        for alpha in range(self._env._complex_network.num_layers):
            m_multi_adj = np.expand_dims(multi_adj[alpha][np.where(~np.eye(multi_adj[alpha].shape[0],dtype=bool))], axis=1)
            policy_action = np.expand_dims(action[alpha], axis=1)

            constraints = [m_multi_adj + basis_matrix @ safe_action >= 0]
            objective = cp.Minimize(0.5 * cp.sum_squares(basis_matrix @ safe_action - basis_matrix @ policy_action))
            problem = cp.Problem(objective, constraints)

            try:
                problem.solve(solver="CPLEX", warm_start=True)
            except:
                try:
                    problem.solve(solver="OSQP", warm_start=True)
                except:
                    problem.solve(solver="SCS", warm_start=True)
    
            action_return.append(safe_action.value)
    
        return np.array(action_return)
