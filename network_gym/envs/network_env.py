import gym
import copy # for resetting our network object
import numpy as np

import numpy.matlib # for repmat function
from scipy.optimize import linprog # for generating training data

# import model_network as model
from network_gym.envs import model_network as model # The multi-network classes
from gym import error, spaces, utils
from gym.utils import seeding

class NetworkEnv(gym.Env):
    """ Make the network environment we will be using
    
    Need all the parameters to generate a multi-layer network.
    
    """

    def __init__(self, n_agents=100, m_layers=1, total_assets=100000,
                 theta=[0.2], beta=0.18, gamma=0.07, r=1, seed_network=[], test_status=False, render_view=False, algo='ddpg', complete=False):
        # m_layers=2
        # theta = [0.2, 0.25]

        # Multigraph parameters
        self.n_agents = n_agents
        self.m_layers = m_layers
        self.total_assets = total_assets
        self.theta = theta
        self.beta = beta
        self.gamma = gamma
        self.r = r
        self.algo=algo
        self.test_status = test_status
        self.debt_rank_difference = 0 # Keep track of the current difference
        self.seed_network = seed_network

        np.random.seed(1)
        self.gamma = np.random.uniform(low=0.01, high=0.99, size=self.n_agents)

        # the following string is for outputing/saving results
        self.out_str = r"C:\Users\richa\Google Drive\York\PhD Research\Systemic Risk and Contagion Effect\main_multi_systemic_risk/main/"

        # lin_prog attributes
        self.lin_B = [] # list of solutions for generating training data

        # Multigraph generation
        self.seed = []
        for _ in range(self.m_layers):
            _, seed = seeding.np_random() # sample a seed.
            self.seed.append(seed)
        
        self.seed = [10332646577102660931, 12274766005326253660, 17055971746499108055] # 0.56 0.66 0.8 DR21.75

        # self.seed = [1368136728076724917, 3358079919057940271, 1815898706741926387] # 0.36 0.56 0.8 DR17.68
        # self.seed = [15979176219349663194, 10779374275418390906, 728195288722530705] # 0.4 0.6 0.76 DR15.589
        # self.seed = [618363560750805410, 3168382882310773625, 13037267547241089125] # 0.46 0.6 0.66 DR12.89
    
        # self.seed = [12651355500842978437, 12834394066135242591] # 0.60 0.93 # N = 30 M = 2 [0.15, 0.35]
        # self.seed = [9228757970845416616, 12426184427081272126, 16240788194300949610] # 0.56, 0.66, 0.83 # N = 30
        # self.seed = [17478251406936763236, 1927580469113704002, 1815761972447937380] # 2.23 8.66 22.62 # N=50, Theta = [0.05, 0.15, 0.35]
        # self.seed = [12112488653495184781, 5608574975806238446, 11502308385113050134] # 0.6 0.8 0.86 N = 30
        # self.seed = [14697865829595467184, 9358211537346931304, 8713815957311266725] # 0.75 0.8 0.85 N = 20

        # self.seed = [13224213844712561583, 16547175490081550161] # 0.59, N=10, THETA = [0.25, 0.15]
        # self.seed = [6699789379182494805, 7395512043961769305] # 50 - 11.9 - A=100000 theta = [0.25, 0.25]
        # self.seed = [6699789379182494805, 7395512043961769305] #### 30 - 11.48  A = 1000000 [0.25, 0.25]
        # [1801630078800656849, 2805974891459011083] # 30 - 12.4 A=1000000 [0.25, 0.25]
        # self.seed = [15901563195528714188, 13513948128051099312] # 30 - 13.7 A=1000000 [0.25, 0.25]
        # self.seed = [14378573846673374268, 5969348592484235318]
        
        # self.seed = [5022604557666073483, 3159162357123426043]
        # seed = 16470104418303546103 # 0.8...2
        # self.seed = [9811873559313375142, 8864586363246305003] # N=10, DR=0.1, R=100, ALBERT_EDGE=1

        self.network = model.Multigraph(            
            self.n_agents,
            self.m_layers,
            self.total_assets,
            self.theta,
            self.beta,
            self.gamma,
            self.r,
            self.seed,
            complete=complete
            )


        # Why do you have it set up like this? Use the seed to generate original network structure
        # the seed network is a variation on this original structure. I.e. the balance sheet are defined by
        # the original seed and the seed network is the new variation on the topology to use.
        if len(seed_network) > 0:
            for m in range(self.m_layers):
                self._update_network_properties(self.seed_network[m], m)

            self.network.update()
            self.network.init_debtrank = self.network.calculate_multilayer_debt_rank()
            self.network.curr_debtrank = self.network.calculate_multilayer_debt_rank()

        self.curr_episode_debtrank = self.network.init_debtrank.sum()

        debtrank_str = str(self.curr_episode_debtrank)
        with open(self.out_str + "initial_network/" + debtrank_str + '_network.npy', 'wb') as f:
            np.savez(
                f,
                network=self.network.multi_adj,
                debtrank=self.network.init_debtrank,
                networth=[bank_node.net_worth for bank_node in  self.network.bank_list],
                creditrisk=[bank_node.leverage_ratio for bank_node in  self.network.bank_list]
                )

        # NOTE: Don't need this anymore if circumventing constraints
        # if c_eps == None:
        #     # set the constraint as the
        #     self.c_eps = np.zeros(self.n_agents)
        #     for n, nbank in enumerate(self.network.bank_list):
        #         self.c_eps[n] = nbank.cash + nbank.other_assets
        # else:
        #     self.c_eps = c_eps # The threshold for checking debtrank difference.


        self.network_original = copy.deepcopy(self.network)

        # # List of bank objects
        # self.agents = self.network.bank_list
        # self.time = 0

        if self.algo == 'ddpg':
            # action_space = spaces.Box(
            #     low=0,
            #     high=1,
            #     shape=(self.m_layers*(self.n_agents-1)*(self.n_agents-1),),
            #     dtype=np.float32
            # )

            # Here is just a single softmax output we use the make_experience() to generate actions
            action_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.m_layers*self.n_agents*self.n_agents,),
                dtype=np.float32
            )

        else:
            action_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.m_layers, self.n_agents),
                dtype=np.float32
            )
        self.action_space = action_space

        adj_mat_info = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.m_layers*self.n_agents*self.n_agents,),
            dtype=np.float32
        )

        obs_space = spaces.Tuple([
            adj_mat_info, # The action of the agent
        ])


        self.obs_space_length = len(obs_space)

        self.observation_space = obs_space

        # rendering can be done here?
        if render_view:
            pass


    def step(self, action_n):
        """

        # The action_n  is the matrix A whose row sum to equal 1 or 0 depending
        on whether there is a lending relationship present or not.

        """

        # okay so here you used the softmax function to get actions for each
        # action_n = np.concatenate((action_n[0], np.zeros((1, action_n.shape[2]))), axis=0)
        # d = action_n.shape[0]
        # assert action_n.shape[1] == d - 1
        # temp_mat = np.ndarray((d, d+1), dtype=action_n.dtype)
        # temp_mat[:, 0] = 0
        # temp_mat[:-1, 1:] = action_n.reshape((d-1, d))
        # action_n = temp_mat.reshape(-1)[:-d].reshape(1, d,d)

        # this stuff is just to make smaller than n arrays into approaiate actions
        #######################################################################

        obs_n = []
        # reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.network.bank_list # make it easier to reference

        # This function should set the edges or the previous one should
        self._world_step(action_n)
        
        # record the observation for each agent and calculates the reward
        # NOTE: The debtrank calculuated should only for valid network 
        # configuration
        if self.algo=='ddpg':
            obs_n = self._get_obs(action_n)

            reward = self._get_reward(action_n, True, True) # NOTE: Need to remove the last argument later...

            done_n = self._get_done()

            if done_n == 0.0:
                # soft reset for the next time we calculate the 
                self.soft_reset()

        else:
            for agent in self.agents:
                obs_n.append(self._get_obs(action_n))
                # Check no more decrease in DebtRank

            done_n = self._get_done()

            reward = self._get_reward(False)    

        return obs_n, reward, done_n, info_n

    def _world_step(self, action_n):
        # Here we clear the current graph edges of the network and rebuild with
        # the network.

        adj_mat = np.array(action_n) # make the action of each agent into an array.

        # borrowing_diff_m = [] # the borrowing difference
        # borrowing_candidate_m = [] # borrowing resulting from the actions
        # constraint_status_m = [] # the borrowing constraint status
        for m in range(self.m_layers):
            # For each layer we will step through the world and calculate the
            # constraint status of each layer.

            # The loan requirment of each bank
            m_total_loans = self.network.m_total_loan[m] 
            m_adj_mat = adj_mat[m]
            # m_adj_mat = adj_mat[m] * np.reshape(m_total_loans, (self.n_agents, 1))
            # adj_mat is now in terms of the dollar amount

            # NOTE: don't need to check borrowing constraint anymore.
            # borrowing_diff, borrowing_candidate, con_stat = self._constraint_check(
            #     adj_mat,
            #     self.network.m_total_borrowing[m]
            #     )

            # borrowing_diff_m.append(borrowing_diff)
            # borrowing_candidate_m.append(borrowing_candidate)
            # constraint_status_m.append(con_stat)

            # Update the current layer network configuration
            self._update_network_properties(m_adj_mat, m)

        # update the other attributes of the network i.e. the degrees
        self.network.update()
        
        # The penalty and constraint status for each layer of the network.
        # self.borrowing_diff_m = borrowing_diff_m
        # self.borrowing_candidate_m = borrowing_candidate_m # append this
        # self.constraint_status_m = constraint_status_m

    def _update_network_properties(self, adj_mat, alpha):
        """ Update the current layer properties using the new adj_mat
        """
    
        # This creates a temporary graph for layer alpha
        temp_graph = model.nx.DiGraph(adj_mat) # TODO: CHECK THAT THIS MAKES SENSE WITH THE OUTER FUNCTION
        # Need to modify the "weight" label in the dict of dicts
        temp_adj = model.nx.to_dict_of_dicts(temp_graph)

        # pre-process the edge_data to feed into updating the network graph.
        edge_data = []
        for n in temp_adj.keys():
            for m in temp_adj[n].keys():
                edge = (
                    n,
                    m,
                    {'loans': temp_adj[n][m].pop('weight')}
                )

                edge_data.append(edge)

        # update the original network with the new edge data
        self.network.G[alpha].update(edges=edge_data)



    # Set the action for a particular agent.
    def _set_action(self, action, agent, time=None):
        agent.action = action

    # Get the observation for a particular agent
    def _get_obs(self, action_n):
        # For now each node will be able to see all the other nodes including
        # itself. Therefore it will be able to see the actions of all other
        # banks
        # maybe let them see the degree distribution of other banks?
        # also see the total loans and borrowings of other banks.
        """ This function simply collects the observations """

        """
                        # complete observation
                    obs_space = spaces.Tuple([
                        network_obs, # The loan amounts to the entire network
                        bank_info, # num_ingoing degrees
                        bank_info # num_outgoing degrees
                        bank_info, # the correct max borrowings of entire networks
                        bank_inf, # the chosen borrowing
                    ])
        """

        # Normalize the observation... try one with and without normalizing.
        row_sum_b = np.sum(action_n, axis=2, keepdims=True)
        row_sum_b[row_sum_b==0] = 1.0

        action_n = action_n / np.broadcast_to(row_sum_b, (self.m_layers, self.n_agents, self.n_agents))


        adj_mat = np.array(action_n).flatten() # the actions of each layer for all agents

        if self.algo == 'ddpg':
            # for m in range(self.m_layers):
            #     m_total_loans = self.network.m_total_loan[m] 
            #     adj_mat[m] = adj_mat[m] * np.reshape(m_total_loans, (self.n_agents, 1))


            ################# NOTE: CUSTOM FOR SMALLER THAN N-1 matrices #############################
            # np.fill_diagonal(adj_mat[0], np.nan)
            # adj_mat = adj_mat[~np.isnan(adj_mat)].reshape(adj_mat.shape[1], adj_mat.shape[2]-1)
            # adj_mat = adj_mat[:adj_mat.shape[0]-1,:].reshape(1, adj_mat.shape[0]-1, adj_mat.shape[1])
            # adj_mat = adj_mat / adj_mat.sum(axis=2).reshape(4,1)
            ##########################################################################################

            #NOTE: Everytime you change this you need to change the observation_space
            # obs_list = [
            #     adj_mat, # the action of all nodes
            #     # self.network.m_total_borrowing/np.sum(self.network.m_total_borrowing), # borrowing constraint to satisfy
            #     np.array(self.borrowing_candidate_m)/np.sum(self.network.m_total_borrowing) # The borrowing that was chosen.
            # ]

            obs_list = [
                adj_mat # the action of all the nodes.
            ]

            # np.array(self.network.m_out_degree), # the out degree of the agents
            # np.array(self.network.m_in_degree), # the in degree of the agents

            obs_n = obs_list
            # obs_n = tuple(obs_list)
        else:
            print("ERROR")

        obs_n = self._obs_process(obs_n) # This processor just flattens every element of obs_n 

        return obs_n

    # Get the reward
    def _get_reward(self, comb_debtrank=False, multi_debtrank=True, test_debtrank=False):
        """ test_debtrank is any new changes you are testing. Otherwise go back to the old averaging of the debtrank... """

        # if np.any(np.array(self.borrowing_diff_m) >= self.c_eps):
        #     # Calculate the reward from the borrowing constraint
        #     # reward_n = self.borrowing_diff_m
        #     reward_n = self.borrowing_diff_m[0]/self.network.m_total_borrowing[0]



        # if np.all(self.constraint_status_m) and comb_debtrank == False and multi_debtrank == False:
        #     # if action constraints are satisfied in all layers
        #     debt_rank = np.zeros((self.m_layers, len(self.network.G[0])))
        #     for alpha in range(self.m_layers):
        #         debt_rank[alpha] = self.network.calculate_alpha_debtrank(alpha=alpha) # calculate the debtrank of each layer

        #     avg_debtrank = self.network.calculate_average_alpha_debt_rank(
        #         debt_rank,
        #         self.network.num_layers
        #     )

        #     # print("DEBTRANK WAS CALCULATED")
        #     # NEED TO ADD HYPER PARAMETERS ON THE FIRST TERM
        #     net_val = np.sum(self.network.m_total_loan)

        #     rew = []
        #     for alpha in range(self.network.num_layers):
        #         debtrank_diff = self.network.curr_debtrank[alpha] - avg_debtrank[alpha]
        #         self.network.curr_debtrank[alpha] = avg_debtrank[alpha] # set the current debtrank as this
        #         # borr_rew = reward_n[alpha]

        #         # seems like the current debtrank keeps changing after every reset? 
        #         # rew.append(self.lambda_dr * net_val * debtrank_diff)
            
        #         rew.append(self.lambda_dr * debtrank_diff/self.network.curr_debtrank[alpha])

        #         if self.curr_best_debtrank_achieved[alpha] > avg_debtrank[alpha]:
        #             self.curr_best_debtrank_achieved[alpha] = avg_debtrank[alpha]
        #         self.debt_rank_difference = debtrank_diff

        #     self.debtrank_calculated = True
        #     rew = sum(rew)

        if multi_debtrank == True and test_debtrank == True:
            """ here we just simply sum up the debtrank of the individual nodes so 
            an overall decrease would be an overall decrease in the sums

            remember, a decrease is good i.e. \delta < 0
            """

            new_debtrank = self.network.calculate_multilayer_debt_rank()
            self.network.curr_debtrank = new_debtrank.sum()
            self.curr_episode_debtrank = new_debtrank.sum() # saves this attribute in the environment object
           
            # weights = self.network.init_debtrank / self.network.init_debtrank.sum(axis=1)[:, None]
            # debtrank_diff = (new_debtrank - self.network.init_debtrank)*weights

            weights = np.array([node.leverage_ratio for node in self.network.bank_list])
            # weights = 1
            debtrank_diff = (new_debtrank*weights).sum(axis=1) - (self.network.init_debtrank*weights).sum(axis=1)
            # debtrank_diff = (new_debtrank*weights).sum() - (self.network.init_debtrank*weights).sum()
            # debtrank_diff = new_debtrank.sum(axis=1) - self.network.init_debtrank.sum(axis=1)

            if debtrank_diff.sum() > 0:
                debtrank_str = str(new_debtrank.sum())
                with open(self.out_str + "seed_network_samples/" + debtrank_str + '_network.npy', 'wb') as f:
                    np.savez(
                        f,
                        network=self.network.multi_adj,
                        debtrank=new_debtrank,
                        networth=[bank_node.net_worth for bank_node in  self.network.bank_list],
                        creditrisk=[bank_node.leverage_ratio for bank_node in  self.network.bank_list]
                    )

            if debtrank_diff.sum() < 0:
                # found good config
                if self.test_status == True:
                    # save the debtrank network if you are testing and you found a good 
                    debtrank_str = str(new_debtrank.sum())
                    with open(self.out_str + "result_network_dr/" + debtrank_str + '_network.npy', 'wb') as f:
                        # np.save(f, np.array([self.network.multi_adj, new_debtrank]))
                        np.savez(
                            f,
                            network=self.network.multi_adj,
                            debtrank=new_debtrank,
                            networth=[bank_node.net_worth for bank_node in  self.network.bank_list],
                            creditrisk=[bank_node.leverage_ratio for bank_node in  self.network.bank_list]
                        )

            rew = -debtrank_diff.sum() # need a negative sign here because of how we defined difference

        elif multi_debtrank == True and test_debtrank == False:

            new_debtrank = self.network.calculate_multilayer_debt_rank()
            self.network.curr_debtrank = new_debtrank
            self.curr_episode_debtrank = new_debtrank

            if new_debtrank > self.network.init_debtrank:
                debtrank_str = str(new_debtrank)
                with open(self.out_str + "seed_network_samples/" + debtrank_str + '_network.npy', 'wb') as f:
                    np.save(f, np.array(self.network.multi_adj))


            self.next_debtrank_good = new_debtrank < self.curr_best_debtrank_achieved

            if self.next_debtrank_good:
                if self.test_status == True:
                    # save the debtrank network if you are testing and you found a good 
                    debtrank_str = str(new_debtrank)
                    with open(self.out_str + "result_network/" + debtrank_str + '_network.npy', 'wb') as f:
                        np.save(f, np.array(self.network.multi_adj))


            rew = self.network.init_debtrank - new_debtrank

            #return the new debtrank every run as well so we can keep track of that?

        else:
            # else penalize for not following constraints
            raise ValueError("ERROR wrong param values")
            # rew = - self.rho * np.sum(reward_n)

            # # QUESTIONABLE HOW YOU SHOULD TREAT THE END OF THE EPISODE
            # if self.debtrank_calculated == True:
            #     self.debtrank_calculated = False
            #     rew = 0

        # return rew
        return -1.0

    def _get_constraint_values(self):
        
        return c

    def _get_done(self):
        """ This function checks whether we are done the episode? 
        
            For now we will be done when we reach the terminal episode, N.

        """

        # if np.all(self.constraint_status_m) == False:
        #     # If there is atleast one 
        #     return 1.0
        # else:
        #     # Obtained a valid configuration, continue.
        #     return 0.0

        # if self.next_debtrank_good == False:
        #     return 1.0
        # else:
        #     return 0.0

        return 0.0
        # NOTE: another idea is to be done if the action results in a worse DebtRank than the previous
        # set DebtRank. 
        # OR
        # end the episode if the new DebtRank is worse than the original DebtRank.
        # FOR now we just keep going until N.

            
    def _obs_process(self, obs):
        observation = np.array([])
        for ele in obs:
            observation = np.concatenate((observation, ele.flatten()))
        return np.array(observation)


    def reset(self, comb_debtrank=False, new_network=True):
        """ 
        Reset the environment
        """
        if new_network == True:
            " If this is true then we will clear the network of any changes but will still use the same initial network."


            # seed = 13755327573478333088 # 0.8 for n=5
            # self.seed = 16470104418303546103 # 0.8 for n=5
            
            # NOTE: we use the same seed when intitially creating the environment. 

            self.network = model.Multigraph(            
                self.n_agents,
                self.m_layers,
                self.total_assets,
                self.theta,
                self.beta,
                self.gamma,
                self.r,
                self.seed
                )

            # Creates a copy of the network n case
            self.network_original = copy.deepcopy(self.network)

            # List of bank objects
            self.agents = self.network.bank_list
            self.time = 0

            # Set the observation
            self.borrowing_candidate_m = self.network.m_total_borrowing

            # row_sum = np.sum(self.network.multi_adj, axis=2, keepdims=True)
            # row_sum[row_sum == 0] = 1.0
            # action = self.network.multi_adj / np.broadcast_to(row_sum, (self.m_layers, self.n_agents, self.n_agents))

            obs_n = self._get_obs(self.network.multi_adj)

            # obs_n = self._obs_process(obs_n) # NOTE: perhaps put this into the _get_obs function

            # Set the current best debtrank as the one generated by the initial generation. Currently resetting the network
            # so the current best dabtrank achieved should be the current debtrank.
            self.curr_episode_debtrank = self.network.init_debtrank.sum()
            self.debt_rank_difference = 0

            self.debtrank_calculated = False # new debt_rank has not been calculated yet 

            return obs_n

        else:
            # NOTE: This should be okay.
            self.network = copy.deepcopy(self.network_original)

            # List of bank objects
            self.agents = self.network.bank_list
            self.time = 0

            # Set the observation
            # self.borrowing_candidate_m = self.network.m_total_borrowing

            # row_sum = np.sum(self.network.multi_adj, axis=2, keepdims=True)
            # row_sum[row_sum == 0] = 1.0
            # action = self.network.multi_adj / np.broadcast_to(row_sum, (self.m_layers, self.n_agents, self.n_agents))

            obs_n = self._get_obs(np.array(self.network.multi_adj))

            # Set the current best debtrank as the one generated by the initial generation
            self.curr_episode_debtrank = self.network.init_debtrank.sum()
            self.debt_rank_difference = 0

            self.debtrank_calculated = False

            return obs_n

#############################################################################################################

    def soft_reset(self, comb_debtrank=False, new_network=True):
        """ This function resets for the episode step """

        # NOTE: This should be okay.
        self.network = copy.deepcopy(self.network_original)

        # List of bank objects
        self.agents = self.network.bank_list
        self.time = 0

        # Set the observation
        # self.borrowing_candidate_m = self.network.m_total_borrowing
        # obs_n = self._get_obs(self.network.multi_adj)

        # # Set the current best debtrank as the one generated by the initial generation
        # self.curr_best_debtrank_achieved = self.network.curr_debtrank
        # self.debt_rank_difference = 0


    def solve_linprog(self, alpha):
        row_total = self.network.m_total_loan[alpha]
        col_total = self.network.m_total_borrowing[alpha]

        m = np.size(row_total)
        n = np.size(col_total)

        col1 = np.matlib.repmat(np.arange(m), 1, n)[0] + 1
        col1 = np.concatenate((col1, np.repeat(np.arange(n)+m, m) + 1))
        col2 = np.matlib.repmat(np.arange(m*n), 1, 2)[0] + 1

        
        mat_pairs = np.array([col1, col2]).T

        # Aeq = np.zeros((n+m+1, n*m))
        # for i in range(n+m):
        #     for j in range(m*n):
        #         for k in range(m*n*2):
        #             if np.all(mat_pairs[k] == np.array([i+1, j+1])):
        #                 Aeq[i, j] += 1
    
        # Aeq[-1, :] = np.identity(n).reshape(n*m)

        identity_n = np.matlib.repmat(np.identity(n), 1, n)
        
        row_sum_arr = []
        for i in range(n):
            row_sum_con = np.zeros((n, n))
            row_sum_con[i] = np.ones(n)
            row_sum_arr.append(row_sum_con)
            # row_sum_arr = np.concatenate((row_sum_arr, row_sum_con)) 

        row_sum_arr = np.concatenate(row_sum_arr, axis=1)

        diag_con = np.identity(n).flatten()

        Aeq= np.vstack([identity_n, row_sum_arr, diag_con])


        # np.fill_diagonal(Aeq, 0)

        beq = np.concatenate((col_total, row_total))
        beq = np.concatenate((beq, np.array([0])))
        lb = 0
        ub = np.max(beq)
        
        B = np.zeros((m*n, m*n))
        for k in range(m*n):
            f = np.zeros(m*n)
            f[k] = -1
            B[:, k] = linprog(f, A_eq=Aeq, b_eq=beq, method='simplex', options={'rr': False})['x']
            # B[:, k] = linprog(f, A_eq=Aeq, b_eq=beq, method='simplex')['x']

        self.lin_B.append(B) # append the basic feasible solutions.

    def make_experience(self, x=None):
        # Make the actions from the convex combination for all layers.

        n = self.n_agents

        exp_mat = []
        for alpha in range(self.m_layers):
            exp_mat.append(np.matmul(self.lin_B[alpha], x[alpha]).reshape((n, n)))

        return exp_mat            

    def render(self):
        pass
