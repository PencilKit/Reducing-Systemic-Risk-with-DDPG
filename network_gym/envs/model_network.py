import networkx as nx
import numpy as np
from scipy.linalg import null_space


class Bank(object):
    """
    The bank's balance sheet:
    
    Assets: Cash, Other Assets, Loans
    Liabilities & Equity: Borrowing, Deposits

    """

    def __init__(self, node_id, beta, gamma, theta, total_assets, total_adj):
        # Bank parameters
        self.node_id = node_id
        self.beta = beta
        self.theta = theta
        self.total_adj = np.array(total_adj)
        if len(gamma) > 1:
            self.gamma = gamma[self.node_id]
        else:
            self.gamma = gamma[0]
    
        # Bank balance sheet calculations
        total_weight_mat = np.sum(self.total_adj, axis=0)

        # The total amount for lending
        self.total_network_lending = np.sum(total_weight_mat)

        # m_loan total loans for each layer for each node.
        self.m_loan = self.total_adj[:, self.node_id].sum(axis=1)

        # self.loans are the total loans across all layers.
        self.loans = total_weight_mat.sum(axis=1) # This is the  loans for all banks
        self.borrowings = total_weight_mat.sum(axis=0)

        self.loan = self.loans[self.node_id]
        self.borrowing = self.borrowings[self.node_id]

        self.other_assets = self.calculate_other_assets(total_assets)
        self.net_worth = self.calculate_net_worth()
        self.deposits = self.calculate_deposits()
        self.cash = self.calculate_cash()

        # Calculate the credit risk. We use the leverage ratio as a proxy for credit risk.
        self.total_assets = self.cash + self.loan + self.other_assets
        self.total_liabilities = self.deposits + self.borrowing
        self.leverage_ratio = self.total_liabilities / self.total_assets

        # States used to calculuate the DebtRank
        self.distress = 0 # Initially 
        self.state = 'U' # initially undistressed?

    def calculate_cash(self):
        numer = ( self.beta*(1-self.gamma)*(self.loan + self.other_assets) -
                 self.beta*self.borrowing )

        denom = 1 + self.beta*self.gamma - self.beta

        return numer/denom

    def calculate_deposits(self):
        numer = ( (1-self.gamma)*(self.loan + self.other_assets) - 
                 self.borrowing )

        denom = 1 + self.beta*self.gamma - self.beta

        return numer/denom

    def calculate_net_worth(self):
        numer = ( self.gamma*(self.loan + self.other_assets) - 
                 self.beta*self.gamma*self.borrowing )
        
        denom = 1 + self.beta*self.gamma - self.beta

        return numer/denom

    def calculate_other_assets(self, total_assets):
        first = np.maximum(self.borrowing - self.loan, 0)

        term1 = ( (1-self.theta) - self.beta*(1-self.theta) + 
            self.beta*self.theta ) * total_assets
        term2 = np.sum(np.maximum(self.borrowings - self.loans,0))

        second = (term1 - term2)*self.loan/self.total_network_lending
        
        other_assets = first + second

        return other_assets


class Multigraph(object):
    """ 
    Made using the methods described in 
    Li2019: https://doi.org/10.1016/j.frl.2019.07.005
    Maeno2012: 	10.1109/CIFEr.2013.6611695
    """

    def __init__(self, N_nodes, num_layers, total_assets, thetas, beta, gamma, 
                 r, c_eps, rew_lambda, rew_rho, network_importer=None, is_eval=False):

        self.N_nodes = N_nodes
        self.num_layers = num_layers

        self.total_assets = total_assets
        self.beta = beta
        self.gamma = gamma
        self.thetas = thetas # Enter as a list of theta values per layer. Used in calculating balance sheet
        self.r = r
        self.is_eval = is_eval

        self.network_importer = network_importer

        self.G = {} # The set G containing the graphs of different layers.

        self.rew_lambda = rew_lambda
        self.rew_rho = rew_rho

        # Creating the interbank lending network for each layer.
        self.multi_adj = []
        for alpha in range(self.num_layers):
            self.G[alpha] = self.create_layer(alpha, network_importer=network_importer)
            self.multi_adj.append(nx.to_numpy_array(self.G[alpha], weight='loans'))

        self.multi_adj = np.array(self.multi_adj)


        self.initial_multi_adj = self.multi_adj.copy()
        self.initial_multi_adj.setflags(write=False) # make unwritable...

        # These are the initial lending amounts and constant
        self.m_total_loan = np.sum(self.multi_adj, axis=2)
        self.m_total_loan.setflags(write=False)
        self.m_total_borrowing = np.sum(self.multi_adj, axis=1)
        self.m_total_borrowing.setflags(write=False)

        # Add the balance sheet information of each node. 
        self.set_balance_sheet()
        self.update()

        self.init_debtrank = self.calculate_multilayer_debtrank()
        self.init_debtrank.setflags(write=False) # make unwritable...
        self.curr_debtrank = self.init_debtrank.copy()
        self.prev_debtrank = self.init_debtrank.copy()

        # To check that the difference after borrowing and lending
        if not hasattr(self, 'c_eps') == None:
            # set the constraint as the
            self.c_eps = np.zeros(self.N_nodes)
            for n, nbank in enumerate(self.bank_list):
                self.c_eps[n] = nbank.other_assets 
        elif len(c_eps) > 0:
            self.c_eps = c_eps
        else:
            raise ValueError("c_eps value error.")

        # Calculate the basis vector for the given number of nodes. This is used in the creating the action of the agent.
        self._calculate_basis_vectors() 

    def set_balance_sheet(self):
        """ Set the balance sheet as the node attribute using the key 'bs'.
        The first layer is the default layer whose nodes contain the
        attributes.
        """

        # Set the node attribute of node_id with key 'bs' as the balance
        # sheet. The key value is a class Bank whose attributes are the
        # balance sheet values.
        self.bank_list = []
        for node_id in self.G[0].nodes:

            self.G[0].nodes[node_id]['bs'] = Bank(
                node_id,
                self.beta,
                self.gamma,
                np.sum(self.thetas),
                self.total_assets,
                self.multi_adj
            )
            self.bank_list.append(self.G[0].nodes[node_id]['bs']) # used in MADDPG?

        return self

    def create_layer(self, alpha, network_importer=None, network_array=None):
        """ We create the layers of the network depending on if we are importing
        the network from an external source or not. """

        if network_importer is None and network_array is None:

            # Default
            # sfparam = [0.41, 0.54, 0.05]
            # delta = [0.2, 0.0]

            sfparam = [0.15, 0.6, 0.25] 
            delta = [0.2, 0.2]

            graph = nx.scale_free_graph(
                self.N_nodes,
                alpha=sfparam[0],
                beta=sfparam[1],
                gamma=sfparam[2],
                delta_in=delta[0],
                delta_out=delta[1],
                seed=self.seed[alpha]
            )
            graph = nx.DiGraph(graph)
            graph.remove_edges_from(list(nx.selfloop_edges(graph)))

            A = nx.to_numpy_array(graph)
            graph = nx.from_numpy_matrix(
                A,
                parallel_edges=False,
                create_using=nx.DiGraph
            )

            adj_mat = nx.to_numpy_array(graph)

            # Need to get the loans and borrowings of all edges before creating the
            # the balance sheet.
            for (source, target, weight) in graph.edges(data=True):
                # For the edges incident on the node return a tuple
                # (source, target, weight)
                
                weight.clear()

                weight['loans'] = self.calculate_loans(
                    source,
                    target,
                    adj_mat,
                    graph,
                    self.thetas[alpha]
                )

        elif network_importer is not None:
            # Here we are importing one of the networks that was generated using the R package.

            custom_network = network_importer.liability_network[alpha].copy()
            custom_network[custom_network > 0] = 1
            graph = nx.convert_matrix.from_numpy_array(
                custom_network,
                parallel_edges=False,
                create_using=nx.DiGraph
                )
            graph = nx.complete_graph(self.N_nodes, nx.DiGraph())

            adj_mat = nx.to_numpy_array(graph)

            # Need to get the loans and borrowings of all edges before creating the
            # the balance sheet.
            for (source, target, weight) in graph.edges(data=True):
                # For the edges incident on the node return a tuple
                # (source, target, weight)
                
                weight.clear()

                weight['loans'] = network_importer.liability_network[alpha][source, target].copy()

        elif network_array is not None:
            # This means you have a custom network that doesn't use the network importer module
            custom_network = network_array[alpha].copy()
            custom_network[custom_network > 0] = 1
            graph = nx.convert_matrix.from_numpy_array(
                custom_network,
                parallel_edges=False,
                create_using=nx.DiGraph
                )
            graph = nx.complete_graph(self.N_nodes, nx.DiGraph())

            adj_mat = nx.to_numpy_array(graph)

            # Need to get the loans and borrowings of all edges before creating the
            # the balance sheet.
            for (source, target, weight) in graph.edges(data=True):
                # For the edges incident on the node return a tuple
                # (source, target, weight)
                
                weight.clear()

                weight['loans'] = network_array[alpha][source, target].copy()

        return graph

    def calculate_loans(self, source, target, adj_mat, graph, theta):
        """ Calculate the loan ammount """

        out_degrees = np.squeeze(adj_mat.sum(axis=1))
        in_degrees = np.squeeze(adj_mat.sum(axis=0))
        out_in_degree_mat = np.outer(out_degrees, in_degrees) ** self.r

        # get the in degree and out degree
        out_deg = graph.out_degree(source)
        in_deg = graph.in_degree(target)

        numerator = adj_mat[source, target] * (out_deg * in_deg) ** self.r
        denominator = np.sum(np.multiply(adj_mat, out_in_degree_mat))

        loan = numerator * theta * self.total_assets / denominator
        return loan

    def calculate_distress(self, node_id, impact_mat, state_status):
        self.calculate_next_distress(impact_mat)
        self.calculate_next_state(state_status)


    def run_dynamics(self, impact_mat):
        """ Run the dynamics to calculaute the DebtRank for a single distressed
        node. Run the algorithm until all nodes are in either states U or I. 
        """

        state_status = []
        for node_id in self.G[0].nodes():
            state_status.append(self.G[0].nodes[node_id]['bs'].state)

        distressed = np.zeros((len(self.G[0]), ))
        prev_status = state_status.copy()
        while 'D' in state_status:
            # While there's atleast one bank in distress run the contagion
            # dynamics. Calculuate the new distress for all nodes in
            # self.G[0].nodes()

            self.calculate_next_distress(impact_mat)
            self.calculate_next_state(state_status)

            if np.all(state_status == 'U'): print("ERROR there's all U state")
            
            if prev_status == state_status: print("Check while loop for infinite loop")
            prev_status = state_status.copy()

        for node_id in self.G[0].nodes():
            distressed[node_id] = self.G[0].nodes[node_id]['bs'].distress
        return distressed

    def calculate_next_state(self, state_status):
        """ To calculate the next state of the DR algorithm """

        for node_id in self.G[0].nodes():
            prev_state = self.G[0].nodes[node_id]['bs'].state
            curr_distress = self.G[0].nodes[node_id]['bs'].distress

            if prev_state == 'D':
                curr_state = 'I'
                self.G[0].nodes[node_id]['bs'].state = curr_state
                state_status[node_id] = self.G[0].nodes[node_id]['bs'].state

            elif curr_distress > 0 and prev_state != 'I':
                curr_state = 'D'
                self.G[0].nodes[node_id]['bs'].state = curr_state
                state_status[node_id] = self.G[0].nodes[node_id]['bs'].state
            else:
                state_status[node_id] = self.G[0].nodes[node_id]['bs'].state


    def calculate_next_distress(self, impact_mat):
        """ Calculate the next distress level with initial node_id as the 
        node that is distressed. """

        current_h = np.zeros((len(self.G[0]),))
        for node_id in self.G[0].nodes():
            # current_h also acts as an indicator as it'll be zero wherever
            # the if statement is not true.
            if self.G[0].nodes[node_id]['bs'].state == 'D':
                current_h[node_id] = self.G[0].nodes[node_id]['bs'].distress

        for node_id in self.G[0].nodes():
            node_h = self.G[0].nodes[node_id]['bs'].distress

            next_h = np.minimum(
                1,
                node_h + np.sum(impact_mat[:, node_id]*current_h)
            )

            # Set the next step's distress for node with node_id 
            self.G[0].nodes[node_id]['bs'].distress = next_h

    def reset_nodes(self):
        # """ Resets the dynamic states to an undistressed network """

        for node_id in self.G[0].nodes():
            self.G[0].nodes[node_id]['bs'].distress = 0
            self.G[0].nodes[node_id]['bs'].state = 'U'


    def calculate_multilayer_debtrank(self):
        """ Calculate the multilayer debtrank where contagion is spread between the layers. """

        # Calculate the impact matrix.
        net_worths = np.zeros((len(self.G[0],)))
        for node_id in self.G[0].nodes():
            net_worths[node_id] = self.G[0].nodes[node_id]['bs'].net_worth

        # Run dynamics
        alpha_debtrank = np.zeros((self.num_layers, len(self.G[0] )))
        for dis_node in self.G[0].nodes():
            previous_distress = [] # previous layer's distress. we append the distress from last layer
            self.reset_nodes()
            # Begin by calculating the distress levels of the first most lasyer.
            for alpha in range(self.num_layers):
                # For every distressed node we will calculate the debt rank. This
                # involves running the dynamics for all other nodes for every layer.

                # Calculuate the relative economic value for all nodes in the layer.
                relative_val = np.sum(self.multi_adj[alpha], axis=1) / np.sum(self.multi_adj[alpha])

                if alpha == 0:
                    impact_mat = np.minimum(1, (self.multi_adj[alpha]/np.expand_dims(net_worths, axis=1)).T)
                else:
                    # If you're not in the first layer use the other impact function
                    impact_denom = np.maximum(
                        self.multi_adj[alpha].T,
                        net_worths -  np.sum([(self.multi_adj[alpha] * h_dis).sum(axis=1) for h_dis in previous_distress], axis=0)
                    )

                    impact_denom[impact_denom == 0] = 1.0

                    impact_mat = self.multi_adj[alpha].T / impact_denom

                # Set initial distress for this layer:
                if alpha == 0:
                    self.G[0].nodes[dis_node]['bs'].distress = 1
                    self.G[0].nodes[dis_node]['bs'].state = 'D'
                    initial_dis = np.array([self.G[0].nodes[nn]['bs'].distress for nn in self.G[0].nodes()])
                else:
                    for n in self.G[0].nodes():
                        if distressed[n] > 0:
                            # Need to turn all inactive nodes into D nodes cuz that means they were distressed
                            self.G[0].nodes[n]['bs'].distress = distressed[n]
                            self.G[0].nodes[n]['bs'].state = 'D'
                        else:
                            # Test without this later
                            self.G[0].nodes[n]['bs'].distress = distressed[n]
                            self.G[0].nodes[n]['bs'].state = "U"
                    initial_dis = np.array([self.G[0].nodes[nn]['bs'].distress for nn in self.G[0].nodes()])

                # Run the dynamics
                distressed = self.run_dynamics(impact_mat)

                # Add the DRs to each row's DR list
                if alpha == 0:
                    alpha_debtrank[alpha, dis_node] = (np.sum(distressed * relative_val) - 
                        np.sum(initial_dis*relative_val))
                else:
                    alpha_debtrank[alpha, dis_node] = np.sum(distressed * relative_val) 

                # contains all the level of distress for all nodes under distress due to
                # node dis_node.
                previous_distress.append(distressed)

        # Once all the debtrank for each node for each layer is calculated we do the averaging.
        debt_rank = alpha_debtrank

        return debt_rank

    def update(self):
 
        # udpate the in, out, and debtrank of the network.
        in_degree = []
        out_degree = []
        # debt_rank = np.zeros((self.num_layers, self.N_nodes))
        for alpha in range(self.num_layers):
            # Get a list of incoming and outgoing degrees for each layer
            in_degree.append([degree for _, degree in self.G[alpha].in_degree])
            out_degree.append([degree for _, degree in self.G[alpha].out_degree])

        self.m_in_degree = in_degree
        self.m_out_degree = out_degree

        # Get normalized liability matrix
        self.get_normalized_lability_mat(np.array(self.multi_adj))

        return self

    def _world_step(self, action_n, safe_sample=False):
        # Here we clear the current graph edges of the network and rebuild with
        # the network.
        action_step_size = 1.0
        # Ensures any small negative errors are removed so they don't propagate
        next_multi_adj = np.maximum(self.multi_adj + action_step_size*action_n, 0)
        
        self.multi_adj = next_multi_adj.copy()
        for alpha in range(self.num_layers):
            # Update the current layer network configuration
            self._update_network_properties(self.multi_adj[alpha], alpha)

        # update the other attributes of the network i.e. the degrees, multi_adj 
        self.update()


    def _update_network_properties(self, adj_mat, alpha):
        """ Update the current layer properties using the new adj_mat
        """
    
        # This creates a temporary graph for layer alpha
        temp_graph = nx.DiGraph(adj_mat)
        temp_adj = nx.to_dict_of_dicts(temp_graph)

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
        self.G[alpha].update(edges=edge_data)

    def _calculate_basis_vectors(self):
        """ Calculate the basis to find \Delta L. This needs to only be done once as we use the same
        basis for all layers and """

        row_sum_arr = []
        col_sum_arr = []
        for i in range(self.N_nodes):
            row_sum_con = np.zeros((self.N_nodes, self.N_nodes-1))
            row_sum_con[i] = np.ones(self.N_nodes-1)

            if not (i == self.N_nodes-1):
                borr_sum_con = np.roll(np.identity(self.N_nodes), shift=i, axis=0)
                col_sum_arr.append(borr_sum_con)

            row_sum_arr.append(row_sum_con)
            
        row_sum_arr = np.concatenate(row_sum_arr, axis=1)
        col_sum_arr = np.concatenate(col_sum_arr, axis=1)

        constraint_matrix =  np.vstack([row_sum_arr, col_sum_arr])

        self.basis = np.array(null_space(constraint_matrix))
        self.null_space_dim = self.basis.shape[1]
        
        return self.basis

    def _sum_debtranks(self, debtrank):
        alpha_relative_value = np.array([self.multi_adj[m].sum() for m in range(self.num_layers)])/self.multi_adj.sum()
        relative_dr = debtrank * np.expand_dims(alpha_relative_value, axis=1)
        return relative_dr

    def _calculate_debtrank_difference(self, multi_debtrank=True, borrowing_constraint=False):

        def exp_weight(x, weight):
            return np.exp(x*weight)

        if multi_debtrank == True and borrowing_constraint == False:
            """ Here we just simply sum up the debtrank of the individual nodes so 
            an overall decrease would be an overall decrease in the sums

            remember, a decrease is good i.e. \delta < 0

            NOTE: To change the weights uncomment/comment the weights you want to use
            """
            # Note: Takes long here
            new_debtrank = self.calculate_multilayer_debtrank()
            self.curr_debtrank = new_debtrank

            weights = 1.0
            # weights = np.array([node.leverage_ratio for node in self.bank_list])
            # k = 10.0
            # weights = exp_weight(np.array([node.leverage_ratio for node in self.bank_list]), k)

            # rnew_debtrank = self._sum_debtranks(new_debtrank)[0][:int(self.N_nodes/2)]
            # rprev_debtrank = self._sum_debtranks(self.prev_debtrank)[0][:int(self.N_nodes/2)]

            rnew_debtrank = self._sum_debtranks(new_debtrank)
            rprev_debtrank = self._sum_debtranks(self.prev_debtrank)

            lambda_rew = 1.0
            debtrank_diff = np.maximum(1 - lambda_rew*((rnew_debtrank*weights).sum()/(rprev_debtrank*weights).sum()), 0) # new relative
            rew = self.rew_lambda * debtrank_diff

        else:
            raise ValueError("ERROR wrong param values")

        self._reward = rew

    def reward(self):
        return self._reward

    def reset(self):

        if self.is_eval == True:
            self.multi_adj = self.initial_multi_adj.copy()
            
            # Update other properties
            for alpha in range(self.num_layers):
                # Update the current layer network configuration
                self._update_network_properties(self.multi_adj[alpha], alpha)

            self.curr_debtrank = self.init_debtrank.copy()
            self.prev_debtrank = self.init_debtrank.copy()

        elif self.is_eval == False:
            self.multi_adj = self.network_importer.sample_liability_network()
                
            # Update other properties
            for alpha in range(self.num_layers):
                # Update the current layer network configuration
                self._update_network_properties(self.multi_adj[alpha], alpha)

            self.init_debtrank = self.calculate_multilayer_debtrank()
            self.init_debtrank.setflags(write=False) # make unwritable...
            self.curr_debtrank = self.init_debtrank.copy()
            self.prev_debtrank = self.init_debtrank.copy()
        else:
            raise ValueError("The Value of self.is_eval is incorrect")

        # update the other attributes of the network i.e. the degrees, multi_adj 
        self.update()

    def _check_debtrank_is_lower(self):
        current_debtrank = self.curr_debtrank.sum()
        previous_debtrank = self.prev_debtrank.sum()

        if current_debtrank < previous_debtrank:
            self.prev_debtrank = self.curr_debtrank.copy()
            return True
        elif current_debtrank >= previous_debtrank:
            return False
        else:
            raise ValueError("Problem with checking debtrank is lower.")

    def get_normalized_lability_mat(self, mat):

        x_value = np.array(mat)
        norm_x_value = []
        for alpha in range(self.num_layers):
            max_loans = self.total_assets * self.thetas[alpha]
            norm_x_value.append(x_value[alpha]/max_loans)
            
        self.normalized_liability_mat = np.array(norm_x_value)

        return self

    def _get_obs(self):
        return self.normalized_liability_mat.flatten()

    def _get_done(self, safe_sample=False):

        debtrank_is_good_check = self._check_debtrank_is_lower()

        if debtrank_is_good_check:
            self._done = False
        else:
            self._done = True

        return self._done


class LiteBank(object):
    """ For quick prototyping """
    def __init__(self, networth):
        self.net_worth = networth
        # States used to calculuate the DebtRank

        self.distress = 0 # Initially 
        self.state = 'U' # initially undistressed?

class LiteMultigraph(Multigraph):
    """Used to quickly calculate the DR """
    def __init__(self, network, networth):
        # Creating the interbank lending network for each layer.
        self.num_layers = network.shape[0]
        self.N_nodes = network.shape[1]
        self.multi_adj = []
        self.G = {} # The set G containing the graphs of different layers.
        for alpha in range(self.num_layers):
            self.G[alpha] = self.create_layer(alpha, network_array=network)
            self.multi_adj.append(nx.to_numpy_array(self.G[alpha], weight='loans'))

        # Set the node attribute of node_id with key 'bs' as the balance
        # sheet. The key value is a class Bank whose attributes are the
        # balance sheet values.
        self.bank_list = []
        for node_id in self.G[0].nodes:

            self.G[0].nodes[node_id]['bs'] = LiteBank(
                networth=networth[node_id]
            )
            self.bank_list.append(self.G[0].nodes[node_id]['bs'])
