import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import os

# Load the data

path_base = os.path.normpath(os.getcwd() + os.sep + os.pardir)
path_str = os.path.join(path_base, "data", "network_py_data", "to_print/")
ext_str = ".npy"
out_str = os.path.join(path_base, "data", "prints/")


# NOTE: Don't forget to double check these
n_eps = 1
dr_eps = 1e-5

# # N=30, M=1 # no weight leverage
init_network_str = "10.209733547617143_network"
red_network_str = "2.57661598537072_network"

initial_network = np.load(path_str + init_network_str + ext_str)
reduced_network = np.load(path_str + red_network_str + ext_str)

num_layers = initial_network['network'].shape[0]
nodes = initial_network['network'].shape[1]
print("PRINTING RESULTS FOR (N, M): (", str(nodes), ", ", str(num_layers), ")")

if num_layers > 1:
    multi_layer_dr = True
else:
    multi_layer_dr = False

path_str = path_str + str(nodes) + "-"



network_str_list = ['initial', 'reduced']

mat_list = [initial_network['network'], reduced_network['network']]
debtrank_list = [initial_network['debtrank'], reduced_network['debtrank']] 
credit_list = [initial_network['creditrisk'], reduced_network['creditrisk']]

loan_check = np.abs(mat_list[0].sum(axis=2) - mat_list[1].sum(axis=2))
borrowing_check = np.abs(mat_list[0].sum(axis=1) - mat_list[1].sum(axis=1))

if np.any(loan_check.sum(axis=0) + borrowing_check.sum(axis=0) > initial_network['c_eps']):
    raise ValueError("Difference between loan or borrowing too high.")

print("Drawing for datasets: ", init_network_str, " and ", red_network_str, "...")

net_worth = initial_network['networth']

s3 = 16
s2 = 9
s1 = 5
shells = list(range(nodes))
shells = np.split(shells, [s1, s1+s2, s1+s2+s3])


gridspec = {'width_ratios': [1, 1, 0.05]}
if num_layers > 1:
    fig, axes = plt.subplots(nrows=num_layers, ncols=3, figsize=(8, 4), gridspec_kw=gridspec) # M=3
else:
    fig, axes = plt.subplots(nrows=num_layers, ncols=2, figsize=(8, 4))

ax = axes.flatten()
subplot_ind = 0

max_debtranks = []
for m in range(num_layers):
    max_debtranks.append(max(debtrank_list[0][m]))

max_debtrank = max(max_debtranks)

title_label = {
    "initial": "Initial",
    "reduced": "Optimized"
}

for m in range(num_layers):
    for weight_mats, debtranks, credit_score, network_str in zip(mat_list, debtrank_list, credit_list, network_str_list):

        weight_mat = weight_mats[m]
        weight_mat[weight_mat < n_eps] = 0.
        debtrank = debtranks[m]
        debtrank[debtrank < dr_eps] = 0.

        if multi_layer_dr == True:
            debtrank = debtrank * (weight_mat.sum() / weight_mats.sum())



        G = nx.from_numpy_array(weight_mat, create_using=nx.DiGraph)


        size_param = 100 # for single layer
        node_sizes = np.array(net_worth)/size_param

        # Create a list of node colors: proportional to debtrank
        node_colors_dr = debtranks[m]
        node_colors_cr = credit_score

        pos = nx.circular_layout(G)
        g_nodes_dr = nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=sorted(G.nodes),
            node_size=node_sizes,
            node_color=node_colors_dr,
            edgecolors='black',
            cmap=cm.jet,
            vmin=0.,
            vmax=max_debtrank,
            ax=ax[subplot_ind]
            )


        edges = nx.draw_networkx_edges(G, pos, ax=ax[subplot_ind], arrowstyle="->", alpha=0.2)

        if num_layers > 1:
            ax[subplot_ind].set_title(network_str + " layer-" + str(m+1), fontsize="small")
        else:
            ax[subplot_ind].set_title(title_label[network_str]  + " Network", fontsize="small")

        ax[subplot_ind].set_axis_off() # draw on different subplot
        if num_layers > 1 and network_str == "reduced":
            subplot_ind += 1
            cax = ax[subplot_ind]
            plt.colorbar(g_nodes_dr, ax=ax[subplot_ind], cax=cax)

        subplot_ind += 1

plt.tight_layout()
# plt.show()
if num_layers == 1:
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7]) # [0.85, 0.15, 0.05, 0.7]
    fig.colorbar(g_nodes_dr, cax=cbar_ax)

plt.show()
# plt.savefig(out_str + "/network_structure-"+red_network_str+".svg")
