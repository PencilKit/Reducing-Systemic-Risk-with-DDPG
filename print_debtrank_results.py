import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from network_gym.envs import model_network as network_model
import os
import math

# Width for plots
fig_width = 397.48499

# Load the data
path_base = os.path.normpath(os.getcwd() + os.sep + os.pardir)
path_str = os.path.join(path_base, "data", "network_py_data", "to_print/")
ext_str = ".npy"
out_str = os.path.join(path_base, "data", "prints/")

re_calculate_DR = True

def calculate_DR(network, networth):
    temp_network = network_model.LiteMultigraph(network, networth)
    return temp_network.calculate_multilayer_debtrank()


# Uncomment to choose which you want to print 
# N=30, M=3
init_network_str = "33.295454885318456_network"
red_network_str = "19.583415955389167_network"

# N=30, M=1 # no weight leverage
init_network_str = "10.209733547617143_network"
red_network_str = "2.57661598537072_network"


initial_network = np.load(path_str + init_network_str + ext_str)
reduced_network = np.load(path_str + red_network_str + ext_str)

num_layers = initial_network['network'].shape[0]
nodes = initial_network['network'].shape[1]
print("PRINTING RESULTS FOR (N, M): (", str(nodes), ", ", str(num_layers), ")")


multi_layer_dr = True # THIS MEANS TO CALCULATE THE DR while weighting the layer value


path_str = path_str + str(nodes) + "-"


init_mat = initial_network['network']
red_mat = reduced_network['network']


loan_check = np.abs(init_mat.sum(axis=2) - red_mat.sum(axis=2))
borrowing_check = np.abs(init_mat.sum(axis=1) - red_mat.sum(axis=1))

if np.any(loan_check.sum(axis=0) + borrowing_check.sum(axis=0) > initial_network['c_eps']):
    raise ValueError("Difference between loan or borrowing too high.")

if re_calculate_DR == False:
    init_debtrank = initial_network['debtrank']
    red_debtrank = reduced_network['debtrank']
else:
    init_debtrank = calculate_DR(init_mat, initial_network['networth'])
    red_debtrank = calculate_DR(red_mat, reduced_network['networth'])
credit_list = initial_network['creditrisk']
x =  np.arange(nodes) + 1

width = 0.4 # width of bar
fontP = FontProperties()
fontP.set_size('small')

print("Plotting for datasets: ", init_network_str, " and ", red_network_str, "...")

if num_layers == 0:
    fig, ax = plt.subplots(num_layers, figsize=(8, 5))
else:
    fig, ax = plt.subplots(num_layers, figsize=(8, 8))

sort_ind = np.argsort(init_debtrank.sum(axis=0))[::-1]

credit_list = 1.0

ytitle_layers = ["$v^1 R_{i}(\mathbf{L}^{1}, \mathbf{e})$",
"$v^2 R_{i}(\mathbf{L}^{2}, \mathbf{e})$",
"$v^3 R_{i}(\mathbf{L}^{3}, \mathbf{e})$"]

init_total = 0
red_total = 0
if num_layers > 1:
    for m in range(num_layers):
        init_vals = init_debtrank[m][sort_ind]
        red_vals = red_debtrank[m][sort_ind]
        if multi_layer_dr == True:
            init_vals = (init_mat[m].sum() / init_mat.sum()) * init_vals
            red_vals = (red_mat[m].sum() / red_mat.sum()) * red_vals

            init_total += init_vals.sum()
            red_total += red_vals.sum()

            print("The initial DR for layer ", str(m), ":", init_vals.sum())
            print("The reduced DR for layer ", str(m), ":", red_vals.sum())

        # Set plot parameters
        ax[m].bar(x,  init_vals, -width, alpha=0.5, color='r', align="edge", label="initial")
        ax[m].bar(x + width, red_vals, -width, alpha = 0.5, color='b', align="edge", label="optimized")

        ax[m].set_ylabel(ytitle_layers[m])
        ax[m].set_yticks(np.arange(0, math.ceil(round(max(max(init_vals), max(red_vals)) * 10.0, 2)) / 10, step=0.1))
        ax[m].set_xticks(x, fontsize=5)
        ax[m].set_xticklabels(sort_ind+1)

    ax[0].legend(loc="upper right", prop=fontP)
    ax[-1].set_xlabel("Bank ID")

else:
    init_vals = init_debtrank[0][sort_ind]
    red_vals = red_debtrank[0][sort_ind]

    init_total = init_vals.sum()
    red_total = red_vals.sum()

    ax.bar(x,  init_vals, width, alpha=0.5, color='r', label="initial")
    ax.bar(x + width, red_vals, width, alpha = 0.5, color='b', label="optimized")
    ax.set_ylabel("$R_{i}(\mathbf{L}, \mathbf{e})$")
    # ax.set_yticklabels()
    ax.set_xticks(x)
    ax.set_xticklabels(sort_ind+1)
    
    ax.legend(loc="upper right", prop=fontP)
    ax.set_xlabel("Bank ID")

print("Initial Total DR: ", init_total)
print("Reduced Total DR:", red_total)
print("Percentage reduction ", (red_total - init_total)/init_total)
plt.tight_layout()
# plt.show()
plt.savefig(out_str + "debtrank_layer-" + str(num_layers+1) + "-" + red_network_str + ".svg")
