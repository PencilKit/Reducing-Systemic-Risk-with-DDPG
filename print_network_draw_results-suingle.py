import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mplt
from matplotlib import cm

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load the data
path_str = r"C:\Users\richa\My Drive\York\PhD Research\Systemic Risk\data/network_py_data\to_print/" 
ext_str = ".npy"
out_str = r"C:\Users\richa\My Drive\York\PhD Research\Systemic Risk\data/prints/"  

# NOTE: Don't forget to double check these
num_layers = 3
n_eps = 1
dr_eps = 1e-5
nodes = 30
# path_str = path_str + "/"

# N=30, M=3
init_network_str = "33.295454885318456_network"
red_network_str = "19.583415955389167_network"

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

# Print the data as a histogram displaying the
# initial debtrank for each node and the final node of each bank.

network_str_list = ['initial', 'reduced']

mat_list = [initial_network['network'], reduced_network['network']]
debtrank_list = [initial_network['debtrank'], reduced_network['debtrank']] 
credit_list = [initial_network['creditrisk'], reduced_network['creditrisk']]


# Check you loaded the correct networks.
# loan_check = np.abs(mat_list[0].sum(axis=2) - mat_list[1].sum(axis=2))
# borrowing_check = np.abs(mat_list[0].sum(axis=1) - mat_list[1].sum(axis=1))
# if np.any(loan_check > 1.0) or np.any(borrowing_check > 1.0):
#     raise ValueError("Difference between loan or borrowing too high.")

# loan_check = np.abs(mat_list[0].sum(axis=2) - mat_list[1].sum(axis=2))
# borrowing_check = []
# for alpha in range(num_layers):
#     borrowing_check.append(np.abs(mat_list[0][alpha].sum(axis=0) - mat_list[1][alpha].sum(axis=0)))

# borrowing_check = np.array(borrowing_check)

# if np.any(loan_check > 1.0) or np.any(borrowing_check.sum(axis=0) > initial_network['c_eps']):
#     raise ValueError("Difference between loan or borrowing too high.")

loan_check = np.abs(mat_list[0].sum(axis=2) - mat_list[1].sum(axis=2))
borrowing_check = np.abs(mat_list[0].sum(axis=1) - mat_list[1].sum(axis=1))

if np.any(loan_check.sum(axis=0) + borrowing_check.sum(axis=0) > initial_network['c_eps']):
    raise ValueError("Difference between loan or borrowing too high.")

print("Drawing for datasets: ", init_network_str, " and ", red_network_str, "...")

net_worth = initial_network['networth']

18
11
5

s3 = 16
s2 = 9
s1 = 5
shells = list(range(nodes))
shells = np.split(shells, [s1, s1+s2, s1+s2+s3])
# 10
# net_worth = [1537.164625977962, 1171.8899314178889, 1985.5792661310359, 433.4844258947722, 559.7033788635473, 573.3399038883921, 505.69947669683296, 139.35748721382535, 376.91237143495005, 216.2686040157444]

# 20 complete
# net_worth = [3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368, 3825.36632236368]

# 30
# net_worth = [11933.726584392429, 3795.586443499747, 13761.22700114593, 9280.678286068403, 4115.920868980709, 3036.688754308228, 4369.030007869903, 6545.974324681261, 385.94205686076157, 2354.2278737008533, 4527.728017852595, 255.75002212920268, 317.84773336069594, 1198.4513262195164, 698.2963893125597, 1172.443604604721, 587.9154044508101, 582.1966036630238, 2141.386108220105, 320.1943600114012, 571.4492727209891, 282.341938074456, 307.5559556517176, 718.7390753172293, 599.1041274742959, 329.45390347185844, 746.3939300705359, 633.7311553168347, 522.4429537181754, 414.90236412466913]

# 50
# net_worth = [9563.18716185095, 11316.618707987753, 8255.96726385548, 5821.671731455569, 2819.823662909873, 1877.0002343618282, 2319.8360648541475, 2170.178687388205, 3450.869758388722, 1520.6384447735652, 1193.2245374593151, 1823.9520358858133, 1246.1779479589698, 956.5541383983033, 407.7260389659407, 1724.8923930163908, 794.1899539171111, 195.2179854882322, 742.5125860363829, 1413.5743609499445, 1978.10538037067, 662.1888789120319, 415.37579207166806, 612.2390914880867, 1225.043442191807, 324.7244005153826, 277.4330629879655, 1275.4928376052446, 202.48431285728267, 449.9124002470738, 1051.9103572491927, 277.4330629879655, 467.8242600680144, 711.4796392614129, 1371.3485397958882, 241.28255563502395, 159.61373035245813, 376.8749355689767, 278.0926962799301, 180.7543812759144, 195.76986675138727, 526.2050048934126, 787.2049721550167, 301.37750019143317, 425.8559764139693, 215.17144917123258, 526.2050048934126, 478.0018949617856, 508.29314507247204, 389.8141791450001]

# net_worth = [15683.449336585038, 20097.67715238473, 14718.173037772762, 4306.5367231357295, 827.3657030877666, 1861.5314904619256, 658.4540867825438, 917.0495177725267, 992.8115827982383, 1799.9962059043044, 1728.2197839060843, 341.635362958588, 3372.729912270289, 315.34912959445506, 1721.7030901900494, 1083.7067715236255, 306.7453837266065, 635.7382005975703, 764.3454799419649, 644.9523782829872, 657.9332717207445, 444.6436904156541, 572.8753649602751, 188.7624129151311, 360.21254788222006, 673.8146326188049, 477.18292385716916, 586.9640530593806, 248.91021060614653, 274.5228755223325]

# net_worth = [10314.907273228038, 34323.7391309832, 5446.3878464277095, 3823.98206971669, 1468.596895653609, 785.4512876035074, 864.8309788046458, 2546.879082919748, 1424.7167868135007, 875.691711397131, 1979.5526182604476, 787.9701138577974, 785.4512876035074, 634.9960897436217, 909.008324925568, 690.7455759841151, 1815.1985198943316, 720.656666295331, 722.3431231757489, 253.02605327396918, 720.656666295331, 621.8414911807871, 403.4812511338548, 645.0600704134184, 677.6410356538865, 797.9249696690603, 118.61865766506581, 608.2317631659137, 651.8292092813107, 87.90989625275381]
# fig, axes = plt.subplots(nrows=num_layers, ncols=2, figsize=(10, 8))
# fig, axes = plt.subplots(nrows=num_layers, ncols=2, figsize=(15, 8)) # M=1

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
    # Need to floor loans less than threshold 

        weight_mat = weight_mats[m]
        weight_mat[weight_mat < n_eps] = 0.
        debtrank = debtranks[m]
        debtrank[debtrank < dr_eps] = 0.

        if multi_layer_dr == True:
            debtrank = debtrank * (weight_mat.sum() / weight_mats.sum())



        G = nx.from_numpy_array(weight_mat, create_using=nx.DiGraph)

        # Create a list of node size: proportional to debtrank
        # size_param = [2, 100]
        # node_sizes = size_param[1]*np.exp(size_param[0]*debtranks[m])
        # size_param = 200 # bigger smaller
        size_param = 250
        size_param = 100 # for single layer
        node_sizes = np.array(net_worth)/size_param
        # node_sizes = 500

        # Create a list of node colors: proportional to debtrank
        node_colors_dr = debtranks[m]
        node_colors_cr = credit_score

        # do you need to sort nodelist?
        # pos = nx.shell_layout(G, shells)
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
            # node_shape=mplt.markers.MarkerStyle(marker='o', fillstyle='left')
        # g_nodes_cr = nx.draw_networkx_nodes(
        #     G,
        #     pos=pos,
        #     nodelist=sorted(G.nodes),
        #     node_size=node_sizes,
        #     node_color=node_colors_cr,
        #     edgecolors='black',
        #     cmap=plt.cm.PiYG,
        #     vmin=0.,
        #     vmax=1.0,
        #     ax=ax[subplot_ind],
        #     node_shape=mplt.markers.MarkerStyle(marker='o', fillstyle='right')
        #     )

        edges = nx.draw_networkx_edges(G, pos, ax=ax[subplot_ind], arrowstyle="->", alpha=0.2)
        # g_nodes = nx.draw_networkx_nodes(
        #     G, pos=pos, nodelist=sorted(G.nodes), node_size=node_sizes, node_color=node_colors, edgecolors='black', cmap=plt.cm.bwr, vmin=0., vmax=1.0
        #     )
        # edges = nx.draw_networkx_edges(G, pos)

        # G.edges(data=True)
        # plt.colorbar(g_nodes)
        # plt.axis('off')
        # plt.clf()
        # plt.savefig("network_" + network_str + "_layer-" + str(m))
        

        if num_layers > 1:
            ax[subplot_ind].set_title(network_str + " layer-" + str(m+1), fontsize="small")
        else:
            ax[subplot_ind].set_title(title_label[network_str]  + " Network", fontsize="small")

        ax[subplot_ind].set_axis_off() # draw on different subplot
        if num_layers > 1 and network_str == "reduced":
            subplot_ind += 1
            cax = ax[subplot_ind]
            plt.colorbar(g_nodes_dr, ax=ax[subplot_ind], cax=cax)

        # # plt.colorbar(g_nodes_cr, ax=ax[subplot_ind])
        # plt.axis('off')

        # ax[subplot_ind].set_axis_off() # draw on different subplot
        subplot_ind += 1

# for ind in range(len(ax)):
#     ax[ind].set_axis_off() # draw on different subplot

plt.tight_layout()
# plt.show()
if num_layers == 1:
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7]) # [0.85, 0.15, 0.05, 0.7]
    fig.colorbar(g_nodes_dr, cax=cbar_ax)

    # cbar_ax = fig.add_axes([ax[1].get_position().x1-0.02, ax[1].get_position().y0,0.02, ax[1].get_position().height]) # 0.03 on x position
    # fig.colorbar(g_nodes_dr, cax=cbar_ax, fraction=0.046, pad=0.04)

# plt.show()
plt.savefig(out_str + "/network_structure-"+red_network_str+".svg")
