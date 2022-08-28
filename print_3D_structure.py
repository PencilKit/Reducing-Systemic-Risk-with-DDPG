
"""
Plot multi-graphs in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import math
import os

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

class LayeredNetworkGraph(object):

    def __init__(self, graphs, node_labels=None, layout=nx.spring_layout, ax=None, plane_spacing=None, size_param=None, network_str=None):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Arguments:
        ----------
        graphs : list of networkx.Graph objects
            List of graphs, one for each layer.

        node_labels : dict node ID : str label or None (default None)
            Dictionary mapping nodes to labels.
            If None is provided, nodes are not labelled.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.

        """

        # book-keeping
        self.graphs = graphs
        self.total_layers = len(graphs)

        self.node_labels = node_labels
        self.layout = layout

        self.network_str = network_str

        self.plane_spacing = plane_spacing
        self.size_param = size_param
        self.radius = {}
        self.size = {}
        if ax:
            self.ax = ax

        # create internal representation of nodes and edges
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        # compute layout and plot
        self.get_node_positions()
        self.draw()


    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer, dr, networth)."""
        self.nodes = []
        for z, g in enumerate(self.graphs):
            for node_id in g.nodes():
                attributes = {'debtrank': g.nodes[node_id]['debtrank'], 'networth': g.nodes[node_id]['networth']}
                self.nodes.extend([((node_id, z), attributes)])
            # self.nodes.extend([(node, z) for node in g.nodes()])


    def get_edges_within_layers(self):
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges()])


    def get_edges_between_layers(self):
        """Determine edges between layers. Nodes in subsequent layers are
        thought to be connected if they have the same ID."""
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]
            shared_nodes = set(g.nodes()) & set(h.nodes())
            self.edges_between_layers.extend([((node, z1), (node, z2)) for node in shared_nodes])


    def get_node_positions(self, *args, **kwargs):
        """Get the node positions in the layered layout."""

        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        pos = self.layout(composition, *args, **kwargs)

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update({(node, z) : (*pos[node], z) for node in g.nodes()})


    def draw_nodes(self, nodes, *args, **kwargs):
        # Draw the nodes first so you can get their size

        x, y, z = zip(*[self.node_positions[node[0]] for node in nodes])
        temp_z = ()
        for zval in z:
            if zval > 0:
                temp_z = temp_z + (zval*self.plane_spacing,)
            else:
                temp_z = temp_z + (zval,)

        z = temp_z

        # Node colour
        colours = [node[1]['debtrank'] for node in nodes]
        # Node size
        size = {node[0]: node[1]['networth']/self.size_param for node in nodes}
        self.size.update(size)

        self.radius.update({node[0]: math.sqrt(node[1]['networth']/self.size_param)/2 for node in nodes})

        p = self.ax.scatter(x, y, z, cmap=cm.jet, c=colours, vmin=0, vmax=1, s=[size_v for size_v in size.values()], *args, **kwargs)
        return p


    def draw_edges(self, edges, within, *args, **kwargs):
        # edges contain a list of pairs of coordinates with the second coordinate being the layer
        segments = [(self.node_positions[source], self.node_positions[target], (source, target)) for source, target in edges]

        # Plot the edges
        if within == True:
            for vizedge in segments:
                # ax.arrow((*vizedge), (*vizedge.T), color="tab:gray")
                arw = Arrow3D(
                    [vizedge[0][0], vizedge[1][0]],
                    [vizedge[0][1], vizedge[1][1]],
                    [vizedge[0][2], vizedge[1][2]],
                    shrinkA=self.radius[vizedge[2][0]], # gets radius of node A at point the (nodeA, layer)
                    shrinkB=self.radius[vizedge[2][1]], # gets radius of node B at point the (nodeB, layer)
                    arrowstyle="->",
                    lw = 0.5,
                    mutation_scale=20,
                    *args,
                    **kwargs
                    )
                ax.add_artist(arw)

    def get_extent(self, pad=0.1):
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), \
            (ymin - pad * dy, ymax + pad * dy)


    def draw_plane(self, z, *args, **kwargs):
        if z > 0:
            z = self.plane_spacing*z
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u ,v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)


    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z in self.nodes:
            if node in node_labels:
                ax.text(*self.node_positions[(node, z)], node_labels[node], *args, **kwargs)

    def draw_layer_labels(self):
        for alpha in range(self.total_layers):
            ax.text(-2.0,-0.8, alpha, s="Layer " + str(alpha+1)) # (1, 1.5)

    def draw(self):
        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.2, zorder=1)
            p = self.draw_nodes(
                [node for node in self.nodes if node[0][1]==z],
                zorder=3
                )
        if self.network_str == "reduced":
            cbar_ax = fig.add_axes([ax.get_position().x1-0.02,ax.get_position().y0,0.02,ax.get_position().height]) # 0.03 on x position
            fig.colorbar(p, cax=cbar_ax, fraction=0.046, pad=0.04)

        self.draw_edges(self.edges_within_layers,  within=True, color='k', alpha=0.2, linestyle='-', zorder=0)
        self.draw_edges(self.edges_between_layers, within=False, color='k', alpha=0.2, linestyle='--', zorder=0)
        if self.node_labels:
            self.draw_node_labels(self.node_labels,
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  zorder=100)

        if self.network_str == "initial":
            self.draw_layer_labels()


if __name__ == '__main__':
    """ You need the .npy files in the correct folder. """

    # Load the data
    path_base = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    path_str = os.path.join(path_base, "data", "network_py_data", "to_print/")
    ext_str = ".npy"
    out_str = os.path.join(path_base, "data", "prints/")

    n_eps = 1
    dr_eps = 1e-5
    multi_layer_dr = True

    # N=30, M=3
    init_network_str = "33.295454885318456_network"
    red_network_str = "19.583415955389167_network"


    initial_network = np.load(path_str + init_network_str + ext_str)
    reduced_network = np.load(path_str + red_network_str + ext_str)

    num_layers = initial_network['network'].shape[0]
    NODES = initial_network['network'].shape[1]
    print("PRINTING RESULTS FOR (N, M): (", str(NODES), ", ", str(num_layers), ")")

    path_str = path_str + str(NODES) + "-"

    network_str_list = ['initial', 'reduced']
    title_label = {
        "initial": "Initial",
        "reduced": "Optimized"
    }
    mat_list = [initial_network['network'], reduced_network['network']]
    debtrank_list = [initial_network['debtrank'], reduced_network['debtrank']] 
    credit_list = [initial_network['creditrisk'], reduced_network['creditrisk']]

    loan_check = np.abs(mat_list[0].sum(axis=2) - mat_list[1].sum(axis=2))
    borrowing_check = np.abs(mat_list[0].sum(axis=1) - mat_list[1].sum(axis=1))

    if np.any(loan_check.sum(axis=0) + borrowing_check.sum(axis=0) > initial_network['c_eps']):
        raise ValueError("Difference between loan or borrowing too high.")

    print("Drawing for datasets: ", init_network_str, " and ", red_network_str, "...")

    # initialise figure and plot
    fig = plt.figure(figsize=(8,8))
    plane_spacing = 1
    # subplot_ind = 0
    subplot_count = 1
    for weight_mats, debtranks, credit_score, network_str in zip(mat_list, debtrank_list, credit_list, network_str_list):
        graph_list = []
        for m in range(num_layers):
            weight_mat = weight_mats[m]
            weight_mat[weight_mat < n_eps] = 0.
            dr_dict = {node_id: {'debtrank': dr} for (node_id, dr) in zip(list(range(NODES)), debtranks[m])}
            networth_dict = {node_id: {'networth': netw} for (node_id, netw) in zip(list(range(NODES)), initial_network['networth'])}
            # Add the DR properties into the graph here
            g = nx.from_numpy_array(weight_mat, create_using=nx.DiGraph) # used for colour
            nx.set_node_attributes(g, dr_dict)
            nx.set_node_attributes(g, networth_dict)
            graph_list.append(g)

        node_labels = {nn : str(nn) for nn in range(4*NODES)}

        ax = fig.add_subplot(int(str(12)+str(subplot_count)), projection='3d', computed_zorder=False)
        subplot_count = subplot_count+1
        LayeredNetworkGraph(graph_list, node_labels=node_labels, ax=ax, layout=nx.circular_layout, size_param=800, plane_spacing=plane_spacing, network_str=network_str)

        ax.set_title(title_label[network_str] + " Network")
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        ax.view_init(23,-68)
        plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()
    # plt.savefig(out_str + "/3d_struct.svg", bbox_inches='tight',pad_inches = 0)
