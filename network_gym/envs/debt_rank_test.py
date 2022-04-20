import model_network as md
N_AGENTS = 5
M_LAYERS = 1
THETA = [0.40]

ASSETS = 10000
BETA=0.18
GAMMA=0.07
R=1

SEED = 16470104418303546103
COMB_DEBTRANK = False

graph = md.Multigraph(
    N_AGENTS,
    M_LAYERS,
    ASSETS,
    THETA,
    BETA,
    GAMMA,
    R,
    SEED,
    COMB_DEBTRANK
)

graph.calculate_alpha_debtrank(0)