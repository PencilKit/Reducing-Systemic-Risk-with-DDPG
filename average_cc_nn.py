import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pickle
import os
import average_results as ar

if __name__ == "__main__":

    policy_dict = {
        30: {
            1: "18.161872585900607",
            2: "26.31708381856567",
            3: "33.295454885318456"
        },
        20: {
            1: "13.872525203609701",
            2: "23.260422907686706",
            3: "30.402851902433095"
        },
        10: {
            1: "7.7408578246030135",
            2: "14.021826409799196",
            3: "21.37639441829311"
        }
    }

    policy_uniform_dict = {
        30: {
            1: "12.534112174999047",
            2: "11.848197437601694",
            3: "14.058939038527265"
        },
        20: {
            1: "12.275934293341955",
            2: "11.680745742070073",
            3: "14.629125199358128"
        },
        10: {
            1: "7.074051130008529",
            2: "11.373744396360374",
            3: "14.514386717235313"
        }
    }

    lev_policy_dict = {
        30: {"uniform": "12.665814697812916-lev-uniform-30",
        "linear": "12.665814697812916-lev-linear-30",
        "exponential1": "12.665814697812916-lev-exp1-30", 
        "exponential10": "12.665814697812916-lev-exp10-30"
        },
        20: {"uniform": "11.118127820283755-lev-uniform-20",
        "linear": "11.118127820283755-lev-linear-20",
        "exponential1": "11.118127820283755-lev-exp1-20", 
        "exponential10": "11.118127820283755-lev-exp10-20"
        },
        10: {"uniform": "7.846093132926077-lev-uniform-10",
        "linear": "7.846093132926077-lev-linear-10",
        "exponential1": "7.846093132926077-lev-exp1-10", 
        "exponential10": "7.846093132926077-lev-exp10-10"
        }
    }


    column_name = ["Initial DR", "Optimized DR", "% Reduction", "Init std", "Opt Std", "Red Std"]

    # Load the data
    path_base = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    out_path = os.path.join(path_base, "data", "prints", "table")

    # DR
    do_leverage_results = False
    number_of_episodes = 1

    fontP = FontProperties()
    fontP.set_size('small')
    color_dict = {
        "initial": 'r',
        "reduced": 'b'
    }

    label_dict = {
        "initial": "initial",
        "reduced": "optimized"
    }
    ytitle_layers = ["$R(\mathbf{L}^{1}, \mathbf{e})$",
    "$R(\mathbf{L}^{2}, \mathbf{e})$",
    "$R(\mathbf{L}^{3}, \mathbf{e})$"]

    if do_leverage_results == False:
        layer_ind = [3]
        density_table_list = []
        jaccard_table_list = []
        for case in ["original"]:
            if case == "uniform":
                policy_dict = policy_uniform_dict
                print(case)
            elif case=="original":
                policy_dict = policy_dict
            rows = []
            for n in [30]:
                for m in layer_ind:
                    fig_cc, ax_cc = plt.subplots(m, figsize=(5, 6))
                    fig_and, ax_and = plt.subplots(m, figsize=(5, 6))
                    print("WORKING ON: N=", n, " M=", m)
                    initial_debtrank, optimized_debtrank, init_cc, opt_cc, init_nn, opt_nn = ar.average_dr(
                        n,
                        m,
                        number_of_episodes,
                        LEVERAGE_EXP=do_leverage_results,
                        CASE=case,
                        POLICY=policy_dict[n][m],
                        results="cc_nn"
                        )
                    save_data = [initial_debtrank, optimized_debtrank, init_cc, opt_cc, init_nn, opt_nn]
                    with open(out_path + "/cc_nn100.pkl", "wb") as f:
                        pickle.dump(save_data, f)

                    for mm in range(m):
                        ax_cc[mm].scatter(
                            init_cc[mm],
                            initial_debtrank[mm],
                            color=color_dict["initial"],
                            label=label_dict["initial"]
                            )
                        ax_cc[mm].scatter(
                            opt_cc[mm],
                            optimized_debtrank[mm],
                            color=color_dict["reduced"],
                            label=label_dict["reduced"]
                            )

                        ax_and[mm].scatter(
                            init_nn[mm],
                            initial_debtrank[mm],
                            color=color_dict["initial"],
                            label=label_dict["initial"]
                            )
                        ax_and[mm].scatter(
                            opt_nn[mm],
                            optimized_debtrank[mm],
                            color=color_dict["reduced"],
                            label=label_dict["reduced"]
                            )
                        if mm == m-1:
                            ax_cc[mm].set_xlabel("Average Clustering Coefficient")
                            ax_and[mm].set_xlabel("Total Average Neighbourhood Degree")
                        ax_cc[mm].set_ylabel(ytitle_layers[mm])
                        ax_and[mm].set_ylabel(ytitle_layers[mm])

                    ax_cc[0].legend(loc="upper left", prop=fontP)
                    ax_and[0].legend(loc="upper left", prop=fontP)
                    fig_cc.align_ylabels(ax_cc)
                    fig_and.align_ylabels(ax_and)
                    fig_cc.tight_layout()
                    fig_and.tight_layout()
                    fig_cc.savefig(out_path + "/cc_structure-"+str(n)+ "-" + str(m) + "-" + str(number_of_episodes) +".svg")
                    fig_and.savefig(out_path + "/and_structure-"+str(n)+ "-" + str(m) +"-" + str(number_of_episodes) + ".svg")
                    
                    print("stop")

    print("done")

