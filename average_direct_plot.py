import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.font_manager import FontProperties
import os

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
    plot_labels = {
        "uniform": 'Uniform',
        "linear": 'Linear',
        "exponential1": 'Exponential',
        "exponential10": 'Exponential v=10.0'
    }

    colors = {
        "uniform": 'r',
        "linear": 'b',
        "exponential1": 'y',
        "exponential10": 'c'
    }

    column_name = ["Initial DR", "Optimized DR", "% Reduction", "Init std", "Opt Std", "Red Std"]

    # Load the data
    path_base = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    out_path = os.path.join(path_base, "data", "prints", "table")

    # DR
    do_leverage_results = True
    number_of_episodes = 1

    bar_width = 0
    width = 0.1
    fontP = FontProperties()
    fontP.set_size('small')
    fig_err, ax_err = plt.subplots(1, figsize=(8, 6))
    fig, ax = plt.subplots(1, figsize=(8, 6))
    x =  np.arange(2)

    with open(out_path + "/leverage_dictionary10.pkl", 'rb') as f:
        saved_data_dict = pickle.load(f)

    if do_leverage_results == True:
        case="uniform"
        column_name = column_name*len(lev_policy_dict[30].keys())
        n_ind = [30]
        table_rows = []
        for n in n_ind:
            weight_rows = []
            for weight_func in lev_policy_dict[n].keys():
                print("WORKING ON: N=", n, " M=", 1, "Weight: ", weight_func)

                # Take and assign these from the loaded data
                avg_high_dr, avg_low_dr, std_high_dr, std_low_dr = saved_data_dict[weight_func].values()

                vals = [avg_low_dr, avg_high_dr]
                error = [std_low_dr, std_high_dr]

                ax.bar(x + bar_width, vals, width, alpha=0.5, color=colors[weight_func], align="edge", label=plot_labels[weight_func])
                ax_err.bar(x + bar_width, vals, width, alpha=0.5, color=colors[weight_func], align="edge", label=plot_labels[weight_func],yerr=error, capsize=2.5)
                bar_width += width

        d_from_edge = 0.2
        ax.legend(loc="upper left", prop=fontP)
        ax.set_xticks(x+width*2, ["Banks with Low Leverage Ratios", "Banks with High Leverage Ratios"])
        ax.set_xlim([-d_from_edge, 1.4+d_from_edge])
        ax.set_ylabel("Average Total DebtRanks")
        ax.set_ylim([0, 2.3])

        ax_err.legend(loc="upper left", prop=fontP)
        ax_err.set_xticks(x+width*2, ["Banks with Low Leverage Ratios", "Banks with High Leverage Ratios"])
        ax_err.set_xlim([-d_from_edge, 1.4+d_from_edge])
        ax_err.set_ylabel("Average Total DebtRanks")
        ax_err.set_ylim([0, 2.3])

        plt.tick_params(bottom = False)
        # fig.savefig(out_path + "/leverage-" + str(n) + "-"+ str(number_of_episodes) +".svg")
        # fig_err.savefig(out_path + "/leverage-error-" + str(n) + "-"+ str(number_of_episodes) +".svg")

        plt.show()
    print("DONE")