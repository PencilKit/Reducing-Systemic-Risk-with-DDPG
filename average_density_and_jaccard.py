from pickle import FALSE, TRUE
import importing_modules as im
import copy as cy
import pandas as pd
import average_results as ar
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

    column_name = ["Initial DR", "Optimized DR", "% Reduction", "Init std", "Opt Std", "Red Std"]

    # Load the data
    path_base = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    out_path = os.path.join(path_base, "data", "prints", "table")

    # DR
    do_leverage_results = False # don't change
    number_of_episodes = 1

    if do_leverage_results == False:
        layer_ind = [1,2,3]
        density_table_list = []
        jaccard_table_list = []
        for case in ["original"]:
            if case == "uniform":
                policy_dict = policy_uniform_dict
                print(case)
            elif case=="original":
                policy_dict = policy_dict
            rows = []
            for n in [10, 20, 30]:
                for m in layer_ind:
                    print("WORKING ON: N=", n, " M=", m)
                    avg_init_density_dict, avg_opt_density_dict, avg_init_jaccard_mat, avg_opt_jaccard_mat, std_init_density_dict, std_opt_density_dict, std_init_jaccrd_mat, std_opt_jaccard_mat = ar.average_dr(
                        n,
                        m,
                        number_of_episodes,
                        LEVERAGE_EXP=do_leverage_results,
                        CASE=case,
                        POLICY=policy_dict[n][m],
                        results="stats"
                        )
                    
                    density_df = pd.concat(
                        [
                            pd.DataFrame.from_dict(avg_init_density_dict, orient="index", columns=["Initial Density"]),
                            pd.DataFrame.from_dict(avg_opt_density_dict,  orient="index", columns=["Optimized Density"]),
                            pd.DataFrame.from_dict(std_init_density_dict, orient="index", columns=["Initial Density STD"]),
                            pd.DataFrame.from_dict(std_opt_density_dict, orient="index", columns=["Optimized Density STD"])
                            ],
                            axis=1
                        )

                    jaccard_df = pd.concat(
                        [
                            pd.DataFrame(avg_init_jaccard_mat),
                            pd.DataFrame(avg_opt_jaccard_mat),
                            pd.DataFrame(std_init_jaccrd_mat),
                            pd.DataFrame(std_opt_jaccard_mat)
                        ],
                        axis=1
                    )


                    density_df.to_csv(out_path + "/density-"+str(n)+ "-" + str(m) +"_"+ str(do_leverage_results) + "_" + str(number_of_episodes) + "_" + case + ".csv")
                    jaccard_df.to_csv(out_path + "/jaccard-"+str(n)+ "-" + str(m) +"_"+ str(do_leverage_results) + "_" + str(number_of_episodes) + "_" + case + ".csv")
                    print("stop")

    print("done")

