import sys

from CPLS.alg_cpls import CPLS
from CPLS.get_true_pf import get_true_pareto_front
from pymoo.indicators.igd import IGD
from baselines.pymoo_lib import *
from baselines.pygmo_lib import *
from prediction_model.help import *
from prediction_model.prediction import *

if __name__ == '__main__':

    itr = sys.argv[1]
    system_list = ['Android','Apache','BGL','Hadoop','HDFS','HealthApp','HPC','Linux','Mac','OpenSSH','OpenStack',
                   'Proxifier','Spark','Thunderbird','Windows','Zookeeper']
    system_list = sorted(system_list, key=lambda x: x.lower())

    system_pair = pd.read_csv(f"dataset/data_selection.csv")
    data_name = f"dataset/log_parsing.csv"

    problem_type = 'LLMs'
    model_list = ["Mixtral_8x7B", "llama2_7b", "llama2_13b", "llama2_70b", "Yi_34B", "Yi_6B", "j2_mid", "j2_ultra",
                  "gpt1"]

    for test_sys in system_list:
        save_dir = f"res/{test_sys}/{problem_type}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train_sys = system_pair[system_pair['test_system'] == test_sys]['best'].values[0]
        df_pre_accuracy, df_true_accuracy, df_cost, df_label_accuracy = log_preprocess(data_name, model_list,
                                                                                       train_sys, test_sys)
        true_pt, true_pt_solution = get_true_pareto_front(df_true_accuracy, df_cost, model_list).run()
        true_pt.to_csv(f"{save_dir}/true_pt_{itr}.csv")

        smsemoa_res, smsemoa_solutions = sms_emoa(df_pre_accuracy, df_true_accuracy, df_cost,
                                            model_list, termination=100).run()
        smsemoa_res.to_csv(f"{save_dir}/smsemoa_res_{itr}.csv")

        nsga2_res, nsga2_solutions = nsga2(df_pre_accuracy, df_true_accuracy, df_cost,
                                            model_list, termination=100).run()
        nsga2_res.to_csv(f"{save_dir}/nsga2_res_{itr}.csv")

        rnsga2_res, rnsga2_solutions = rnsga2(df_pre_accuracy, df_true_accuracy, df_cost,
                                            model_list, termination=100).run()
        rnsga2_res.to_csv(f"{save_dir}/rnsga2_res_{itr}.csv")

        mopso_res, mopso_solutions = MOPSO(df_pre_accuracy, df_true_accuracy, df_cost,
                                            model_list, termination=100).run()
        mopso_res.to_csv(f"{save_dir}/mopso_res_{itr}.csv")

        moead_res, moead_solutions = MOEAD(df_pre_accuracy, df_true_accuracy, df_cost,
                                            model_list, termination=100).run()
        moead_res.to_csv(f"{save_dir}/moead_res_{itr}.csv")

        moeadgen_res, moeadgen_solutions = MOEADGEN(df_pre_accuracy, df_true_accuracy, df_cost, model_list,
                                                    termination=100).run()
        moeadgen_res.to_csv(f"{save_dir}/moeadgen_res_{itr}.csv")

        CPLS_res, CPLS_solution = CPLS(df_pre_accuracy, df_true_accuracy, df_cost, model_list, sampling_sum=10, initial_solution_size=87).run()
        CPLS_res.to_csv(f"{save_dir}/OptLLM_res.csv")

