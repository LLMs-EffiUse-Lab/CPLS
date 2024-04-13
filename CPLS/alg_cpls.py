import heapq
import multiprocessing as mp
import random
import time
from itertools import combinations
from tqdm import tqdm
from utilities import *

class CPLS(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, sampling_sum, initial_solution_size):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.model_list = model_list
        self.df_cost = df_cost
        self.scope =  len(self.df_true_accuracy.iloc[0]) # minus the name column
        self.sampling_sum = sampling_sum
        self.initial_solution_size = initial_solution_size
        self.job_num = len(self.df_true_accuracy)
        # self.delta = 5

    def ls_fitness_function_(self, s_allocation):
        s_cost = np.zeros(len(s_allocation))
        s_accuracy = np.zeros(len(s_allocation))
        for i, allocation in enumerate(s_allocation):
            s_cost[i] = self.df_cost.iloc[i, allocation]
            s_accuracy[i] = self.df_pre_accuracy.iloc[i, allocation]
        s_total_cost = np.sum(s_cost)
        s_accuracy_mean = np.mean(s_accuracy)
        return [s_total_cost, s_accuracy_mean]

    def get_maximum_improvement_machine_(self, current_allocation, job_id):
        if job_id < 0 or job_id > self.scope - 1:
            raise NotImplementedError(f"""the number of job exceeds the limit, please check the job_id: {job_id}""")

        accuracy = self.df_pre_accuracy.iloc[job_id]
        cost_list = self.df_cost.iloc[job_id]

        accuracy_diff = []
        cost_list_diff = []
        job_index = []
        for ele in range(len(cost_list)):
            if current_allocation[job_id] < 0 or current_allocation[job_id] > self.scope - 1:
                print(current_allocation[job_id])
                raise NotImplementedError(
                    f"""the allocation exceeds the limit, please check the job_id: {current_allocation[job_id]}""")
            if ele != current_allocation[job_id]:
                cost_list_diff.append(cost_list.iloc[ele] - cost_list.iloc[current_allocation[job_id]])
                accuracy_diff.append(accuracy.iloc[ele] - accuracy.iloc[current_allocation[job_id]])
                job_index.append(ele)

        improvement_list = []
        improvement_index = []
        for i in range(len(cost_list_diff)):
            if accuracy_diff[i] <= 0:
                improvement_list.append(0)
                improvement_index.append(job_index[i])
            else:
                improvement_list.append(accuracy_diff[i] / cost_list_diff[i])
                improvement_index.append(job_index[i])
        return improvement_index[improvement_list.index(max(improvement_list))], max(improvement_list)

    def maximum_improvement_per_job_(self, current_allocation):
        all_improvement = []
        all_level = []
        for i in range(len(current_allocation)):
            improvement_index, max_improvement = self.get_maximum_improvement_machine_(current_allocation, i)
            all_improvement.append(max_improvement)
            all_level.append(improvement_index)
        return all_improvement, all_level

    # optimise a solution to a non-dominated solution
    def get_maximum_optimisation_(self, current_allocation, job_id):
        accuracy = self.df_pre_accuracy.iloc[job_id]
        cost_list = self.df_cost.iloc[job_id]
        accuracy_diff = []
        cost_list_diff = []
        job_index = []

        for ele in range(len(cost_list)):
            if ele != current_allocation[job_id]:
                cost_list_diff.append(cost_list.iloc[ele] - cost_list.iloc[current_allocation[job_id]])
                accuracy_diff.append(accuracy.iloc[ele] - accuracy.iloc[current_allocation[job_id]])
                job_index.append(ele)

        accuracy_change_unit_cost = []
        accuracy_change_unit_cost_index = []

        for i in range(len(cost_list_diff)):
            if cost_list_diff[i] <= 0 and accuracy_diff[i] >= 0 and (accuracy_diff[i] - cost_list_diff[i]) > 0:
                accuracy_change_unit_cost.append(float('inf'))
                accuracy_change_unit_cost_index.append(job_index[i])
            elif cost_list_diff[i] > 0 and accuracy_diff[i] > 0:
                accuracy_change_unit_cost.append(accuracy_diff[i] / cost_list_diff[i])  # bigger is better
                accuracy_change_unit_cost_index.append(job_index[i])
            elif cost_list_diff[i] < 0 and accuracy_diff[i] < 0:
                accuracy_change_unit_cost.append(-accuracy_diff[i] / cost_list_diff[i])  # bigger is worse
                accuracy_change_unit_cost_index.append(job_index[i])
            else:
                accuracy_change_unit_cost.append(float('-inf'))
                accuracy_change_unit_cost_index.append(job_index[i])
        return accuracy_change_unit_cost, accuracy_change_unit_cost_index

    def optimise_per_job_(self, current_allocation):
        accuracy_change_unit_cost_list = []
        accuracy_change_unit_cost_index_list = []
        for i in range(len(current_allocation)):
            accuracy_change_unit_cost, accuracy_change_unit_cost_index = self.get_maximum_optimisation_(current_allocation,
                                                                                                        i)
            accuracy_change_unit_cost_list.append(accuracy_change_unit_cost)
            accuracy_change_unit_cost_index_list.append(accuracy_change_unit_cost_index)

        return accuracy_change_unit_cost_list, accuracy_change_unit_cost_index_list

    def get_minimize_influecne_(self, current_allocation, job_id):
        accuracy = self.df_pre_accuracy.iloc[job_id]
        cost_list = self.df_cost.iloc[job_id]

        accuracy_diff = []
        cost_list_diff = []
        job_index = []
        for ele in range(len(cost_list)):
            if current_allocation[job_id] < 0 or current_allocation[job_id] > self.scope - 1:
                raise NotImplementedError(
                    f"""the allocation exceeds the limit, please check the job_id: {current_allocation[job_id]}""")
            if ele != current_allocation[job_id]:
                cost_list_diff.append(cost_list.iloc[ele] - cost_list.iloc[current_allocation[job_id]])
                accuracy_diff.append(accuracy.iloc[ele] - accuracy.iloc[current_allocation[job_id]])
                job_index.append(ele)

        cost_saving_list = []
        cost_saving_index = []
        for i in range(len(cost_list_diff)):
            if cost_list_diff[i] >= 0:
                cost_saving_list.append(0)
                cost_saving_index.append(job_index[i])
            else:
                cost_saving_list.append(cost_list_diff[i] / accuracy_diff[i])
                # if accuracy_diff[i] == 0:
                #     print(cost_list_diff[i] / accuracy_diff[i])
                cost_saving_index.append(job_index[i])
        return cost_saving_index[cost_saving_list.index(max(cost_saving_list))], max(cost_saving_list)

    def minimize_influecne_per_job_(self, current_allocation):
        all_cost_saving = []
        all_level = []
        for i in range(len(current_allocation)):
            influence_index, min_influence = self.get_minimize_influecne_(current_allocation, i)
            # print(max_improvement)
            all_cost_saving.append(min_influence)
            all_level.append(influence_index)
        # job_index = all_improvement.index(max(all_improvement))
        return all_cost_saving, all_level

    def is_pareto_(self, costs, return_mask=True):
        costs = np.array(costs)
        costs[:, 1] = costs[:, 1] * -1
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    def is_dominated_(res1, res2):
        if res1[0] >= res2[0] and res1[1] > res2[1]:
            return True
        elif res1[0] > res2[0] and res1[1] >= res2[1]:
            return True
        else:
            return False

    def optimise_to_nondominated_(self, current_solution):
        # Create new_solutions_set and new_solutions_res_set with the current_solution as the initial set
        new_solutions_set = [current_solution]
        new_solutions_res_set = [self.ls_fitness_function_(current_solution)]

        new_solution = current_solution.copy()
        accuracy_change_unit_cost_list, accuracy_change_unit_cost_index_list = self.optimise_per_job_(new_solution)
        np_accuracy_change_unit_cost_list = np.array(accuracy_change_unit_cost_list)
        if np.any(np.isposinf(np_accuracy_change_unit_cost_list)):
            inf_index_row1, inf_index_col1 = np.where(np.isposinf(np_accuracy_change_unit_cost_list))
            new_inf_index = np.unique(inf_index_row1, return_index=True)
            # get the coressponding col index in inf_index_col
            inf_index_row = new_inf_index[0]
            inf_index_col = inf_index_col1[new_inf_index[1]]
            for i in range(len(inf_index_row)):
                new_solution[inf_index_row[i]] = accuracy_change_unit_cost_index_list[inf_index_row[i]][
                    inf_index_col[i]]
                accuracy_change_unit_cost, accuracy_change_unit_cost_index = self.get_maximum_optimisation_(
                    new_solution, inf_index_row[i])
                np_accuracy_change_unit_cost_list[inf_index_row[i]] = accuracy_change_unit_cost
                accuracy_change_unit_cost_index_list[inf_index_row[i]] = accuracy_change_unit_cost_index
        new_solutions_res_set.append(self.ls_fitness_function_(new_solution))
        new_solutions_set.append(new_solution.copy())

        condition_i = (np_accuracy_change_unit_cost_list != float('inf')) & (np_accuracy_change_unit_cost_list > 0)
        condition_d = (np_accuracy_change_unit_cost_list != float('-inf')) & (np_accuracy_change_unit_cost_list < 0)

        if np_accuracy_change_unit_cost_list[condition_d].size == 0 or np_accuracy_change_unit_cost_list[
            condition_i].size == 0:
            return new_solutions_res_set, new_solutions_set
        max_increase_value = max(np_accuracy_change_unit_cost_list[condition_i])
        max_decrease_value = max(np_accuracy_change_unit_cost_list[condition_d])

        while abs(max_decrease_value) < max_increase_value and max_increase_value > 0 and abs(max_decrease_value) > 0:
            max_increase_value_index_row, max_increase_value_index_col = np.where(
                np_accuracy_change_unit_cost_list == max_increase_value)
            max_decrease_value_index_row, max_decrease_value_index_col = np.where(
                np_accuracy_change_unit_cost_list == max_decrease_value)
            for i in range(len(max_increase_value_index_row)):
                new_solution[max_increase_value_index_row[i]] = \
                    accuracy_change_unit_cost_index_list[max_increase_value_index_row[i]][
                        max_increase_value_index_col[i]]
                # job_id = max_increase_value_index[0][i]
                accuracy_change_unit_cost, accuracy_change_unit_cost_index = self.get_maximum_optimisation_(
                    new_solution,max_increase_value_index_row[i])
                np_accuracy_change_unit_cost_list[max_increase_value_index_row[i]] = accuracy_change_unit_cost
                accuracy_change_unit_cost_index_list[max_increase_value_index_row[i]] = accuracy_change_unit_cost_index
            for i in range(len(max_decrease_value_index_row)):
                new_solution[max_decrease_value_index_row[i]] = \
                    accuracy_change_unit_cost_index_list[max_decrease_value_index_row[i]][
                        max_decrease_value_index_col[i]]
                accuracy_change_unit_cost, accuracy_change_unit_cost_index = self.get_maximum_optimisation_(
                    new_solution, max_decrease_value_index_row[i])
                np_accuracy_change_unit_cost_list[max_decrease_value_index_row[i]] = accuracy_change_unit_cost
                accuracy_change_unit_cost_index_list[max_decrease_value_index_row[i]] = accuracy_change_unit_cost_index

            while np.any(np.isposinf(np_accuracy_change_unit_cost_list)):
                inf_index_row1, inf_index_col1 = np.where(np.isposinf(np_accuracy_change_unit_cost_list))
                new_inf_index = np.unique(inf_index_row1, return_index=True)
                # get the coressponding col index in inf_index_col
                inf_index_row = new_inf_index[0]
                inf_index_col = inf_index_col1[new_inf_index[1]]
                for i in range(len(inf_index_row)):
                    new_solution[inf_index_row[i]] = accuracy_change_unit_cost_index_list[inf_index_row[i]][
                        inf_index_col[i]]
                    accuracy_change_unit_cost, accuracy_change_unit_cost_index = self.get_maximum_optimisation_(
                        new_solution, inf_index_row[i])
                    np_accuracy_change_unit_cost_list[inf_index_row[i]] = accuracy_change_unit_cost
                    accuracy_change_unit_cost_index_list[inf_index_row[i]] = accuracy_change_unit_cost_index
            new_solutions_res_set.append(self.ls_fitness_function_(new_solution))
            new_solutions_set.append(new_solution.copy())
            max_increase_value = max(
                np_accuracy_change_unit_cost_list[np_accuracy_change_unit_cost_list != float('inf')])
            condition = (np_accuracy_change_unit_cost_list != float('-inf')) & (np_accuracy_change_unit_cost_list < 0)
            max_decrease_value = max(np_accuracy_change_unit_cost_list[condition])

        return np.array(new_solutions_res_set), new_solutions_set

    def get_exreme_solution_(self):
        def get_high_accuracy_():
            s_high_accuracy = []
            for x in range(len(self.df_pre_accuracy)):
                accuracy = self.df_pre_accuracy.iloc[x]
                max_index = accuracy[accuracy == accuracy.max()].index[0]
                max_index_list = accuracy[accuracy == accuracy.max()].index.tolist()
                if len(max_index_list) > 1:
                    # print(max_index_list)
                    cost = self.df_cost.iloc[x]
                    all_cost = cost[max_index_list]
                    min_cost = min(all_cost)
                    max_index = all_cost[all_cost == min_cost].index[0]
                for j in range(len(self.model_list)):
                    if max_index == self.model_list[j]:
                        s_high_accuracy.append(j)
            return s_high_accuracy

        def get_low_cost_():
            s_low_cost = []
            for x in range(len(self.df_pre_accuracy)):
                cost = self.df_cost.iloc[x]
                min_index = cost[cost == cost.min()].index[0]
                min_index_list = cost[cost == cost.min()].index.tolist()
                if len(min_index_list) > 1:
                    accuracy = self.df_pre_accuracy.iloc[x]
                    all_accuracy = accuracy[min_index_list]
                    max_accuracy = max(all_accuracy)
                    min_index = all_accuracy[all_accuracy == max_accuracy].index[0]
                for j in range(len(self.model_list)):
                    if min_index == self.model_list[j]:
                        s_low_cost.append(j)
            return s_low_cost

        s_high_accuracy = get_high_accuracy_()
        s_cheapest = get_low_cost_()

        return s_cheapest, s_high_accuracy

    def get_pareto_front_from_cheapest(self):
        start_time = time.time()
        print('Start to get pareto front!')
        new_solution = []
        for i in range(self.job_num):
            new_solution.append(0)
        res_from_cheapest = []
        all_solutions_from_cheapest = []
        all_solutions_from_cheapest.append(new_solution.copy())
        res_from_cheapest.append(self.ls_fitness_function_(new_solution))
        all_improvement, all_level = self.maximum_improvement_per_job_(new_solution)
        max_improvement = max(all_improvement)
        while max_improvement > 0:
            max_improvement_index = all_improvement.index(max(all_improvement))
            max_level = all_level[max_improvement_index]
            new_solution[max_improvement_index] = max_level
            all_solutions_from_cheapest.append(new_solution.copy())
            res_from_cheapest.append(self.ls_fitness_function_(new_solution))
            new_index, new_improvement = self.get_maximum_improvement_machine_(new_solution, max_improvement_index)
            all_improvement[max_improvement_index] = new_improvement
            all_level[max_improvement_index] = new_index
            max_improvement = max(all_improvement)
        elapsed_time = time.time() - start_time

        res_from_cheapest = pd.DataFrame(res_from_cheapest, columns=['cost', 'expected_accuracy'])
        res_from_cheapest['time'] = elapsed_time
        res_from_cheapest['true_accuracy'] = self.get_true_accuracy_obj_(all_solutions_from_cheapest)

        print("Get the Pareto Front spending time: ", elapsed_time)
        return res_from_cheapest, all_solutions_from_cheapest, elapsed_time

    def initialization_random_(self):
        initial_solution = []
        for _ in range(self.initial_solution_size):
            num = random.randint(1, self.scope)
            candidate = list(range(0, self.scope))
            candidate_ID = random.sample(candidate, num)
            random_solution = []
            for j in range(self.job_num):
                random_solution.append(random.choice(candidate_ID))
            initial_solution.append(random_solution)

        return initial_solution

    def initialization2_(self):
        my_list = list(range(self.scope))
        all_combinations = []
        for r in range(1, len(my_list) + 1):
            combinations_r = list(combinations(my_list, r))
            all_combinations.extend(combinations_r)
        all_combinations_as_lists = [list(combination) for combination in all_combinations]
        print(len(all_combinations_as_lists))

        s = []
        for candidate_ID in all_combinations_as_lists:
            random_solution = []
            for j in range(self.job_num):
                random_solution.append(random.choice(candidate_ID))
            s.append(random_solution)

        for i in range(1, 6):
            s.append([i] * self.job_num)

        return s
    def initialization_(self):
        initial_solution = []
        initial_solution_size = self.initial_solution_size
        for _ in range(initial_solution_size):
            num = random.randint(1, self.scope)
            candidate = list(range(0, self.scope))
            candidate_ID = random.sample(candidate, num)
            random_solution = []
            for j in range(self.job_num):
                random_solution.append(random.choice(candidate_ID))
            initial_solution.append(random_solution)

        for i in range(1, self.scope - 1):
            initial_solution.append([i] * self.job_num)

        n_cpu = os.cpu_count()
        with mp.Pool(processes=n_cpu) as pool:
            res_before_optimized = pool.starmap(self.ls_fitness_function_,
                                                [(solution) for solution in initial_solution])

        # print(res_before_optimized)
        res_before_optimized = np.array(res_before_optimized)
        pareto_efficient_mask = self.is_pareto_(res_before_optimized.copy())
        nondominated_inital_solutions = np.array(initial_solution)[pareto_efficient_mask]
        nondominated_inital_res = res_before_optimized[pareto_efficient_mask]

        while len(nondominated_inital_solutions) < self.initial_solution_size:
            num = random.randint(1, self.scope)
            candidate = list(range(0, self.scope))
            candidate_ID = random.sample(candidate, num)
            random_solution = []
            for j in range(self.job_num):
                random_solution.append(random.choice(candidate_ID))
            nondominated_inital_solutions = np.vstack([nondominated_inital_solutions, random_solution])


        while len(nondominated_inital_solutions) > self.initial_solution_size:
            # delete one row in nondominated_inital_solutions randomly
            nondominated_inital_solutions = np.delete(nondominated_inital_solutions,
                                                      random.randint(0, len(nondominated_inital_solutions) - 1), axis=0)
        return nondominated_inital_solutions

    # get the index of the maximum improvement, if more than one, out all of them into a list
    def search_the_gap(self, s_cheapest, s_high_accuracy):
        res1 = self.ls_fitness_function_(s_cheapest)
        res2 = self.ls_fitness_function_(s_high_accuracy)
        all_res = [res1, res2]
        all_solutions = [s_cheapest, s_high_accuracy]
        new_solution_higher_accuracy = s_cheapest.copy()
        new_solution_cheaper = s_high_accuracy.copy()

        all_improvement, all_improvement_allocation = self.maximum_improvement_per_job_(new_solution_higher_accuracy)
        all_saving, all_saving_allocation = self.minimize_influecne_per_job_(new_solution_cheaper)

        start_time = time.time()
        while res1[0] < res2[0]:
            max_improvement_index = all_improvement.index(max(all_improvement))
            max_improvement_allocation = all_improvement_allocation[max_improvement_index]
            new_solution_higher_accuracy[max_improvement_index] = max_improvement_allocation
            all_solutions.append(new_solution_higher_accuracy.copy())
            res1 = self.ls_fitness_function_(new_solution_higher_accuracy)
            all_res.append(res1)
            new_index1, new_improvement1 = self.get_maximum_improvement_machine_(new_solution_higher_accuracy,
                                                                                 max_improvement_index)
            all_improvement[max_improvement_index] = new_improvement1
            all_improvement_allocation[max_improvement_index] = new_index1

            max_saving_index = all_saving.index(max(all_saving))
            max_saving_allocation = all_saving_allocation[max_saving_index]
            new_solution_cheaper[max_saving_index] = max_saving_allocation
            all_solutions.append(new_solution_cheaper.copy())
            res2 = self.ls_fitness_function_(new_solution_cheaper)
            all_res.append(res2)
            new_index2, new_improvement2 = self.get_minimize_influecne_(new_solution_cheaper, max_saving_index)
            all_saving[max_saving_index] = new_improvement2
            all_saving_allocation[max_saving_index] = new_index2
        # print("Finished and the time is: ", time.time() - start_time)
        return all_res, all_solutions

    def fill_gap_(self, levels_token, nondominated_res, nondominated_solutions):
        new_res = nondominated_res.copy()
        new_res.sort(axis=0)
        np.argsort(nondominated_res, axis=0)
        orig_index = np.argsort(nondominated_res, axis=0)
        orig_index_list = orig_index[:, 0].tolist()
        # each row is a point, where the first column is x and second column is y, calculate the Euclidean difference between each two adjacent points
        segdists = []
        for i in range(len(new_res) - 1):
            segdists.append(np.sqrt((new_res[i] - new_res[i + 1]) ** 2).sum())
        # get the index of the largest difference
        avg_diff = np.mean(segdists)
        max_diff = np.max(segdists)
        if max_diff > avg_diff * 5:
            index_list = heapq.nlargest(10, range(len(segdists)), segdists.__getitem__)
            for index in index_list:
                index_a1 = index
                index_a2 = index + 1
                gap_res1, gap_solutions1 = self.search_the_gap(nondominated_solutions[orig_index_list[index_a1]],
                                                               nondominated_solutions[orig_index_list[index_a2]])
                nondominated_res = np.vstack((nondominated_res, gap_res1))
                nondominated_solutions = np.vstack((nondominated_solutions, gap_solutions1))
            pareto_efficient_mask = self.is_pareto_(nondominated_res.copy())
            nondominated_solutions = np.array(nondominated_solutions)[pareto_efficient_mask]
            nondominated_res = nondominated_res[pareto_efficient_mask]
        return nondominated_res, nondominated_solutions

    def fill_gap_multiprocessing(self, nondominated_res, nondominated_solutions):
        new_res = nondominated_res.copy()
        new_res.sort(axis=0)
        np.argsort(nondominated_res, axis=0)
        orig_index = np.argsort(nondominated_res, axis=0)
        orig_index_list = orig_index[:, 0].tolist()
        # each row is a point, where the first column is x and second column is y, calculate the Euclidean difference between each two adjacent points
        segdists = []
        for i in range(len(new_res) - 1):
            segdists.append(np.sqrt((new_res[i] - new_res[i + 1]) ** 2).sum())
        max_diff = np.max(segdists)
        avg_diff = np.mean(segdists)
        # gap_num = int(max_diff / avg_diff)
        # print("The gap number is: ", gap_num)
        index_list = heapq.nlargest(self.sampling_sum, range(len(segdists)), segdists.__getitem__)
        gaps_to_process = [
            (nondominated_solutions[orig_index_list[index]], nondominated_solutions[orig_index_list[index + 1]]) for
            index in index_list]

        n_cpu = os.cpu_count()
        with mp.Pool(processes=n_cpu) as pool:
            results = pool.starmap(self.search_the_gap, [(gap[0], gap[1]) for gap in gaps_to_process])

        for result in results:
            gap_res1, gap_solutions1 = result
            nondominated_res = np.vstack((nondominated_res, gap_res1))
            nondominated_solutions = np.vstack((nondominated_solutions, gap_solutions1))

        pareto_efficient_mask = self.is_pareto_(nondominated_res.copy())
        nondominated_solutions = np.array(nondominated_solutions)[pareto_efficient_mask]
        nondominated_res = nondominated_res[pareto_efficient_mask]
        pool.close()
        pool.join()
        return nondominated_res, nondominated_solutions

    def get_true_accuracy_obj_(self, pareto_solutions):
        parsed_num_list = []
        true_accuracy = []
        for solution in tqdm(np.array(pareto_solutions)):
            res = 0
            for i in range(len(solution)):
                res += self.df_true_accuracy.iloc[i, solution[i]]
            parsed_num_list.append(res)
            true_accuracy.append(res / len(solution))
        return true_accuracy

    def run(self):
        start_time = time.time()
        print("Start local search!")
        s_cheapest, s_high_accuracy = self.get_exreme_solution_()

        # initial_solution = self.initialization_()
        initial_solution = self.initialization2_()
        s1 = self.ls_fitness_function_(s_cheapest)
        s2 = self.ls_fitness_function_(s_high_accuracy)
        ls_solutions = [s_cheapest, s_high_accuracy]
        nondominated_res = np.vstack([s1, s2])

        n_cpu = os.cpu_count()
        with mp.Pool(processes=n_cpu) as pool:
            results = pool.starmap(self.optimise_to_nondominated_, [(solution,) for solution in initial_solution])

        # Combine the results
        for result in results:
            np_new_solutions_res_set, new_solutions_set = result
            nondominated_res = np.vstack((nondominated_res, np_new_solutions_res_set))
            ls_solutions.extend(new_solutions_set)
        pool.close()
        pool.join()
        # # Filter for Pareto-efficient solutions
        pareto_efficient_mask = self.is_pareto_(nondominated_res.copy())
        nondominated_solutions = np.array(ls_solutions)[pareto_efficient_mask]
        nondominated_res = nondominated_res[pareto_efficient_mask]
        elapsed_time = time.time() - start_time

        print("Local search finished and the searchiing time is: ", elapsed_time)

        lspre_res = pd.DataFrame(nondominated_res, columns=['cost', 'expected_accuracy'])
        lspre_res['time'] = elapsed_time
        lspre_res['true_accuracy'] = self.get_true_accuracy_obj_(nondominated_solutions)

        return lspre_res, nondominated_solutions


