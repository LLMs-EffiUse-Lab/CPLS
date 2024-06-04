# CPLS
This is the repository for the paper: 

CPLS: Optimizing the Assignment of LLM Queries

Abstract: 
Large Language Models (LLMs) like ChatGPT have gained significant attention because of their impressive capabilities, leading to a dramatic increase in their integration into intelligent software engineering. However, their usage as a service with varying performance and price options presents a challenging trade-off between desired performance and the associated cost. To address this challenge, we propose CPLS, a framework that utilizes transfer learning and local search techniques for assigning intelligent software engineering jobs to LLM-based services. CPLS aims to minimize the total cost of LLM invocations while maximizing the overall accuracy. The framework first leverages knowledge from historical data across different projects to predict the probability of an LLM processing a query correctly. Then, CPLS incorporates problem-specific rules into a local search algorithm to effectively generate Pareto optimal solutions based on the predicted accuracy and cost. 

## 1. Framework
CPLS consists of two main components: cross-project prediction and local-search based optimization. The prediction component utilizes knowledge from historical data across different projects to estimate the probability of an LLM correctly processing a query. With the cost and predicted accuracy, the optimization component selects the most suitable LLM for each query through problem-specific rules.

<p align="center"><img src="figs/framework.jpg" width="800"><br>An overview of CPLS</p>

## 2. Additional Experiments

To demonstrate the generalizability of CPLS, we conducted an additional experiment on text classification. For this experiment, we used the Overruling dataset as the training data to optimize the assignment of LLMs for the Headlines dataset.
The Headline dataset consists of news headlines about gold prices, and the task is to classify each headline into one of four categories: up, down, neutral, or none, based on the price movement mentioned in the headline. An example is given as follows:

_Please determine the price direction (up, down, neutral, or none) in the following news headlines.
Question: Feb. gold gains $10.30, or 0.9%, to settle at $1,162/oz_

(Answer: up)

An example of a job from the Overruling dataset is:

_Context: As such, La. Civ. Code art. 2365 is not applicable to the matter before us, and we specifically overrule this holding in Nash.
Question: Is it overruling?_

(Answer: yes)

By using the Overruling dataset to train CPLS and then applying it to the Headline classification task, we aim to showcase the adaptability of our approach to different domains. The results of this experiment are presented below:

<p align="center"><img src="figs/headlines_acc.png" width="800"><br>The solution with the highest accuracy by all algorithms for LLMs allocation</p>

<p align="center"><img src="figs/headlines.png" width="800"><br>Cost ($) and Savings by CPLS to match the baseline's highest accuracy</p>

<p align="center"><img src="figs/headlines_metrics.png" width="800"><br>Comparisons of Solution Sets from All Algorithms in terms of IGD, $\Delta$ and Time</p>

We present the solution with the highest accuracy and the solution with the lowest cost. Compared to the baselines, CPLS achieves a 2.91-7.21% improvement in accuracy or a 90-98% reduction in cost. Moreover, the metrics used to evaluate the quality of the solution set considering both cost and accuracy, such as the Inverted Generational Distance (IGD) and the delta measure ($\Delta$), show that CPLS outperforms the baselines. 

[//]: # (## 3. Benchmarks)

[//]: # (To evaluate the proposed approach, we conduct extensive experiments on LLM-based log parsing, a typical software maintenance task. )

[//]: # ()
[//]: # (We leverage log data originated from the LogPai benchmark as a study case. LogPai is a comprehensive collection of log data originating from 16 diverse systems)

## 3. Baselines and Parameter Setting
### 3.1 Baselines
CPLS utilizes a heuristic search-based algorithm in optimization. We compare the effectiveness of this algorithm with well-known multi-objective optimization algorithms, including the Non-dominated Sorting Genetic Algorithm (NSGA-\rom{2})^[8], Multi-objective Particle Swarm Optimisation (MOPSO)^[9], and Multi-objective Evolutionary Algorithm with Decomposition (MOEA/D)^[10]. These three algorithms have been extensively studied and have proven to be effective in solving a wide range of multi-objective optimization problems. In addition, three variants of classic algorithms are also compared, including R-NSGA-\rom{2}^[11], SMS-EMOA^[12], and MOEA/D-GEN^[13]. It is important to note that all the evaluated multi-objective optimization algorithms are integrated with the same prediction component as CPLS, to enable a fair comparison of the optimization strategies. 
### 3.2 Parameter Setting
Optuna is a widely used hyperparameter optimization package. To ensure the effectiveness and efficiency of all algorithms, we conduct parameter tuning using Optuna to choose optimal parameter settings. Based on the experiments, the parameters of algorithms are set as follows:

| Algorithm  | Parameter Settings                                                                                                               |
|------------|----------------------------------------------------------------------------------------------------------------------------------|
| CPLS       | initial_solution_size: 87, sampling_sum: 29                                                                                      |
| NSGA-II    | crossover_prob: 0.5169, crossover_eta: 3, mutation_prob: 0.4258, mutation_eta: 27, sampling: 'LHS', selection: 'RandomSelection' |
| R-NSGA-II  | epsilon: 0.6912                                                                                                                  |
| SMS-EMOA   | crossover_prob: 0.8244, crossover_eta: 20, mutation_prob: 0.3149, mutation_eta: 19, sampling: 'FloatRandomSampling'              |
| MOEA/D     | weight_generation: 'grid', decomposition: 'weighted', neighbours: 29                                                             |
| MOEA/D-GEN | weight_generation: 'low discrepancy', decomposition: 'tchebycheff', neighbours: 16                                               |
| MOPSO      | omega: 0.3634, c1: 0.8446, c2: 0.2482, v_coeff: 0.9121                                                                           |

The record of the tunning process is available under `CPLS/parameter_setting/res` directory.
## 4 Results
### 4.1 Metrics 
#### 4.1.1 Evaluating solution performance
When assessing the performance of a single solution, such as submitting all jobs to an individual LLM, a direct comparison of the optimization objectives is feasible. 
- $f_{cost}$: total cost of invoking LLM APIs
- $f_{acc}$: the percentage of jobs processed accurately
#### 4.1.2 Multi-objective optimization evaluation metrics
- Inverted Generational Distance (IGD): The IGD metric is used to measure the distance between the obtained solution set and the Pareto front (reference point set). A lower value of IGD represents a better performance.</p>

- $\Delta$ metric: The $\Delta$ metric assesses the diversity and distribution of solutions across the Pareto front by measuring Euclidean distances between solutions and two extreme solutions.

- Computation time: The time for obtaining the solution set, calculated by minute.</p>

### 4.2 Resutls and Analysis
To verify the comparison, we conduct a statistical test to evaluate the performance of CPLS and the baselines. We use the following statistical tests:

Friedman Test: The Friedman test is a non-parametric statistical test that ranks the algorithms for each dataset separately. It tests the null hypothesis that all algorithms perform equally well. If the null hypothesis is rejected, it means that there are significant differences among the algorithms' performances.

Nemenyi Test: The Nemenyi test is a post-hoc test that is performed after the Friedman test if the null hypothesis is rejected. It is used to determine which specific pairs of algorithms have significant differences in their performance.
####
#### 4.2.1.1 Comparison with the Baselines
<p align="center"><img src="figs/baselines_acc.png" width="800"><br></p>

<p align="center"><img src="figs/baselines_cost.png" width="800"><br></p>

<p align="center"><img src="figs/baselines_metrics.png" width="800"><br></p>

##### 4.2.1.2 Stastical Test

The Friedman test is a non-parametric statistical test used to compare multiple paired samples. The test is based on ranking the data within each block (i.e., each sample) and comparing the average ranks between the different groups. The following table shows the p-values of the Friedman test for the 16 instances on IGD, $\Delta$, Accuracy and Cost are as follows:




<p align="center"><img src="figs/RQ1_Fri.png" width="700"><br></p>

Overall, the Friedman test results for all five datasets show extremely small p-values, indicating strong evidence against the null hypothesis. This suggests that there are significant differences between the groups being compared for each dataset. The results provide compelling evidence to reject the null hypothesis and accept the alternative hypothesis that at least one group differs from the others.


#### 4.2.2.1 Ablation Study
<p align="center"><img src="figs/ablation.png" width="800"><br></p>

<p align="center"><img src="figs/ablation_metric.png" width="500"><br></p>

<p align="center"><img src="figs/ablation_acc.png" width="500"><br></p>

##### 4.2.2.2 Stastical Test

#### 4.2.3 Effect of Hyper-Parameter Settings

<p align="center"><img src="figs/beta.png" width="500"><br></p>

<p align="center"><img src="figs/delta.png" width="500"><br></p>

## 5. Requirements
All the code is available under `CPLS` directory.
### 5.1 Library
1. Python 3.11
2. Pymoo
3. tiktoken

4. ...

To install all libraries:
$ pip install -r requirements.txt

### 5.2 How to run CPLS
$ python exp.py $

### 5.3 Source code
All source code is available under `CPLS` directory.

We used the standard version of NSGA-II, R-NSGA-II and SMS-EMOA implemented in the Pymoo library^[14], and MOPSO and MOEA/D in the Pygmo. 
The source code of the baselines is available under `CPLS/baselines` directory.

The code os the proposed CPLS is available under `CPLS/CPLS` directory.

| script       | Description                                                               |
| ------------ |---------------------------------------------------------------------------|
| `nsga2.py`   | Non-dominated Sorting Genetic Algorithm (NSGA-II)                         |
| `rnsga2.py`  | Reference point based Non-dominated Sorting Genetic Algorithm (R-NSGA-II) |
| `smsemoa.py` | SMS-EMOA                                                                  |
| `moead.py`   | Multi-objective EA with Decomposition (MOEA/D)                            |
| `moeadgen.py`| MOEA/D-GEN                                                                |
| `mopso.py`   | Multi-objective Particle Swarm Optimization (MOPSO)                       |
