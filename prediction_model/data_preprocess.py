import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import pairwise_kernels
from prediction_model.prediction import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform


def preprocess_features(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def select_features(X, y, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    return X_selected, selected_indices

def apply_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    return labels

def visualize_features(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.title('Log Messages Visualization')
    plt.show()

def MMD(Xs, Xt):
    """
    Maximum Mean Discrepancy

    Calculate the distance between source Xs and target Xt
    """
    ns = Xs.shape[1]
    nt = Xt.shape[1]

    # Calculate kernel matrix
    sigma = np.median(pdist(np.vstack((Xs, Xt)).T))
    Ks = constructKernel(Xs.T, None, sigma)
    Kt = constructKernel(Xt.T, None, sigma)
    Kst = constructKernel(Xs.T, Xt.T, sigma)

    c_Ks = 1 / (ns ** 2)
    c_Kt = 1 / (nt ** 2)
    c_Kst = 2 / (ns * nt)

    dist = np.sum(c_Ks * Ks) + np.sum(c_Kt * Kt) - np.sum(c_Kst * Kst)

    return dist


def constructKernel(X, Y, sigma):
    """
    Construct kernel matrix
    """
    nx = X.shape[0]
    sqX = np.sum(X ** 2, axis=1)

    if Y is None:
        Q = np.tile(sqX, (nx, 1))
        D = Q + Q.T - 2 * np.dot(X, X.T)
    else:
        ny = Y.shape[0]
        sqY = np.sum(Y ** 2, axis=1)
        Q = np.tile(sqX, (ny, 1)).T
        R = np.tile(sqY, (nx, 1))
        D = Q + R - 2 * np.dot(X, Y.T)

    K = np.exp(-D / (2 * sigma ** 2))

    return K


def maximum_mean_discrepancy(X, Y, kernel='rbf', gamma=None):
    XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)

    nx = XX.shape[0]
    ny = YY.shape[0]

    mmd = np.sum(XX) / (nx * (nx - 1)) + np.sum(YY) / (ny * (ny - 1)) - 2 * np.sum(XY) / (nx * ny)
    return mmd

def CORR(Xs, Ys, Xt):
    dim = Xs.shape[0]
    dist = []

    for i in range(dim):
        d = [spearmanr(Xs[i, :], Ys[j, :])[0] for j in range(Ys.shape[0])]
        dist.append(d)

    dist = np.array(dist)
    dist[np.isnan(dist)] = 0

    idx = np.argsort(dist)[-3:]
    q = Xs[idx, :]
    r = Xt[idx, :]

    Q = np.zeros((q.shape[0], 1))
    for i in range(q.shape[0]):
        corr_sum = 0
        for j in range(q.shape[0]):
            if i != j:
                corr_coefficient = spearmanr(q[i, :], q[j, :])[0]  # Get correlation coefficient
                corr_sum += corr_coefficient  # Accumulate correlation coefficient
        Q[i] = corr_sum

    R = np.zeros((r.shape[0], 1))
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            if i != j:
                R[i] = spearmanr(r[i, :], r[j, :])[0]

    Dist = squareform(pdist(np.vstack([Q.T, R.T]), metric='euclidean'))

    return Dist
def calculate_similarity_matrix(all_data, system_list, method):
    def process_row(row):
        x = log_features_extraction(row['content'])  # Assuming 'query' is the input feature
        x = preprocessing.scale(x)
        y = [row[model] for model in model_list]
        return x, y

    similarity_matrix = pd.DataFrame(index=system_list, columns=system_list)

    for i in range(len(system_list)):
        train_sys = system_list[i]
        train_data = all_data[all_data['dataset'] == train_sys]
        train_x, train_y = zip(*train_data.apply(process_row, axis=1))

        for j in range(len(system_list)):
            # if i == j:
            #     continue  # Skip when train_sys and test_sys are the same

            test_sys = system_list[j]
            test_data = all_data[all_data['dataset'] == test_sys]
            test_x, test_y = zip(*test_data.apply(process_row, axis=1))

            matrix1 = np.array(train_x)
            matrix2 = np.array(test_x)

            if method == "Frobenius":
                value = np.linalg.norm(matrix1 - matrix2, ord='fro')
            elif method == "Cosine":
                value = 1 - cosine(matrix1.flatten(), matrix2.flatten())
            elif method == "Pearson":
                value, _ = pearsonr(matrix1.flatten(), matrix2.flatten())
            elif method == "Jaccard":
                binary_matrix1 = matrix1 > 0.5
                binary_matrix2 = matrix2 > 0.5
                intersection = np.logical_and(binary_matrix1, binary_matrix2)
                union = np.logical_or(binary_matrix1, binary_matrix2)
                value = intersection.sum() / union.sum()
            elif method == "Spearman":
                value, _ = spearmanr(matrix1.flatten(), matrix2.flatten())
            elif method == "MMD":
                value = maximum_mean_discrepancy(matrix1, matrix2)
            elif method == "CORR":
                Xs = np.array(train_x).T
                Ys = np.array(train_y).T
                Xt = np.array(test_x)
                value = CORR(Xs, Ys, Xt)

            similarity_matrix.iloc[i, j] = value

    return similarity_matrix

if __name__ == '__main__':
    system_list = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier',
                   'HPC', 'Zookeeper', 'Mac', 'Hadoop', 'Android', 'Windows', 'Apache',
                   'Thunderbird', 'Spark']

    # method = "Jaccard"  # "Frobenius", "Cosine", "Pearson", "Jaccard", "Spearman", "MMD"
    methods = ["CORR"] # ["Frobenius", "Cosine", "Pearson", "Jaccard", "Spearman", "MMD"]

    # model_list = ["Mixtral_8x7B", "llama2_7b", "llama2_13b", "llama2_70b", "Yi_34B", "Yi_6B", "j2_mid", "j2_ultra",
    #               "gpt1"]  # ,

    model_list = ["gpt0", "gpt0_enhance", "gpt0_simple", "gpt1", "gpt2", "gpt4"]


    save_dir = f"../res"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Calculate similarity matrix using Maximum Mean Discrepancy
    all_data = pd.read_csv(f"../dataset/new_log_parsing.csv", index_col=0)
    for model in model_list:
        all_data[model] = all_data['ref_answer'] == all_data[f'{model}_answer']

    for method in methods:
        similarity_matrix = calculate_similarity_matrix(all_data, system_list, method)
        similarity_matrix.to_csv(f"{save_dir}/similarity_{method}.csv")


