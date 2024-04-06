import pandas as pd
import numpy as np
import pickle
import string

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

from prediction_model.TCA import TCA
from prediction_model.help import *
from prediction_model.CORAL import CORAL

def get_avg_accuracy(df_accuracy, model_list):
    return df_accuracy[model_list].mean()
def log_features_extraction(log):
    features = []
    # number of tokens
    features.append(len(log.split()))
    # number of unique tokens
    features.append(len(set(log.split())))
    # number of characters
    features.append(len(log))
    # number of unique characters
    features.append(len(set(log)))
    # number of digits
    features.append(sum(c.isdigit() for c in log))
    # number of letters
    features.append(sum(c.isalpha() for c in log))
    # number of punctuations
    features.append(sum(c in string.punctuation for c in log))
    # average number of characters per token
    features.append(features[2] / features[0])
    # average number of characters per unique token
    features.append(features[2] / features[1])
    # average number of digits per token
    features.append(features[4] / features[0])
    # average number of punctuations per token
    features.append(features[6] / features[0])
    # max length of token
    features.append(max(len(token) for token in log.split()))
    # min length of token
    features.append(min(len(token) for token in log.split()))
    # max punctuation length of token
    features.append(max(sum(c in string.punctuation for c in token) for token in log.split()))
    # min punctuation length of token
    features.append(min(sum(c in string.punctuation for c in token) for token in log.split()))
    # max digit length of token
    features.append(max(sum(c.isdigit() for c in token) for token in log.split()))
    # min digit length of token
    features.append(min(sum(c.isdigit() for c in token) for token in log.split()))
    return features

def get_cost_(job_data, model_list):
    cost_data = pd.DataFrame()
    for model in model_list:
        cost_column = f'{model}_cost'
        cost_data[model] = job_data[cost_column]
    return cost_data

def get_accuracy_(job_data, model_list):
    accuracy_data = pd.DataFrame()
    for model in model_list:
        cost_column = f'{model}'
        accuracy_data[model] = job_data[cost_column]
    return accuracy_data

def log_preprocess(data_dir, model_list, train_sys, test_sys):

    def process_row(row):
        x = log_features_extraction(row['content'])  # Assuming 'query' is the input feature
        x = preprocessing.scale(x)
        y = [row[model] for model in model_list]
        return x, y

    all_data = pd.read_csv(data_dir, index_col=0)
    for model in model_list:
        all_data[model] = all_data['ref_answer'] == all_data[f'{model}_answer']

    train_data = all_data[all_data['dataset'] == train_sys]
    train_x, train_y = zip(*train_data.apply(process_row, axis=1))

    test_data = all_data[all_data['dataset'] == test_sys]
    test_x, test_y = zip(*test_data.apply(process_row, axis=1))

    model_num = len(model_list)

    # conducting transfer learning for cross-project assignment, e.g., transfer component analysis
    # d = int(np.ceil(0.15*np.array(test_x).shape[1])) # projective dimension
    # using PCA to determine projective dimension
    # pca = PCA(n_components='mle', svd_solver='full')
    # pca = PCA()
    # pca.fit(test_x)
    # pert = np.cumsum(pca.singular_values_) / np.sum(pca.singular_values_)
    # d = np.min(np.where(pert >= 0.95))+1
    # print(d)
    #
    # tca = TCA(kernel_type='linear', dim=d, lamb=1, gamma=1)
    # train_x, test_x = tca.fit(np.array(train_x), np.array(test_x))
    #
    # train_x = train_x.real
    # test_x = test_x.real

    # CORAL
    coral = CORAL()
    Xs_new = coral.fit(np.array(train_x), np.array(test_x))

    clf = MultiOutputClassifier(estimator=XGBClassifier(n_jobs=-1, max_depth=10, n_estimators=1000))
    clf.fit(Xs_new, train_y)

    y_pred_accuracy = clf.predict_proba(test_x)
    y_pred = clf.predict(test_x)

    complexity = [[y_pred_accuracy[j][i][1] for j in range(model_num)] for i in range(len(y_pred_accuracy[0]))]
    df_pre_accuracy = pd.DataFrame(complexity, columns=model_list)

    print(classification_report(np.array(test_y), y_pred, digits=3, target_names=model_list))
    print('Accuracy Score: ', accuracy_score(np.array(test_y), y_pred))

    df_cost = get_cost_(test_data, model_list)
    df_true_accuracy = get_accuracy_(test_data, model_list)

    confusion_matrices = multilabel_confusion_matrix(np.array(test_y), y_pred)
    label_accuracy = {}
    for i, label in enumerate(model_list):
        cm = confusion_matrices[i]
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        label_accuracy[label] = accuracy
    df_label_accuracy = pd.DataFrame.from_dict(label_accuracy, orient='index', columns=['Accuracy'])

    return df_pre_accuracy, df_true_accuracy, df_cost, df_label_accuracy



