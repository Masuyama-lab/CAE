#
# Copyright (c) 2022 Naoki Masuyama (masuyama@omu.ac.jp)
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
import numpy as np
import pandas as pd
import networkx as nx
from numpy.random import normal

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from cae import ClassifierCAE, ClusterCAE





print('Start...')


# generate data
# np.random.seed(0)
# n = 10000
# sigma = 0.07
# c = 10 * np.random.rand(n) - 5
# data = np.array([[normal(c[i], sigma), normal(np.sin(c[i]), sigma)] for i in range(len(c))])
# target = np.full(len(data), 0)
# X_train, X_test, Y_train, Y_test = train_test_split(data, target, random_state=0, test_size=0.1)

# 2Ddataset6C_90000 -------------------
D = pd.read_csv('../Dataset/Artificial/2Ddataset6C_90000_withLabel.data', header=None)
D = D.to_numpy()
np.random.seed(0)
np.random.shuffle(D)
data = D[:, 0:D.shape[1] - 1]
target = D[:, D.shape[1] - 1]

X_train, X_test, Y_train, Y_test = train_test_split(data, target, random_state=0, test_size=0.1)

NR = 0.1  # Noise rate
if NR > 0.0:
    np.random.seed(0)
    noiseX = np.random.rand(int(np.round(X_train.shape[0] * NR)), 2)
    noiseY = np.full(len(noiseX), np.max(Y_train) + 2)
    X_train = np.concatenate([X_train[0:int(X_train.shape[0] * (1 - NR))], noiseX])
    Y_train = np.concatenate([Y_train[0:int(Y_train.shape[0] * (1 - NR))], noiseY])
    p = np.random.permutation(len(X_train))
    X_train = X_train[p]
    Y_train = Y_train[p]
# ------------------------------------


# ---------------------------------------------------------------
# CAE for Clustering (train: X_train, test: X_train)
# ---------------------------------------------------------------
# Example fit, and predict
CAE_cluster = ClusterCAE()
CAE_cluster.fit(X_train)
y_pred = CAE_cluster.predict(X_train)

print('NMI:', normalized_mutual_info_score(Y_train, y_pred))
print('ARI:', adjusted_rand_score(Y_train, y_pred))
print('#.Nodes:', len(CAE_cluster.G_.nodes()))
print('#.Clusters:', np.max(list(nx.get_node_attributes(CAE_cluster.G_, 'cluster').values())) + 1)
print('Finished')

# The clustering result of 2D synthetic data is shown by a figure.
CAE_cluster.plotting_net(X_train)


# # Example for fit_predict
# CAE_cluster1 = ClusterCAE()
# y_pred = CAE_cluster1.fit_predict(X_train)
# print('NMI:', normalized_mutual_info_score(Y_train, y_pred))
# print('ARI:', adjusted_rand_score(Y_train, y_pred))
# print('#.Nodes:', len(CAE_cluster1.G_.nodes()))
# print('#.Clusters:', np.max(list(nx.get_node_attributes(CAE_cluster1.G_, 'cluster').values())) + 1)
# print('Finished')
#
# CAE_cluster.plotting_net(X_train)



# ---------------------------------------------------------------
# CAE for Classification  (train: x_train, test: x_test)
# Generated clusters are used as a classifier.
# Label information is not used during training.
# The result is shown by metric scores.
# ---------------------------------------------------------------

def load_openml_dataset(data_name):
    data, tmp_target = fetch_openml(name=data_name, return_X_y=True, as_frame=False, parser='auto')
    # label encoding (A->1, B->2,...)
    df = pd.DataFrame(tmp_target, columns=['label'])
    le = LabelEncoder()
    df['encoded'] = le.fit_transform(df['label'])
    tmp_target = np.array(df['encoded'])
    target = np.array(tmp_target, dtype='int')
    num_classes = np.unique(target).shape[0]
    return data, target, num_classes


data_name = 'iris'

data, target, num_classes = load_openml_dataset(data_name)

# Algorithm
# mode = 'with_label'  # The class of each cluster is determined by label information, then calculating Acc and f1 score.
mode = 'without_label'  # Perform clustering-based evaluations, such as NMI and ARI.
CAE = ClassifierCAE(mode_=mode)

# Evaluation Metrics
def accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)
def f1_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='macro')
def nmi(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)
def ari(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)

score_funcs = {
    'accuracy': make_scorer(accuracy),
    'macro_f1_score': make_scorer(f1_score),
    'nmi': make_scorer(nmi),
    'ari': make_scorer(ari),
    }


# Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=0)
cv_results = cross_validate(estimator=CAE, X=data, y=target, scoring=score_funcs, return_estimator=True, cv=kf, n_jobs=-1)

# Averaged results
n_nodes = [cv_results['estimator'][k].G_.number_of_nodes() for k in range(10)]
n_clusters = [cv_results['estimator'][k].n_clusters_ for k in range(10)]
print('# of Nodes: ', np.mean(n_nodes))
print('# of Clusters: ', np.mean(n_clusters))
print('fit time: ', np.mean(cv_results['fit_time']))
if mode == 'with_label':  # The class of each cluster is determined by label information.
    print('Accuracy:', np.mean(cv_results['test_accuracy']))
    print('macro F1:', np.mean(cv_results['test_macro_f1_score']))
else:  # Perform clustering-based evaluations.
    print('NMI:', np.mean(cv_results['test_nmi']))
    print('ARI:', np.mean(cv_results['test_ari']))
print('Finished')

