from sklearn.cluster import KMeans
import numpy as np
import logging
import sys
import os
import warnings
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

def Cluster(inputs,cfg,logger):
    index = [cfg.look_up[i] for i in cfg.node_list]
    print(type(cfg.node_label))
    Y = [cfg.node_label[i] for i in cfg.node_list]
    X = inputs[index]
    y_true = Y

    cluster_number = len(set(Y))
    logger.info('cluster number = {:d}'.format(cluster_number))

    for i in range(5):
        kmeans = KMeans(n_clusters = cluster_number, random_state = i, n_jobs = 10, max_iter = 500).fit(X)
        y_pred = kmeans.labels_
        for average_method in ['min','geometric','arithmetic','max']:
            nmi_score = normalized_mutual_info_score(y_true,y_pred,average_method = average_method)
            ami_score = adjusted_mutual_info_score(y_true,y_pred,average_method)
            logger.info('current seed = {:d} average_method = {} nmi = {:.3f} ami = {:.3f}'.format(i,average_method,nmi_score,ami_score))

