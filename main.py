#-*-coding:utf-8-*-

from config import Config
import scipy.io as sio
import time
import copy
from graph import Graph
from model.mvdne import MVDNE
from utils.utils import *
from optparse import OptionParser
import os
import logging
from log import setup_logger
from cluster import Cluster
from classifier import Classifier
from linkpred import predictor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='tensorflow')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    parser = OptionParser()
    parser.add_option("-c", dest = "config_file", action = "store", metavar = "CONFIG FILE")
    options, _ = parser.parse_args()
    if options.config_file == None:
        raise IOError("no config file specified")

    config = Config(options.config_file)

    time_now = setup_logger('mvdne', 'mvdne_' + config.data_name,
                            level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('mvdne')

    with open(config.views_file, 'r') as f:
        for line in f:
            ls = line.strip().split()
            edge_file = ls[0]
            directed = bool(ls[1])
            weighted = bool(ls[2])
            config.views.append(Graph(edge_file, weighted, directed))
    f.close()

    with open(config.nodes_file, 'r') as f:
        i = 0
        for line in f:
            ls = line.strip().split()
            config.node_list.append(int(ls[0]))
            config.look_up[int(ls[0])] = i
            i += 1
    f.close()

    config.struct[0] = len(config.node_list)

    for i in range(int(config.views_num)):
        config.adj_mat["view_%d"%i] = preprocess_data(config, i)
    print(len(config.adj_mat))

    # config.src_node, config.dst_node, config.edge_label = preprocess_edge_labels(config)

    if os.path.exists(config.node_label_file):
        config.node_label = config.views[0].read_node_labels(config.multilabel, config.node_label_file)


    model = MVDNE(config)
    model.do_variables_init()

    epochs = 0
    while (True):
        ret,simi_loss,rec_loss,diff_loss,reg_loss,cls_loss = model.fit()
        logger.info('|Epoch = {:d} total loss = {:.5f} simi_loss = {:.5f} rec_loss = {:.5f} diff_loss = {:.5f} reg_loss = {:.5f} cls_loss = {:.5f}'.format(epochs,
            ret,simi_loss,rec_loss,diff_loss,reg_loss,cls_loss))
        if (epochs == int(config.epochs_limit)):
            print("exceed epochs limit terminating")
            break
        epochs += 1

    embedding = model.get_embedding()
    #embedding = np.nan_to_num(embedding)
    # embedding = np.clip(np.nan_to_num(embedding), -1e16, 1e16)
    # emblist = model.get_embedding()
    print('embedding shape is',embedding.shape)

    # Cluster(embedding, config, logger)
    Classifier(embedding, config, logger)
    # model.sess.close()
    # reuse = True
        # tf.reset_default_graph()
    # Link = predictor(embedding,config.edge_label_file,config.look_up,0.1)
    # Link.evaluate()
    # Link = predictor(embedding, config.edge_label_file, config.look_up, 0.3)
    # Link.evaluate()
    # Link = predictor(embedding, config.edge_label_file, config.look_up, 0.5)
    # Link.evaluate()
    # embedding = read_embeddings()
    # embedding = np.array(embedding)
    # embs = emb_reduction(embedding)
    # for i in range(4):
    #     save_embeddings(config.node_list, emblist[i], "/home/songyu/wl/embfile"+str(i)+".txt")
    # save_embeddings(config.node_list, embedding)
    # X = [config.look_up[i] for i in config.node_list]
    # Y = [config.node_label[i] for i in config.node_list]
    # plot_embedding(embedding, X, Y)

