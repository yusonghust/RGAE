#-*-coding:utf-8-*-
import configparser
from typing import List, Any


class Config(object):

    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))
        self.views_file = conf.get("Graph_Data", "views_file")
        self.nodes_file = conf.get("Graph_Data", "nodes_file")
        self.edge_label_file = conf.get("Graph_Data", "edge_label_file")
        self.views_num = conf.get("Graph_Data", "views_num")
        self.data_name = conf.get("Graph_Data", "data_name")
        self.multilabel = bool(int(conf.get("Graph_Data", "multilabel")))
        if conf.has_option("Graph_Data", "node_label_file"):
            self.node_label_file = conf.get("Graph_Data", "node_label_file")
        if conf.has_option("Output", "check_link_prediction"):
            self.check_link_prediction = conf.get("Output", "check_link_prediction")
        if conf.has_option("Output", "check_classification"):
            self.check_classification = conf.get("Output", "check_classification")

        self.views = []
        self.adj_mat = dict()
        self.look_up = dict()
        self.node_label = []
        self.node_list = []
        self.src_node = dict()
        self.dst_node = dict()
        self.edge_label = dict()

        ## hyperparameter
        self.struct = [int(i) for i in conf.get("Model_Setup", "struct").split(',')]
        self.learning_rate = conf.get("Model_Setup", "learning_rate")
        self.smothing = conf.get("Model_Setup", "smothing")
        self.clf_ratio = [float(i) for i in conf.get("Model_Setup", "clf_ratio").strip().split(',')]
        self.view_weight = [float(i) for i in conf.get("Model_Setup", "view_weight").split(',')]
        self.split_num = conf.get("Model_Setup", "split_num")
        self.alpha = conf.get("Model_Setup", "alpha")
        self.beta = conf.get("Model_Setup", "beta")
        self.gamma = [float(i) for i in conf.get("Model_Setup", "gamma").split(',')]
        self.reg = conf.get("Model_Setup", "reg")
        self.recon = conf.get("Model_Setup", "recon")
        self.pseudo = conf.get("Model_Setup", "pseudo")
        self.epochs_limit = conf.get("Model_Setup", "epochs_limit")
        self.display = conf.get("Model_Setup", "display")