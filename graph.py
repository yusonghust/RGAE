#-*-coding:utf-8-*-
import scipy.io as sio
import numpy as np
import random
import copy
from utils import *
import networkx as nx

class Graph(object):
    def __init__(self, edge_file, weighted, directed):
        '''        with open(views_file_path, 'r') as f:
            self.views = []
            self.G = self.build_graph()
            for line in f:
                ls = line.strip().split()
                edgelist = ls[0]
                self.directed = bool(ls[1])
                self.weighted = bool(ls[2])
                self.views.append(self.build_graph(edgelist))'''
        self.edge_file = edge_file
        self.weighted = weighted
        self.directed = directed
        self.G = self.build_graph(self.edge_file)
        self.node_list = list(self.G.nodes())
        self.look_up = {}
        self.node_size = 0
        for node in self.node_list:
            self.look_up[node] = self.node_size
            self.node_size += 1
            
        

    def build_graph(self, edge_file_path):
        '''
        Reads the input network using networkx.
        '''
        if self.weighted:
            G = nx.read_edgelist(edge_file_path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(edge_file_path, nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1

        if not self.directed:
            G = G.to_undirected()
        return G

    def read_node_labels(self, multilabel, filename):
        node_label = {}
        with open(filename, 'r') as f:
            for line in f:
                ls = line.strip().split()
                if multilabel == False:
                    node_label[int(ls[0])] = int(ls[1])
                else:
                    node_label[int(ls[0])] = [int(x) for x in ls[1:]]
        f.close()
        return  node_label


    def read_node_file(self, filename):
        self.node_list = []
        self.look_up = dict() #map node to id
        with open(filename, 'r') as f:
            i = 0
            for line in f:
                ls = line.strip().split()
                self.node_list.append(int(ls[0]))
                self.look_up[int(ls[0])] = i
                i += 1
        f.close()

