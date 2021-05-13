# -*- coding: UTF-8 -*-

import networkx as nx
import numpy as np
import pandas as pd
import os

os.chdir(os.getcwd())

def build_graph(edge_file_path):
    '''
    Reads the input network using networkx.
    '''
    G = nx.read_edgelist(edge_file_path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())

    G = G.to_undirected()
    return G

def write_edgelist(path,network):
    k = 0
    with open(path,'w') as f:
        for edge in network.edges.data('weight'):
            ls = str(edge[0]) + ' ' + str(edge[1]) + ' ' + str(edge[2]) + '\n'
            f.write(ls)
            ls = str(edge[1]) + ' ' + str(edge[0]) + ' ' + str(edge[2]) + '\n'
            f.write(ls)
            k = k+2
    print('edge number is',k)
    f.close()

social_network = build_graph("data/flickr/social.edgelist")
tag_network = build_graph("data/flickr/tag.edgelist")

node_list = []
with open("data/flickr/nodes.txt", 'r') as f:
    for line in f:
        ls = line.strip().split()
        node_list.append(int(ls[0]))
f.close()
node_cnt = len(node_list)

social_node = node_list
tag_node = node_list

while(1):
    social_network = nx.subgraph(social_network, node_list)
    tag_network = nx.subgraph(tag_network, node_list)

    social_node = [n for n in list(social_network.nodes()) if social_network.degree(n)>8]
    tag_node = [n for n in list(tag_network.nodes()) if tag_network.degree(n)>8]

    node_list = list(set(social_node) & set(tag_node))
    if len(node_list) == node_cnt:
        break;
    else:
        node_cnt = len(node_list)
    print('current node number is', node_cnt)
print('final node number is', len(node_list))

social_sub = "data/flickr/social_sub.edgelist"
tag_sub = "data/flickr/tag_sub.edgelist"
node_sub = "data/flickr/nodes_sub.txt"

if os.path.exists(social_sub):
    os.remove(social_sub)
if os.path.exists(tag_sub):
    os.remove(tag_sub)

write_edgelist(social_sub, social_network)
write_edgelist(tag_sub, tag_network)

with open(node_sub, 'w') as f:
    k = 0
    for n in node_list:
        ls = str(n) + '\n'
        f.write(ls)
        k += 1
assert k == len(node_list)
f.close()