# -*- coding: UTF-8 -*-

import networkx as nx
import numpy as np
import pandas as pd
import os

# node_list = []
# with open("data/flickr/nodes_sub.txt", 'r') as f:
#     for line in f:
#         ls = line.strip().split()
#         node_list.append(int(ls[0]))
# f.close()
#
# fp=open("data/flickr/flickr_label_sub.txt","a",encoding="utf-8")
# with open("data/flickr/flickr_label.txt", 'r') as f:
#     for line in f:
#         ls = int(line.strip().split()[0])
#         if ls in node_list:
#             fp.write(line)
# f.close()
# fp.close()

i=0
with open("data/flickr/flickr_label_sub.txt", 'r') as f:
    for line in f:
        i+=1
print(i)

