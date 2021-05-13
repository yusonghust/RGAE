#-*-coding:utf-8-*-

from config import Config
import scipy.io as sio
import time
import copy
from graph import Graph
from model.mvdne_new import MVDNE
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='tensorflow')


if __name__ == "__main__":
    embfile = "/home/songyu/wl/embfile"
    embeddings = []
    with open(embfile, 'r') as f:
        for line in f:
            embedding = line.split()[1:]
            embeddings.append([int(i) for i in embedding])
    f.close()
    emb_reduction(embeddings)
