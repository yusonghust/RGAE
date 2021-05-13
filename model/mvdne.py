#-*-coding:utf-8-*-
import numpy as np
import tensorflow as tf
import time
import copy
import random
import os
from tensorflow.python.ops import variable_scope as vs
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='tensorflow')
init = tf.contrib.layers.xavier_initializer(uniform=False)

class MVDNE:
    def __init__(self, config):
        self.is_variables_init = False
        self.config = config

        # if reuse:  ### 改动部分 ###
        #     vs.get_variable_scope().reuse_variables()
        ######### not running out gpu sources ##########
        # allow_soft_placement=True
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

        self.layers = len(config.struct)
        self.struct = config.struct
        self.views_num = int(config.views_num)

        self.W = {}
        self.b = {}
        self.Y = {}

        struct = self.struct

        self.Y_com = tf.get_variable('Y_com',shape=[struct[0], struct[-1]],initializer = init)
        for i in range(self.layers - 1):
            name = "s_encoder" + str(i)
            self.W[name] = tf.get_variable('w_'+name,shape=[struct[i], struct[i + 1]],initializer = init)
            self.b[name] = tf.Variable(tf.zeros([struct[i + 1]]), name='b_'+name)

        for i in range(self.views_num):
            for j in range(self.layers - 1):
                name = "view" + str(i) + "p_encoder" + str(j)
                self.W[name] = tf.get_variable('w_'+name,shape=[struct[j], struct[j + 1]],initializer = init)
                self.b[name] = tf.Variable(tf.zeros([struct[j + 1]]), name=name)
        struct.reverse()


        self.build_placeholders()
        self.__make_compute_graph()
        self.loss,self.loss_dict = self.__make_loss()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(learning_rate = float(config.learning_rate), global_step = global_step, decay_steps = 10, decay_rate = 0.95)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=global_step)

    def build_placeholders(self):
        self.X = {}
        for i in range(self.views_num):
            self.X["view_%d"%i] = tf.sparse_placeholder(tf.float32,
                shape=tf.constant(self.config.adj_mat["view_%d"%i][2],dtype=tf.int64))

    def __make_compute_graph(self):
        def shared_encoder(X):
            X_init = X
            for i in range(self.layers - 1):
                name = "s_encoder" + str(i)
                if i == 0:
                    X = tf.nn.leaky_relu(tf.sparse_tensor_dense_matmul(X, self.W[name]) + self.b[name])
                else:
                    X = tf.nn.leaky_relu(tf.sparse_tensor_dense_matmul(X_init,tf.matmul(X,self.W[name]) + self.b[name]))
            return X

        def private_encoder(X, view_idx):
            X_init = X
            for i in range(self.layers - 1):
                name = "view" + str(view_idx) + "p_encoder" + str(i)
                if i==0:
                    X = tf.nn.leaky_relu(tf.sparse_tensor_dense_matmul(X, self.W[name]) + self.b[name])
                else:
                    X = tf.nn.leaky_relu(tf.sparse_tensor_dense_matmul(X_init, tf.matmul(X,self.W[name])+ self.b[name]))
            return X

        for i in range(self.views_num):
            name_p = "private_Y" + str(i)
            self.Y[name_p] = private_encoder(self.X["view_%d"%i], i)
            name_s = "shared_Y" + str(i)
            self.Y[name_s] = shared_encoder(self.X["view_%d"%i])

    def __make_loss(self):
        def get_recon_loss():
            loss = 0
            split_per_size = int(len(self.config.node_list)/int(self.config.split_num))
            up_num = len(self.config.node_list)%int(self.config.split_num)
            split_size = [split_per_size for i in range(int(self.config.split_num))]
            for i in range(up_num):
                split_size[i] += 1

            for i in range(self.views_num):
                #with tf.device('/gpu:1'):
                name_p = "private_Y" + str(i)
                Y_split = tf.split(self.Y[name_p], split_size, 0)
                X_split = tf.sparse_split(sp_input=self.X["view_%d"%i], num_split=int(self.config.split_num), axis=0)
                for j in range(int(self.config.split_num)):
                    Y_pred = tf.matmul(Y_split[j],self.Y[name_p],transpose_b=True)
                    Y_true = tf.sparse_to_dense(sparse_indices=X_split[j].indices,
                                                       output_shape=X_split[j].dense_shape,
                                                       sparse_values=X_split[j].values)
                    # Y_true = tf.sparse.to_dense(sp_input=X_split[j])
                    pos = tf.reduce_sum(tf.cast(tf.greater(Y_true,0.0),tf.float32))
                    neg = tf.reduce_sum(tf.cast(tf.equal(Y_true,0.0),tf.float32))
                    pos_ratio = neg/pos
                    loss += self.config.view_weight[i]*tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                        labels = Y_true,
                        logits = Y_pred,
                        pos_weight = pos_ratio))
            return loss

        def cosine_similarity(a,b):
            a = tf.nn.l2_normalize(a,axis=-1)
            b = tf.nn.l2_normalize(b,axis=-1)
            sim = -tf.reduce_sum(tf.multiply(a,b),axis=-1)
            return sim

        def get_simi_loss():
            loss = 0
            view_weight = tf.get_variable('weight', shape = (self.views_num), constraint = lambda x : tf.clip_by_value(x, 0.0, 1.0))
            view_weight = tf.nn.softmax(view_weight,0)
            gamma = self.config.gamma
            view_weight = tf.pow(view_weight,gamma)
            for i in range(self.views_num):
                name = "shared_Y" + str(i)
                # loss += self.config.view_weight[i]*tf.reduce_mean(cosine_similarity(self.Y_com,self.Y[name]))
                # loss += self.config.view_weight[i]*tf.reduce_mean(tf.pow(self.Y_com - self.Y[name],2))
                loss += tf.gather(view_weight,i)*tf.reduce_mean(tf.pow(self.Y_com - self.Y[name],2))
                # loss += self.config.view_weight[i]*tf.reduce_mean(tf.losses.absolute_difference(self.Y_com,self.Y[name],reduction=tf.losses.Reduction.NONE))
                # loss += self.config.view_weight[i]*tf.norm(tf.reduce_mean(tf.multiply(self.Y_com,self.Y[name]),axis=-1))
            return loss

        def get_diff_loss():
            loss = 0
            for i in range(self.views_num):
                name_p = "private_Y" + str(i)
                name_s = "shared_Y" + str(i)
                loss += self.config.view_weight[i]*tf.norm(tf.reduce_mean(tf.multiply(self.Y[name_p],self.Y[name_s]),axis=-1))
            return loss

        loss_dict = {}

        loss_dict['simi_loss'] = get_simi_loss()
        loss_dict['rec_loss']  = get_recon_loss()
        loss_dict['diff_loss'] = get_diff_loss()
        loss_dict['reg_loss']  = tf.constant(0.0,dtype=tf.float32)
        loss_dict['cls_loss']  = tf.constant(0.0,dtype=tf.float32)

        loss = loss_dict['rec_loss'] + 0.5*loss_dict['simi_loss'] + 0.5*loss_dict['diff_loss']
        return loss,loss_dict

    def __get_feed_dict(self):
        feed_dict = {}
        for i in range(self.views_num):
            feed_dict[self.X["view_%d"%i]] = self.config.adj_mat["view_%d"%i]
        return feed_dict

    def do_variables_init(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def fit(self):
        feed_dict = self.__get_feed_dict()
        opt_list = [self.optimizer,self.loss,self.loss_dict['simi_loss'],
            self.loss_dict['rec_loss'],self.loss_dict['diff_loss'],self.loss_dict['reg_loss'],self.loss_dict['cls_loss']]
        _, loss,simi_loss,rec_loss,diff_loss,reg_loss,cls_loss = self.sess.run(opt_list, feed_dict = feed_dict)
        return loss,simi_loss,rec_loss,diff_loss,reg_loss,cls_loss

    def get_embedding(self):
        feed_dict = self.__get_feed_dict()
        candidate = [self.Y_com]
        for i in range(self.views_num):
            name_p = "private_Y" + str(i)
            candidate.append(self.Y[name_p])
        embedding = tf.concat(candidate,axis=1)
        # embedding0 = self.Y["private_Y0"]
        # embedding1 = self.Y["private_Y1"]
        # embedding2 = self.Y["private_Y2"]
        # emb_list = [embedding, embedding0, embedding1, embedding2]
        # emb, emb0, emb1, emb2 = self.sess.run(emb_list, feed_dict = feed_dict)
        # return emb, emb0, emb1, emb2
        return self.sess.run(embedding,feed_dict = feed_dict)


