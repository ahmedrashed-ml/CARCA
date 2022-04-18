# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x 

"""#UTILS"""

import sys
import copy
import random
import numpy as np
from collections import defaultdict
import pandas as pd
import pickle
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x= pickle.load(f)
    except:
        x = []
    return x

def save_data(data,filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)



def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('./Data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess, cxtdict, cxtsize, negnum=100):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    Auc = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seqcxt = np.zeros([args.maxlen,cxtsize], dtype=np.int32)
        testitemscxt = list()
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        #Cxt
        seqcxt[idx] = cxtdict[(u,valid[u][0])]

        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            #Cxt
            seqcxt[idx] = cxtdict[(u,i)]

            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        testitemscxt.append(cxtdict[(u,test[u][0])])
        for _ in range(negnum):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            testitemscxt.append(cxtdict[(u,test[u][0])])


        predictions = -model.predict(sess, np.ones(args.maxlen)*u, [seq], item_idx, [seqcxt], testitemscxt)
        predictions = predictions[0]
        score = -predictions.copy()  
        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        #if valid_user % 100 == 0:
        #    print ('.',sys.stdout.flush())

        tmpans=0
        count=0
        for j in range(1,len(score)): #sample
            if score[0]>score[j]:
                tmpans+=1
            count+=1       
        tmpans/=float(count)
        Auc+=tmpans

    return NDCG / valid_user, HT / valid_user, Auc / valid_user


def evaluate_valid(model, dataset, args, sess, cxtdict, cxtsize, negnum=100):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    Auc = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seqcxt = np.zeros([args.maxlen,cxtsize], dtype=np.int32)
        testitemscxt = list()
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            #cxt
            seqcxt[idx] = cxtdict[(u,i)]
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        testitemscxt.append(cxtdict[(u,valid[u][0])])
        for _ in range(negnum):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            testitemscxt.append(cxtdict[(u,valid[u][0])])

        predictions = -model.predict(sess, np.ones(args.maxlen)*u, [seq], item_idx, [seqcxt], testitemscxt)
        predictions = predictions[0]
        score = -predictions.copy()  

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        tmpans=0
        count=0
        for j in range(1,len(score)): #sample
            if score[0]>score[j]:
                tmpans+=1
            count+=1       
        tmpans/=float(count)
        Auc+=tmpans
        #if valid_user % 100 == 0:
        #    print ('.',sys.stdout.flush())
    

    return NDCG / valid_user, HT / valid_user, Auc / valid_user



def PreprocessData(filname, DatasetName, sep="\t"):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    df['ts'] = pd.to_datetime(df['st'],unit='s')
    df = df.sort_values(by=['ts'])
    df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'] , df['week'] = zip(*df['ts'].map(lambda x: [x.year,x.month,x.day,x.dayofweek,x.dayofyear,x.week]))
    df['year']-=df['year'].min()
    df['year']/=df['year'].max()
    df['month']/=12
    df['day']/=31
    df['dayofweek']/=7
    df['dayofyear']/=365
    df['week']/=4

    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows() :
      userid = int(row['user'])
      itemid = int(row['item'])

      if userid in UsersDict.keys() :
        UsersDict[userid].append(itemid)
      else :
        UsersDict[userid] = list()
        UsersDict[userid].append(itemid)


      year = row['year'] 
      month = row['month'] 
      day = row['day'] 
      dayofweek = row['dayofweek'] 
      dayofyear = row['dayofyear'] 
      week = row['week'] 
      DATEINFO[(userid,itemid)] = [year, month, day, dayofweek, dayofyear, week]
    '''
    f = open('./Data/%s_pre.txt' % DatasetName, 'w')
    for user in UsersDict.keys():
        for i in UsersDict[user]:
            f.write('%d %d\n' % (user, i))
    f.close()
    '''

    return df, DATEINFO


def get_ItemDataBeauty(itemnum):
    #ItemFeatures = load_data('./Data/Beauty_feat_1.dat')
    ItemFeatures = load_data('./Data/Beauty_feat_cat.dat')
    ItemFeatures = np.vstack((np.zeros(ItemFeatures.shape[1]), ItemFeatures))
    return ItemFeatures

def get_UserDataBeauty(usernum):
    UserFeatures = np.identity(usernum,dtype=np.int8) 
    UserFeatures = np.vstack((np.zeros(UserFeatures.shape[1],dtype=np.int8), UserFeatures))
    return UserFeatures

def PreprocessData_Beauty(filname, DatasetName, sep="\t"):
    col_names = ["user", "item", "ts"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)

    df['ts'] = pd.to_datetime(df['ts'],unit='s')
    df = df.sort_values(by=['ts'])
    df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'] , df['week'] = zip(*df['ts'].map(lambda x: [x.year,x.month,x.day,x.dayofweek,x.dayofyear,x.week]))
    df['year']-=df['year'].min()
    df['year']/=df['year'].max()
    df['month']/=12
    df['day']/=31
    df['dayofweek']/=7
    df['dayofyear']/=365
    df['week']/=4

    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows() :
      userid = int(row['user'])
      itemid = int(row['item'])

      year = row['year'] 
      month = row['month'] 
      day = row['day'] 
      dayofweek = row['dayofweek'] 
      dayofyear = row['dayofyear'] 
      week = row['week'] 
      DATEINFO[(userid,itemid)] = [year, month, day, dayofweek, dayofyear, week]

    return df, DATEINFO  

def get_ItemDataMen(itemnum):
    ItemFeatures = load_data('./Data/Men_imgs.dat')
    ItemFeatures = np.vstack((np.zeros(ItemFeatures.shape[1]), ItemFeatures))
    return ItemFeatures

def get_UserDataMen(usernum):
    UserFeatures = np.identity(usernum,dtype=np.int8) 
    UserFeatures = np.vstack((np.zeros(UserFeatures.shape[1],dtype=np.int8), UserFeatures))
    return UserFeatures

def PreprocessData_Men(filname, DatasetName, sep="\t"):
    col_names = ["user", "item", "ts"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)

    df['ts'] = pd.to_datetime(df['ts'],unit='s')
    df = df.sort_values(by=['ts'])
    df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'] , df['week'] = zip(*df['ts'].map(lambda x: [x.year,x.month,x.day,x.dayofweek,x.dayofyear,x.week]))
    df['year']-=df['year'].min()
    df['year']/=df['year'].max()
    df['month']/=12
    df['day']/=31
    df['dayofweek']/=7
    df['dayofyear']/=365
    df['week']/=4

    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows() :
      userid = int(row['user'])
      itemid = int(row['item'])

      year = row['year'] 
      month = row['month'] 
      day = row['day'] 
      dayofweek = row['dayofweek'] 
      dayofyear = row['dayofyear'] 
      week = row['week'] 
      DATEINFO[(userid,itemid)] = [year, month, day, dayofweek, dayofyear, week]

    return df, DATEINFO 


def PreprocessData_Fashion(filname, DatasetName, sep="\t"):
    col_names = ["user", "item", "ts"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)

    df['ts'] = pd.to_datetime(df['ts'],unit='s')
    df = df.sort_values(by=['ts'])
    df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'] , df['week'] = zip(*df['ts'].map(lambda x: [x.year,x.month,x.day,x.dayofweek,x.dayofyear,x.week]))
    df['year']-=df['year'].min()
    df['year']/=df['year'].max()
    df['month']/=12
    df['day']/=31
    df['dayofweek']/=7
    df['dayofyear']/=365
    df['week']/=4

    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows() :
      userid = int(row['user'])
      itemid = int(row['item'])

      year = row['year'] 
      month = row['month'] 
      day = row['day'] 
      dayofweek = row['dayofweek'] 
      dayofyear = row['dayofyear'] 
      week = row['week'] 
      DATEINFO[(userid,itemid)] = [year, month, day, dayofweek, dayofyear, week]

    return df, DATEINFO 

def get_ItemDataFashion(itemnum):
    ItemFeatures = load_data('./Data/Fashion_imgs.dat')
    ItemFeatures = np.vstack((np.zeros(ItemFeatures.shape[1]), ItemFeatures))
    return ItemFeatures

def get_UserDataFashion(usernum):
    UserFeatures = np.identity(usernum,dtype=np.int8) 
    UserFeatures = np.vstack((np.zeros(UserFeatures.shape[1],dtype=np.int8), UserFeatures))
    return UserFeatures

def PreprocessData_Games(filname, DatasetName, sep="\t"):
    col_names = ["user", "item", "ts"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)

    df['ts'] = pd.to_datetime(df['ts'],unit='s')
    df = df.sort_values(by=['ts'])
    df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'] , df['week'] = zip(*df['ts'].map(lambda x: [x.year,x.month,x.day,x.dayofweek,x.dayofyear,x.week]))
    df['year']-=df['year'].min()
    df['year']/=df['year'].max()
    df['month']/=12
    df['day']/=31
    df['dayofweek']/=7
    df['dayofyear']/=365
    df['week']/=4

    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows() :
      userid = int(row['user'])
      itemid = int(row['item'])

      year = row['year'] 
      month = row['month'] 
      day = row['day'] 
      dayofweek = row['dayofweek'] 
      dayofyear = row['dayofyear'] 
      week = row['week'] 
      DATEINFO[(userid,itemid)] = [year, month, day, dayofweek, dayofyear, week]

    return df, DATEINFO 

def get_ItemDataGames(itemnum):
    ItemFeatures = load_data('./Data/Video_Games_feat.dat')
    ItemFeatures = np.vstack((np.zeros(ItemFeatures.shape[1]), ItemFeatures))
    return ItemFeatures

"""#Sampler"""

import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, cxtdict, cxtsize, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        ###CXT
        seqcxt = np.zeros([maxlen,cxtsize], dtype=np.float32)
        poscxt = np.zeros([maxlen,cxtsize], dtype=np.float32)
        negcxt = np.zeros([maxlen,cxtsize], dtype=np.float32)
        ###


        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):

            seq[idx] = i
            pos[idx] = nxt
            neg_i = 0
            if nxt != 0: 
              neg_i = random_neq(1, itemnum + 1, ts)
              neg[idx] = neg_i
            ###CXT
            seqcxt[idx] = cxtdict[(user,i)]
            poscxt[idx] = cxtdict[(user,nxt)]
            negcxt[idx] = cxtdict[(user,nxt)]
            ###

            nxt = i
            idx -= 1
            if idx == -1: break

        return (np.ones(maxlen)*user, seq, pos, neg, seqcxt, poscxt, negcxt)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, cxtdict, cxtsize, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      cxtdict,
                                                      cxtsize,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

"""#Modules"""

# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''


import tensorflow as tf
import numpy as np


def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              l2_reg=0.0,
              scope="embedding", 
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       #initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
    if with_t: return outputs,lookup_table
    else: return outputs


def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None,
                        res=True,
                        with_qk=False):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.leaky_relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.leaky_relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.leaky_relu) # (N, T_k, C)
        #Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        #K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        #V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1)) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        if res:
          outputs *= queries
              
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
 
    if with_qk: return Q,K
    else: return outputs

def multihead_attention2(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None,
                        res=True,
                        with_qk=False):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.leaky_relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.leaky_relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.leaky_relu) # (N, T_k, C)
        #Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        #K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        #V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1)) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        if res:
          outputs *= queries
              
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
 
    if with_qk: return Q,K
    else: return outputs

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.leaky_relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        #outputs = normalize(outputs)
    
    return outputs

"""#Model"""

class Model():
    def __init__(self, usernum, itemnum, args, ItemFeatures=None, UserFeatures=None, cxt_size=None, reuse=None , use_res=False):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.seq_cxt = tf.placeholder(tf.float32, shape=(None, args.maxlen, cxt_size))
        self.pos_cxt = tf.placeholder(tf.float32, shape=(None, args.maxlen, cxt_size))
        self.neg_cxt = tf.placeholder(tf.float32, shape=(None, args.maxlen, cxt_size))

        self.ItemFeats = tf.constant(ItemFeatures,name="ItemFeats", shape=[itemnum + 1, ItemFeatures.shape[1]],dtype=tf.float32)
        #self.UserFeats = tf.constant(UserFeatures,name="UserFeats", shape=[usernum + 1, UserFeatures.shape[1]],dtype=tf.float32)

        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        # sequence embedding, item embedding table
        self.seq_in, item_emb_table = embedding(self.input_seq,
                                              vocab_size=itemnum + 1,
                                              num_units=args.hidden_units,
                                              zero_pad=True,
                                              scale=True,
                                              l2_reg=args.l2_emb,
                                              scope="input_embeddings",
                                              with_t=True,
                                              reuse=reuse
                                              )
        
        # sequence features and their embeddings
        self.seq_feat = tf.nn.embedding_lookup(self.ItemFeats, self.input_seq, name="seq_feat") 
        #Cxt
        self.seq_feat_in = tf.concat([self.seq_feat , self.seq_cxt], -1)
        #cxt
        self.seq_feat_emb = tf.layers.dense(inputs=self.seq_feat_in, units=args.hidden_units*5,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="feat_emb")
        #### Features Part


        # Positional Encoding
        t, pos_emb_table = embedding(
            tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
            vocab_size=args.maxlen,
            num_units=args.hidden_units,
            zero_pad=False,
            scale=False,
            l2_reg=args.l2_emb,
            scope="dec_pos",
            reuse=reuse,
            with_t=True
        )


        #### Features Part
        self.seq_concat = tf.concat([self.seq_in , self.seq_feat_emb], 2)
        self.seq = tf.layers.dense(inputs=self.seq_concat, units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='embComp')
        #### Features Part  
        #### Cxt part
        
        ####                   
        #self.seq += t

        # Dropout
        self.seq = tf.layers.dropout(self.seq,
                                      rate=args.dropout_rate,
                                      training=tf.convert_to_tensor(self.is_training))
        self.seq *= mask

        # Build blocks

        for i in range(args.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):

                # Self-attention
                self.seq = multihead_attention(queries=normalize(self.seq),
                                                keys=self.seq,
                                                num_units=args.hidden_units,
                                                num_heads=args.num_heads,
                                                dropout_rate=args.dropout_rate,
                                                is_training=self.is_training,
                                                causality=False,
                                                scope="self_attention")

                # Feed forward
                self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                        dropout_rate=args.dropout_rate, is_training=self.is_training)
                self.seq *= mask

        self.seq = normalize(self.seq)



        #pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen]) #(128 x 200) x 1 
        #neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen]) #(128 x 200) x 1 

        ##cxt
        #pos_cxt_resh = tf.reshape(self.pos_cxt, [tf.shape(self.input_seq)[0] * args.maxlen, cxt_size]) #(128 x 200) x 6
        #neg_cxt_resh = tf.reshape(self.neg_cxt, [tf.shape(self.input_seq)[0] * args.maxlen, cxt_size]) #(128 x 200) x 6
        ##
        #usr = tf.reshape(self.u, [tf.shape(self.input_seq)[0] * args.maxlen]) #(128 x 200) x 1 


        pos_emb_in = tf.nn.embedding_lookup(item_emb_table, pos) #(128 x 200) x h
        neg_emb_in = tf.nn.embedding_lookup(item_emb_table, neg) #(128 x 200) x h

        #seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units]) # 128 x 200 x h=> (128 x 200) x h
        
        #seq_emb_train = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units]) # 128 x 200 x h=> (128 x 200) x h
        #seq_emb_test = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units]) # 1 x 200 x h=> (1 x 200) x h
        seq_emb_train = self.seq #128 x 200 x h
        seq_emb_test = self.seq #128 x 200 x h



        #############User Embedding
        #user_emb = tf.one_hot(usr , usernum+1)
        #user_emb = tf.concat([tf.nn.embedding_lookup(self.UserFeats, usr, name="user_feat") ,user_emb], -1) 
        #user_emb = tf.layers.dense(inputs=user_emb, units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="user_emb")
        ##
        #seq_emb_train = tf.concat([seq_emb_train, user_emb], -1) 
        #seq_emb_train = tf.layers.dense(inputs=seq_emb_train, units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="seq_user_emb")

        #############

        #### Features Part
        pos_feat_in = tf.nn.embedding_lookup(self.ItemFeats, pos, name="seq_feat")  #(128 x 200) x h
        ##cxt
        pos_feat = tf.concat([pos_feat_in , self.pos_cxt], -1)  #(128 x 200) x h
        ##
        pos_feat_emb = tf.layers.dense(inputs=pos_feat, reuse=True, units=args.hidden_units*5,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="feat_emb")
        pos_emb_con = tf.concat([pos_emb_in, pos_feat_emb], -1)
        pos_emb = tf.layers.dense(inputs=pos_emb_con, reuse=True, units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='embComp') # 128 x 200 x h


        #pos_emb = tf.multiply(pos_emb,user_emb)
 

        neg_feat_in = tf.nn.embedding_lookup(self.ItemFeats, neg, name="seq_feat") 
        ##cxt
        neg_feat = tf.concat([neg_feat_in , self.neg_cxt], -1)
        ##
        neg_feat_emb = tf.layers.dense(inputs=neg_feat, reuse=True, units=args.hidden_units*5,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="feat_emb")
        neg_emb_con = tf.concat([neg_emb_in, neg_feat_emb], -1)
        neg_emb = tf.layers.dense(inputs=neg_emb_con, reuse=True, units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='embComp') # 128 x 200 x h


        #neg_emb = tf.multiply(neg_emb,user_emb)
        #### Features Part


       

        self.test_item = tf.placeholder(tf.int32, shape=(101))
        self.test_item_cxt = tf.placeholder(tf.float32, shape=(101, cxt_size))

        test_item_resh = tf.reshape(self.test_item, [1,101]) 
        test_item_cxt_resh = tf.reshape(self.test_item_cxt, [1,101,cxt_size]) #1 x 101 x 6

        test_item_emb_in = tf.nn.embedding_lookup(item_emb_table, test_item_resh) #1 x 101 x h

        ########### Test user
        self.test_user = tf.placeholder(tf.int32, shape=(args.maxlen))
        #test_user_emb = tf.one_hot(self.test_user , usernum+1)
        #test_user_emb = tf.nn.embedding_lookup(self.UserFeats, self.test_user, name="Test_user_feat") 
        #test_user_emb = tf.concat([tf.nn.embedding_lookup(self.UserFeats, self.test_user, name="Test_user_feat") ,test_user_emb], -1) 
        #test_user_emb = tf.layers.dense(inputs=test_user_emb, reuse=True, units=args.hidden_units,activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="user_emb")    
        

        #### Features Part
        test_feat_in = tf.nn.embedding_lookup(self.ItemFeats, test_item_resh, name="seq_feat")  #1 x 101 x f
        ##cxt 
        test_feat_con = tf.concat([test_feat_in , test_item_cxt_resh], -1) #1 x 101 x f + 6
        ##
        test_feat_emb = tf.layers.dense(inputs=test_feat_con, reuse=True, units=args.hidden_units*5,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="feat_emb") #1 x 101 x h
        test_item_emb_con = tf.concat([test_item_emb_in, test_feat_emb], -1)  #1 x 101 x 2h
        test_item_emb = tf.layers.dense(inputs=test_item_emb_con, reuse=True, units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='embComp')  # 1 x 101 x h


        ############################################################################

        #test_item_emb = tf.multiply(test_item_emb, test_user_emb)
        #### Features Part
        mask_pos = tf.expand_dims(tf.to_float(tf.not_equal(self.pos, 0)), -1)
        mask_neg = tf.expand_dims(tf.to_float(tf.not_equal(self.neg, 0)), -1)


        self.test_logits = None
        for i in range(1):
            with tf.variable_scope("num_blocks_p_%d" % i):

                # Self-attentions, # 1 x 200 x h
                # Self-attentions, # 1 x 101 x h
                self.test_logits = multihead_attention2(queries=test_item_emb,
                                                keys=seq_emb_test,
                                                num_units=args.hidden_units,
                                                num_heads=args.num_heads,
                                                dropout_rate=args.dropout_rate,
                                                is_training=self.is_training,
                                                causality=False,
                                                res = use_res,
                                                scope="self_attention") 

                # Feed forward , # 1 x 101 x h
                #self.test_logits = feedforward(self.test_logits, num_units=[args.hidden_units, args.hidden_units], dropout_rate=args.dropout_rate, is_training=self.is_training) 
                
                

        ##Without User
        self.test_logits = tf.layers.dense(inputs=self.test_logits, units=1,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='logit')  # 1 x 101 x 1
        self.test_logits = tf.reshape(self.test_logits, [1, 101], name="Reshape_pos") # 101 x 1




        ## prediction layer
        ############################################################################
        self.pos_logits =  None
        self.neg_logits = None
        for i in range(1):
            with tf.variable_scope("num_blocks_p_%d" % i):

                # Self-attentions, # 128 x 200 x 1
                self.pos_logits = multihead_attention2(queries=pos_emb,
                                                keys=seq_emb_train,
                                                num_units=args.hidden_units,
                                                num_heads=args.num_heads,
                                                dropout_rate=args.dropout_rate,
                                                is_training=self.is_training,
                                                causality=False,
                                                reuse=True, 
                                                res = use_res,
                                                scope="self_attention") 

                # Feed forward , # 128 x 200 x 1
                #self.pos_logits = feedforward(normalize(self.pos_logits), num_units=[args.hidden_units, args.hidden_units], dropout_rate=args.dropout_rate, is_training=self.is_training,reuse=True) 
                self.pos_logits *= mask_pos

        for i in range(1):
            with tf.variable_scope("num_blocks_p_%d" % i):

                # Self-attentions
                self.neg_logits = multihead_attention2(queries=neg_emb,
                                                keys=seq_emb_train,
                                                num_units=args.hidden_units,
                                                num_heads=args.num_heads,
                                                dropout_rate=args.dropout_rate,
                                                is_training=self.is_training,
                                                causality=False,
                                                reuse=True, 
                                                res = use_res,
                                                scope="self_attention")

                # Feed forward  # 128 x 200 x 1
                #self.neg_logits = feedforward(normalize(self.neg_logits), num_units=[args.hidden_units, args.hidden_units], dropout_rate=args.dropout_rate, is_training=self.is_training,reuse=True)
                self.neg_logits *= mask_neg                
                              



        self.pos_logits = tf.layers.dense(inputs=self.pos_logits, reuse=True, units=1,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='logit') 
        self.neg_logits = tf.layers.dense(inputs=self.neg_logits, reuse=True, units=1,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='logit') 
        #tf.reduce_sum(pos_emb * seq_emb_train, -1)

        self.pos_logits = tf.reshape(self.pos_logits, [tf.shape(self.input_seq)[0] * args.maxlen], name="Reshape_pos") # 128 x 200 x 1=> (128 x 200) x 1
        self.neg_logits = tf.reshape(self.neg_logits, [tf.shape(self.input_seq)[0] * args.maxlen], name="Reshape_neg") # 128 x 200 x 1=> (128 x 200) x 1
        ###########################################################################




        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx, seqcxt, testitemcxt):
        return sess.run(self.test_logits,
                        {self.test_user: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False, self.seq_cxt:seqcxt, self.test_item_cxt:testitemcxt})

"""#Main"""

import os
import time
import argparse
import tensorflow as tf
from tqdm import tqdm

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

dataset_name = sys.argv[1]
args = None

if dataset_name == 'Beauty':
    class Args:
      dataset = 'Beauty'
      train_dir = 'default'
      batch_size = 128
      lr = 0.0001
      maxlen = 75
      hidden_units = 90
      num_blocks = 3
      num_epochs = 1801
      num_heads = 1 #
      dropout_rate = 0.5 #2
      l2_emb = 0.0001
      cxt_size = 6   
      use_res = True
    args = Args()


if dataset_name == 'Fashion':
    class Args:
      dataset = 'Fashion'
      train_dir = 'default'
      batch_size = 100
      lr = 0.00001
      maxlen = 35
      hidden_units = 390
      num_blocks = 3
      num_epochs = 801
      num_heads = 3 #
      dropout_rate = 0.3 #2
      l2_emb = 0.0001
      cxt_size = 6 
      use_res = False
    args = Args()


if dataset_name == 'Men' :
    class Args:
      dataset = 'Men'
      train_dir = 'default'
      batch_size = 128
      lr = 0.000006
      maxlen = 35
      hidden_units = 390
      num_blocks = 3
      num_epochs = 801
      num_heads = 3 #
      dropout_rate = 0.3 #2
      l2_emb = 0.0001
      cxt_size = 6 
      use_res = False
    args = Args()


if dataset_name == 'Video_Games':
    class Args:
        dataset = 'Video_Games'
        train_dir = 'default'
        batch_size = 128
        lr = 0.0001
        maxlen = 50
        hidden_units = 90
        num_blocks = 3
        num_epochs = 801
        num_heads = 3 #
        dropout_rate = 0.5 #2
        l2_emb = 0.0
        cxt_size = 6   
        use_res = True
    args = Args()

#with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
#    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
#f.close()


dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) / args.batch_size
print(usernum,'--',itemnum)
ItemFeatures = None
UserFeatures = None

if args.dataset == 'Beauty' :
  ItemFeatures = get_ItemDataBeauty(itemnum)
  #UserFeatures = get_UserDataBeauty(usernum)
  UserFeatures = []
  CXTDict = load_data('./Data/CXTDictSasRec_Beauty.dat')

if args.dataset == 'Men' :
  ItemFeatures = get_ItemDataMen(itemnum)
  #UserFeatures = get_UserDataMen(usernum)
  UserFeatures = []
  CXTDict = load_data('./Data/CXTDictSasRec_Men.dat')

if args.dataset == 'Fashion' :
  ItemFeatures = get_ItemDataFashion(itemnum)
  #UserFeatures = get_UserDataFashion(usernum)
  UserFeatures = []
  CXTDict = load_data('./Data/CXTDictSasRec_Fashion.dat')

if args.dataset == 'Video_Games' :
  ItemFeatures = get_ItemDataGames(itemnum)
  #UserFeatures = get_UserDataFashion(usernum)
  UserFeatures = []
  CXTDict = load_data('./Data/CXTDictSasRec_Games.dat')


print(ItemFeatures.shape)
#print(UserFeatures)
#print(abc)
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print ('average sequence length: %.2f' % (cc / len(user_train)))

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.allow_soft_placement = True
#sess = tf.Session(config=config)
sess = tf.Session()
sampler = WarpSampler(user_train, usernum, itemnum, CXTDict, args.cxt_size, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(usernum, itemnum, args, ItemFeatures, UserFeatures, args.cxt_size,use_res = args.use_res)
sess.run(tf.initialize_all_variables())
T = 0.0
t0 = time.time()


for epoch in range(1, args.num_epochs + 1):
    for step in tqdm(range(int(num_batch)), total=int(num_batch), ncols=70, leave=False, unit='b'):
    #for step in range(int(num_batch)):
        u, seq, pos, neg, seqcxt, poscxt, negcxt = sampler.next_batch()

        auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                 model.is_training: True, model.seq_cxt:seqcxt, model.pos_cxt:poscxt, model.neg_cxt:negcxt})

    if epoch % 20 == 0:  #20
        t1 = time.time() - t0
        T += t1
        print ('Evaluating')
        t_test = evaluate(model, dataset, args, sess, CXTDict, args.cxt_size)
        t_valid = evaluate_valid(model, dataset, args, sess, CXTDict, args.cxt_size)
        #print(t_test)
        print ('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f, AUC: %.4f), test (NDCG@10: %.4f, HR@10: %.4f, AUC: %.4f)' % (epoch, T, t_valid[0], t_valid[1], t_valid[2], t_test[0], t_test[1], t_test[2]))
        t0 = time.time()


sampler.close()
print("Done")


