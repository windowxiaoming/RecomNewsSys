#-*-coding:utf-8-*-
import os,sys
import torch
import yaml
import codecs
import threading
basedir = os.path.dirname(os.path.dirname(__file__))
import pandas as pd
import numpy as np
from hoge.interactions import Interactions
from hoge.cross_validation import random_train_test_split,user_based_train_test_split
from hoge.factorization.explicit import ExplicitFactorizationModel
from hoge.factorization.implicit import ImplicitFactorizationModel
from hoge.factorization.representations import BilinearNet
from hoge.sequence.implicit import ImplicitSequenceModel
from hoge.layers import BloomEmbedding
from configparser import ConfigParser
from hogedata.time_management import Calculate_H_M_S
from time import time
CUDA = False

FLOAT_MAX = np.finfo(np.float32).max
def _get_newsdata(path):
    data = pd.read_csv(path, header=0)
    return (np.array(data['user_id'].tolist()),
            np.array(data['data_content_id'].tolist()),
            np.array(data['counts'].tolist()),
            np.array(data['create_time'].tolist()))

def get_newsdata(path):
    if os.path.exists(path):
        return Interactions(*_get_newsdata(path))
    else:
        return None

def _regression():
    RANDOM_STATE = np.random.RandomState(40)
    interactions = get_newsdata()
    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)
    model = ExplicitFactorizationModel(loss='regression',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-3,
                                       l2=1e-5,
                                       use_cuda=CUDA)
    model.fit(train)
    predictions = model.predict(np.array([2]))
    result = np.argsort(-predictions)[:10]
    print(result)

def _test_poisson():
    RANDOM_STATE = np.random.RandomState(40)
    interactions = get_newsdata()
    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ExplicitFactorizationModel(loss='poisson',
                                       n_iter=2,
                                       batch_size=1024,
                                       learning_rate=1e-3,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    predictions = model.predict(np.array([2]))
    result = np.argsort(-predictions)[:10]
    print(result)

def _data_implicit_factorization():
    RANDOM_STATE = np.random.RandomState(40)
    interactions = get_newsdata()
    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)
    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=2,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       random_state=RANDOM_STATE,
                                       use_cuda=CUDA)
    model.fit(train)
    predictions = model.predict(np.array([3]))
    result = np.argsort(-predictions)[:10]
    print(result)

def _data_implicit_sequence():
    RANDOM_STATE = np.random.RandomState(40)
    max_sequence_length = 200
    step_size = 200


    interactions = get_newsdata()
    train = user_based_train_test_split(interactions,random_state=RANDOM_STATE)

    train = train.to_sequence(max_sequence_length=max_sequence_length,
                              step_size=step_size)

    intest = Interactions(np.zeros(interactions.num_items-1),1+np.arange(interactions.num_items-1),timestamps=np.arange(interactions.num_items-1))
    t = intest.to_sequence(max_sequence_length=200,step_size=1).sequences


    model = ImplicitSequenceModel(loss='adaptive_hinge',
                                  representation='lstm',
                                  batch_size=8,
                                  learning_rate=1e-2,
                                  l2=1e-3,
                                  n_iter=1,
                                  use_cuda=CUDA,
                                  random_state=RANDOM_STATE)

    model.fit(train, verbose=True)
    for i in range(len(t)):
        predictions = -model.predict(t[i])
        print(predictions.argsort()[:10])
        break

def train_bpr_bloom(interactions,loss='bpr',n_iter=30,batch_size=512,learning_rate=1e-2,l2=1e-6,use_cuda=False):
    RANDOM_STATE = np.random.RandomState(40)
    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    user_embeddings = BloomEmbedding(interactions.num_users, 32,
                                     compression_ratio=1.0,
                                     num_hash_functions=2)
    item_embeddings = BloomEmbedding(interactions.num_items, 32,
                                     compression_ratio=1.0,
                                     num_hash_functions=2)
    network = BilinearNet(interactions.num_users,
                          interactions.num_items,
                          user_embedding_layer=user_embeddings,
                          item_embedding_layer=item_embeddings)
    model = ImplicitFactorizationModel(loss=loss,
                                       n_iter=n_iter,
                                       batch_size=batch_size,
                                       learning_rate=learning_rate,
                                       l2=l2,
                                       representation=network,
                                       use_cuda=use_cuda)

    model.fit(train,verbose=True)
    return model

def save_bpr_bloom(model,region):
    dirname = "model"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    try:
        fname = os.path.join(dirname, "%s_model.pkl"%(region))
        torch.save(model,fname)
        model = torch.load(fname)
    except:
        return None
    return model

def start_subengine(event,cid,config,flag=False):
    threadName = threading.current_thread().getName()
    csv_path = "{}/{}.csv".format(config['data_directory'],cid)

    while True:
        if flag:
            conf = ConfigParser()
            conf.read("Config.ini")
            print(conf)
            info = config
            print(info)
            for c in info:
                info[c] = conf.get(cid,c)
        if os.path.exists(csv_path):
            sys.stdout.write("Customer_Id={},线程{}开始训练!\n".format(cid,threadName))
            sys.stdout.flush()
            start = time()
            interactions = get_newsdata(csv_path)
            model = train_bpr_bloom(interactions,\
                                    loss=config['loss'],\
                                    n_iter=int(config['n_iter']),
                                    batch_size=int(config['batch_size']),\
                                    learning_rate=float(config['learning_rate']),\
                                    l2=float(config['l2']),\
                                    use_cuda=bool(config['use_cuda']))

            _ = save_bpr_bloom(model,cid)

            elaps = time() - start
            H,M,S = Calculate_H_M_S(elaps)
            sys.stdout.write("Customer_Id={},线程{}训练结束，耗时{}h,{}m,{}s!\n".format(cid,threadName,H,M,S))
            sys.stdout.flush()

        event.wait()


def set_customer_conifg(cid,infos):
    cid = str(cid)
    conf = ConfigParser()
    conf.read("Config.ini")
    sections = conf.sections()
    if cid not in sections:
        conf.add_section(cid)
    for info in infos:
        conf.set(cid,info,str(infos[info]))
    with codecs.open("Config.ini",'w','utf-8') as fp:
        conf.write(fp)
    return

def start_engine(event,pymongodb,config):
    event.wait()
    customer_ids = pymongodb.cont[pymongodb.CONTENT_DB].list_collection_names()
    for cid in customer_ids:
        if cid.isdigit():
            set_customer_conifg(cid,config)
            thread = threading.Thread(target=start_subengine,args=(event,str(cid),config,True))
            thread.start()
            sys.stdout.write("Customer_id=%s 启动完毕，计算中...\n"%(cid))
            sys.stdout.flush()

def t_api(Cid_path):
    interactions = get_newsdata(Cid_path)
    model = train_bpr_bloom(interactions, loss='bpr', n_iter=30,batch_size=512, learning_rate=0.01,l2=0.001, use_cuda=True)
    _ = save_bpr_bloom(model, '287')

if __name__ == '__main__':
    t_api("../region_data/287.csv")
