#-*-coding:utf-8-*-
import os
import torch
import numpy as np
import pickle
from bson import ObjectId
def predict_bpr_bloom(user_id,customer_id,k=-1):
    model_path = "model/%s_model.pkl"%(str(customer_id))
    if os.path.exists(model_path):
        model = torch.load(model_path)
        predictions = model.predict(np.array([user_id]))
        result = np.argsort(-predictions)[:k]
        return result
    else:
        return np.empty((0))

def transition_cid(customer_id,indexs):
    index_file = "index/{}_{}_index.pkl".format(str(customer_id),"content_id")
    if not os.path.exists(index_file):
        return None
    with open(index_file,'rb') as f:
        data = pickle.load(f)
    cindex = np.array(data['content_ids_list'])
    return cindex[indexs-1].tolist()

def mongo_push_usercf(pymongo,customer_id):
    user_db = pymongo.cont[pymongo.USER_DB]
    cursor = user_db[customer_id].find({})

    for t in cursor:
        ind = int(t["index"])
        id = t['_id']
        indics = predict_bpr_bloom(user_id=ind, customer_id=customer_id)
        dids = ""
        if indics.any():
            dids = transition_cid(customer_id,indics)[:100]
        user_db[customer_id].update_one({"_id":ObjectId(id)},{"$set":{"usercfval":dids}})

def push_cache(customer_ids,pymodb):
    for customer_id in customer_ids:
        if customer_id.isdigit():
            mongo_push_usercf(pymodb,customer_id)

def start_queue(pyredis,pymodb,event):
    db = pymodb.cont[pymodb.USER_DB]
    customer_ids = db.list_collection_names()
    while True:
        push_cache(customer_ids,pymodb)
        event.wait()

if __name__ == '__main__':
    from pymongo import MongoClient
    constr = "mongodb://10.0.1.110:27017"
    user_db = MongoClient(constr)
    ss = user_db["user_db"].list_collection_names()
    for i in ss:
        if i.isdigit():
            print(i)

    cursor = user_db["user_db"]["165"].find({})

    for t in cursor:
        print(t)