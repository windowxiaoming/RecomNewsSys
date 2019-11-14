import json
import torch
import os
import codecs
import yaml
import pickle
import numpy as np
from pymongo import MongoClient
from pymongo import InsertOne
from configparser import ConfigParser
from flask import Flask,request,jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
conf = ConfigParser()
conf.read("Config.ini")
host = conf.get("HTTP", "ip")
port = conf.get("HTTP", "port")
log_dir = conf.get("LOG",'path')
Mongodb_Ip = conf.get("MONGODB", "ip")
Mongodb_Port = conf.get("MONGODB", "port")
Cache_Database = conf.get("MONGODB", "cache_database")
Ccache_Collection = conf.get("MONGODB", "cache_collection")
host = conf.get("HTTP", "ip")
port = conf.get("HTTP", "port")
conn = MongoClient("%s:%s" % (Mongodb_Ip, Mongodb_Port), maxPoolSize=None)
my_db = conn[Cache_Database]
my_collection = my_db[Ccache_Collection]
directory = conf.get("other", "directory")
count = my_collection.find().count()

def Get_DCid(results,Cid):
    DCid_path = "{}/{}_data_content_ids.yml".format(directory, Cid)
    with codecs.open(DCid_path,'r','utf-8') as fp:
        DCids = yaml.load(fp, Loader=yaml.FullLoader)
    real_DCid = [DCids[result] for result in results]
    return real_DCid

def Get_pkl(cid,uid):
    with open('region_data/{}.pkl'.format(cid), 'rb') as fp:
        data = pickle.load(fp)
    return data[uid]

def Get_Uid(cid,uid):
    uid_path = "{}/{}_users.yml".format(directory, cid)
    with codecs.open(uid_path,'r','utf-8') as fp:
        uids = yaml.load(fp, Loader=yaml.FullLoader)
    for user_id in uids:
        if uids[user_id][0] == uid:
            return user_id
        else:
            continue
    return None


def predict_bpr_bloom(user_id,customer_id,k=1000):
    model_path = "model/%s_model.pkl"%(str(customer_id))
    if os.path.exists(model_path):
        model = torch.load(model_path)
        predictions = model.predict(np.array([user_id]))
        result = np.argsort(-predictions)[:k]
        return result
    else:
        return None

@app.route("/UserBased",methods=["POST"])
def recommend():
    try:
        j_data = request.get_data()
        data = json.loads(j_data)
        customer_id = data['customer_id']
        user_id = data['user_id']
        if 'top_k' not in data:
            top_k = 1000
        else:
            top_k = data['top_k']
        uindex = Get_Uid(customer_id,user_id)
        data_content_ids = predict_bpr_bloom(user_id=uindex,customer_id=customer_id,k=int(top_k))
        real_DCid = Get_DCid(data_content_ids,customer_id)
        results = [t[0] for t in real_DCid]
        if data_content_ids.all():
            data['result']='success'
            data['data']=results
            return jsonify(data)
        else:
            data['result']=''
            data['data']='-1'
            return jsonify(data)
    except Exception as e:
        print(e)
        data['result'] = 'fail'
        data['data'] = '-1'
        return jsonify(data)

def Insert_Data(data):
    insert_datas = []
    insert_datas.append(InsertOne(data))
    my_collection.bulk_write(insert_datas)
    insert_datas.clear()

def get_tfdf(Cid,k):
    fpath = "cache/{}_tfdf.yml".format(Cid)
    if os.path.exists(fpath):
        with codecs.open(fpath,'r','utf-8') as fp:
            Cids = yaml.load(fp,Loader=yaml.SafeLoader)
            res = sorted(Cids.items(),key=lambda d:d[1],reverse=True)
        return res[:k]
    else:
        return None

@app.route('/feeddata',methods=['POST'])
def feeddata():
    global count
    try:
        j_data = request.get_data()
        data = json.loads(j_data)
        print("Receive Data:{}".format(data))
        Insert_Data(data)
        count = count + 1
        return "success"
    except Exception as e:
        print(e)
        return "fail"
    pass

@app.route('/get_cache_counts',methods=['GET'])
def get_numbers():
    data = dict()
    data['counts'] = count
    return jsonify(data)

@app.route('/coldstart',methods=['POST'])
def get_nouser_result():
    try:
        j_data = request.get_data()
        data = json.loads(j_data)
        customer_id = data['customer_id']
        k = data['top_k']
        Dids = get_tfdf(customer_id,k)
        results = [t[0] for t in Dids]
        if results:
            data['result']='success'
            data['data']=results
            return jsonify(data)
        else:
            data['result']=''
            data['data']='-1'
            return jsonify(data)
    except Exception as e:
        data['result'] = 'fail'
        data['data'] = '-1'
        return jsonify(data)

@app.route('/ContentBased',methods=['POST'])
def get_ContentBased():
    try:
        j_data = request.get_data()
        data = json.loads(j_data)
        customer_id = data['customer_id']
        user_id = data['user_id']
        k = data['top_k']
        indexs = Get_pkl(customer_id,user_id)
        indexs = [ int(i)+1 for i in indexs]
        indexs = list(set(indexs))
        Dids = Get_DCid(indexs,customer_id)
        results = [t[0] for t in Dids]
        if results:
            data['result']='success'
            data['data']=results[:k]
            return jsonify(data)
        else:
            data['result']=''
            data['data']='-1'
            return jsonify(data)
    except Exception as e:
        print(e)
        data['result'] = 'fail'
        data['data'] = '-1'
        return jsonify(data)

if __name__ == '__main__':
    app.run(host=host, port=port)