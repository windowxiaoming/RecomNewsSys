#-*-coding:utf-8-*-
#增量导入数据任务
import codecs
import yaml
import os,sys
import time
Basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(Basedir)
from pymongo.errors import BulkWriteError
from bson.objectid import ObjectId

# def copy_into_redis(pyredis,data_content_id,customer_id):
#     name = "{}{}{}{}".format("customer_id_",customer_id,"_",data_content_id)
#     newest_len = pyredis.lpush(name, data_content_id)
#     if newest_len > 10000:
#         pyredis.rpop(name)

def copy_into_collection(pyredis,cursor, dest_coll):
    c = 0
    tw = []
    latest_id = 0
    for t in cursor:
        latest_id = t['_id']
        c += 1
        tw.append(t)
        if c == 10000:
            c = 0
            insert_many(dest_coll, tw)
            tw = []

    if c > 0:
        insert_many(dest_coll, tw)
    return latest_id

def insert_many(coll, documents):
    try:
        coll.insert_many(documents, ordered=False)
    except BulkWriteError:
        pass

def resume_partition(pyredis,db_src,db_dst,COLLECTION,customer_ids):
    for index,customer_id in enumerate(customer_ids):
        if customer_id != '0':
            obj_id = str(customer_ids[customer_id][1])
            if len(obj_id) == 24:
                id = ObjectId(obj_id)
            else:
                id = ObjectId('0'*24)

            results = db_src[COLLECTION].aggregate([
                {"$match":{"_id":{"$gt":id}}},
            ])
            latest_id = copy_into_collection(pyredis,results,db_dst[customer_id])
            customer_ids[customer_id][1] = str(latest_id)
        sys.stdout.write("%d Processed Customer_id %s Success!\n"%(index+1,customer_id))
    return customer_ids

def resume(MODB,pyredis):
    '''
    增量导入业务数据
    '''
    try:
        with codecs.open("customer_ids.yml", 'r', 'utf-8') as f:
            customer_ids = yaml.load(f, Loader=yaml.FullLoader)
        customer_ids = resume_partition(pyredis,MODB.cont[MODB.SRC_DB],MODB.cont[MODB.DST_DB],MODB.SRC_COLL,customer_ids)
        with codecs.open("customer_ids.yml", 'w', 'utf-8') as f1:
            yaml.dump(customer_ids,f1)
    except Exception as e:
        sys.stdout.write("Resume Partition Task Failed!\n")
        sys.stdout.write("%s\n"%(e))

if __name__ == '__main__':
    # from pymongo import MongoClient
    #
    # conn_str = "mongodb://{}:{}/".format("10.0.1.111", 27017)
    # c = MongoClient(conn_str)
    # resume_partition(c['temp'],c['region'],'news_data',{'136':['100','5d208104e3a42916c94b1bc7']})
    #import redis
    #pedis = redis.Redis(host="10.0.1.111", port=6379, password='hogesoft123', db=31)
    #pedis.lpush('key', 1)
    #pedis.lpush('key', 4)
    #
    # result2 = pedis.rpop('key')
    #result1 = pedis.lrange('key', 0, -1)
    # # result3 = pedis.rpop('key')
    #
    # print("result2:", result2)
    #print("result1:", result1)
    # print("result3:", result3)
    pass





