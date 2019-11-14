#-*-coding:utf-8-*-
#执行数据库初始化任务
import codecs
import yaml
import os,sys
Basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(Basedir)
from pymongo.errors import BulkWriteError
import pandas as pd

def get_customer_ids(db,COLLECTION):
    result = db[COLLECTION].aggregate([
        {
            "$group": {"_id": "$customer_id", "counts": {"$sum": 1}, }
        }
    ])

    data = pd.DataFrame(result)
    customer_ids = list(data['_id'])
    count_list = list(data['counts'])
    object_id_list = [0] * len(count_list)
    return dict(zip(map(lambda x:str(x),customer_ids),map(lambda x:list(x),zip(count_list,object_id_list))))

def write_ids2yaml(ids,fName):
    fp = codecs.open(fName,'w','utf-8')
    yaml.dump(ids, fp)
    fp.close()

def drop_collection(db,COLLECTIONS):
    for collection in COLLECTIONS:
        db[collection].drop()

def get_collection_names(db):
    return db.list_collection_names()

def mongo_partition(db_src,db_dst,COLLECTION,customer_ids):
    for index,customer_id in enumerate(customer_ids):
        if customer_id != '0':
            results = db_src[COLLECTION].aggregate([
                {
                   "$match":{
                            "customer_id":int(customer_id)
                        }
                },
                {
                    "$sort":{"_id":1}
                }
            ])
            latest_id,u_latest_id = copy_into_collection(results,db_dst[customer_id])
            customer_ids[customer_id][1] = str(latest_id)
        sys.stdout.write("%d processed customer_id:%s\n"%(index+1,customer_id))
    return customer_ids

def copy_into_collection(cursor, dest_coll):
    c = 0
    tw = []
    latest_id = 0
    u_latest_id = 0
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
    return latest_id,u_latest_id

def insert_many(coll, documents):
    try:
        coll.insert_many(documents, ordered=False)
    except BulkWriteError:
        pass
def init(MODB):

    dst_coll_names = get_collection_names(MODB.cont[MODB.DST_DB])  # ['xx1','xx2']
    drop_collection(MODB.cont[MODB.DST_DB], dst_coll_names)
    customer_ids = get_customer_ids(MODB.cont[MODB.SRC_DB],MODB.SRC_COLL)  # {'121': [3531543, '5d22c157e3a42916c95af4a7']}
    customer_ids = mongo_partition(MODB.cont[MODB.SRC_DB], MODB.cont[MODB.DST_DB], MODB.SRC_COLL, customer_ids)
    write_ids2yaml(customer_ids, 'customer_ids.yml')

if __name__ == '__main__':
    pass
