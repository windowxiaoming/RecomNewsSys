#-*-coding:utf-8-*-
import codecs
import yaml
import os
import pandas as pd
import numpy as np
import sys
from pymongo import MongoClient
from configparser import ConfigParser
from sklearn.feature_extraction.text import TfidfVectorizer
def calculate_uf_cf(db,customer_id):
    uf = dict()
    cf = dict()
    results = db[customer_id].aggregate(
        [
            {
                "$match":{
                        "data_content_id":{"$exists":1}
                }
            },
            {
                "$group": {
                    "_id": {"data_content_id": '$data_content_id'},
                    "user_ids": {"$addToSet": "$user_id"},
                    "counts": {"$sum": 1}

                }
            },
            {
                "$project": {
                    "data_content_id" :'$_id.data_content_id',
                    "user_ids" :"$user_ids",
                    "counts": "$counts",
                    "_id": 0
                }
            }
        ],
        session=None,allowDiskUse=True
    )
    for cursor in results:
        uf[cursor['data_content_id']] = max(len(cursor['user_ids']),1)
        cf[cursor['data_content_id']] = cursor['counts']
    return uf,cf

def save_ufcf(pymodb,customer_id,cf,uf):
    for data_content_id in cf:
        info = {"data_content_id":data_content_id,"rank":cf[data_content_id]*uf[data_content_id]}
        pymodb[customer_id].update_one({"data_content_id":data_content_id},{"$set":info})


def calculate_corpus(pymodb):
    customer_ids = get_customerid(pymodb)
    for customer_id in customer_ids:
        uf,cf = calculate_uf_cf(pymodb.cont[pymodb.DST_DB],customer_id)
        save_ufcf(pymodb.cont[pymodb.ALL_CONTENT_DB],customer_id,cf,uf)

def get_customerid(pymodb):
    customer_ids = pymodb.cont[pymodb.CONTENT_DB].list_collection_names()
    customer_ids = [i for i in customer_ids if i.isdigit()]
    return customer_ids

if __name__ == '__main__':
    #read_data("cache/1.csv")
    pass
