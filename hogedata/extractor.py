# -*-coding:utf-8-*-
#定时将数据流提取至csv文件
#针对登录用户
import os
Basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from sympy import *
import codecs
import glob
import yaml
import shutil
import sys
import math
import pickle
from time import time,sleep
from hogedata.time_management import Calculate_H_M_S
def read_customer_ids(fName):
    with codecs.open(fName,'r','utf-8') as fp1:
        customer_ids = yaml.load(fp1,Loader=yaml.SafeLoader)
    return customer_ids

def get_user_data(customer_ids,db,dst_dir,threshold=0):
    '''
    :param ['127']:
    :param c['region']:
    :param tmp:
    :param 0:
    '''
    for customer_id in customer_ids:
            result = db[customer_id].aggregate(
                [
                    {
                        "$match":{
                            "user_id":{"$exists":1,"$ne":0}
                        }
                    },
                    # {
                    #     "$match": {
                    #         "create_time": {"$gte": 0, "$lte": int(time())}
                    #     }
                    # },
                    {
                        "$group": {
                            "_id": {"user_id": '$user_id', "data_content_id": "$data_content_id"},
                            "counts": {"$sum": 1},
                            "create_time": {"$first":"$create_time"},
                            "data_title" : {"$first":"$data_title"}
                        }
                    },
                    {
                        "$project": {
                            "user_id":"$_id.user_id",
                            "data_content_id" :'$_id.data_content_id',
                            "counts" :"$counts",
                            "create_time" :"$create_time",
                            "data_title":"$data_title",
                            "_id": 0
                        }
                    }
                ],
                session=None,allowDiskUse=True
            )
            data = pd.DataFrame(result)
            if len(data) > threshold:
                data.to_csv("%s/%s.csv"%(dst_dir,customer_id),encoding='utf-8',mode='w',header=True,index=False)
                print("%s.csv is Saved!"%(customer_id))

def save_data_content_info(customer_id,content_ids,df,numbers,db,data_content_ids):
    '''
    :param pyredis:
    :param customer_id: '127'
    :param data_content_ids: {("10210","都兴"):10}
    :param _df:
    :return:
    '''
    customer_id = str(customer_id)
    for index, data_content_id in enumerate(data_content_ids):
        counts = data_content_ids[data_content_id]
        _df = math.log( numbers / (1 + round(len(df[data_content_id[0]]),6)),10)
        tf = int(counts)
        tfidf = tf*_df
        info = {"content_id":data_content_id[0], \
                "content":data_content_id[1], \
                "rank":str(tfidf), \
                "index":content_ids[data_content_id[0]]
                }
        result = db[customer_id].update_one({"content_id":data_content_id[0]},{"$set":info})
        #{'ok': 1, 'nModified': 0, 'n': 0, 'updatedExisting': False}
        if result.raw_result['updatedExisting'] == False:
            db[customer_id].insert_one(info)
    return

def save_user_info(customer_id,user_ids,db,history):
    for user_id in history:
        info = {"user_id":user_id,"history":history[user_id],"index":user_ids[user_id]}
        result = db[customer_id].update_one({"user_id":user_id},{"$set":info})
        if result.raw_result['updatedExisting'] == False:
            db[customer_id].insert_one(info)

def get_content_index(ids,customer_id,index_directory):
    ##{('222', '综合频道'): 7, ('644', '松溪：遭强降雨袭击 转移地灾点群众2100多人'): 155}
    ##customer_id=165,('176562', '【正在直播】赏锦绣中华风 世园会举行中国馆日开馆仪式'): 1, ('176562', '【直播预告】赏锦绣中华风 世园会举行中国馆日开馆仪式')
    ##存在不同data_content_id对应同一标题
    index_dict = dict()
    content_id_dict = []
    count = 0
    for index,content in enumerate(ids):
        if content[0] not in index_dict:
            index_dict[content[0]] = count + 1
            content_id_dict.append(content[0])
            count = count + 1
    path = os.path.join(index_directory,"{}_{}_index.pkl".format(customer_id,"content_id"))

    data = {'content_ids_list': content_id_dict}
    with open(path,'wb') as fp:
        pickle.dump(data,fp)
    return index_dict

def get_uindex(ids):
    index_dict = dict()
    for index,user in enumerate(ids):
        index_dict[user] = index + 1
    return index_dict

def get_df(df):
    #results = {('10', '644'): 1, ('3', '746'): 1, ('4', '644'): 1, ('5', '644'): 1, ('5', '645'): 1}
    #
    results = dict(df.groupby(['data_content_id', 'user_id']).size())
    _df = {}
    user_his = {}
    numbers = len(results)
    for result in results:
        data_content_id, user_id = result
        if data_content_id not in _df:
            _df[data_content_id] = [results[result]]
        else:
            _df[data_content_id].append(results[result])

        if user_id not in user_his:
            user_his[user_id] = [data_content_id]
        else:
            user_his[user_id].append(data_content_id)
    return _df,numbers,user_his

def Cal_Rating(paths,directory,modb):
    for path in paths:
        filename = os.path.basename(path)
        name = filename[:-4]
        dst_path = os.path.join(directory.data_directory,filename)
        pd_type = {'user_id':str,'data_content_id':str,'counts':int,'create_time':int,'data_title':str}
        data_csv = pd.read_csv(path,header=0,dtype=pd_type)
        data_csv.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

        #                                    counts  create_time
        #data_content_id data_title
        #1000            郑恺最恐惧游戏重出江湖      23  28024677599
        #1001            李晨到底说了什么？         2   3113773914
        tt = data_csv.groupby(['data_content_id', 'data_title'])
        mm = tt.sum()
        data_content_ids = dict(mm['counts'])

        count_data = data_csv['counts']
        temp_1 = np.array(count_data, dtype=np.float64)
        temp_1 = Min_Max_Normalization(temp_1)
        temp_1 = sigmoid(temp_1)
        x = Symbol('x')
        y = Symbol('y')
        min_x = np.min(temp_1)
        max_y = np.max(temp_1)
        results = solve([min_x * x - 1 + y, max_y * x - 5 + y], [x, y])
        a = float(results[x])
        b = float(results[y])
        temp_3 = linear(temp_1, a, b)
        temp_3 = np.round(temp_3, 10)
        data_csv['counts'] = pd.array(np.array(temp_3, dtype=np.str))

        #
        #{('121189', '重视传统文化体验 渭南桃花塬成文青逛陕西新热点'): 3, ('160631', '畅游渭南——华州少华山国家森林公园'): 5}
        #
        #data_content_ids = dict(data_csv.groupby(['data_content_id','data_title']).size())

        #
        #{'1019': 6, '1032': 4, '1033': 15, '1036': 3, '1048': 2}
        #
        user_ids = dict(data_csv.groupby('user_id').size())


        #data_content_id-->index
        content_ids_index = get_content_index(data_content_ids,name,directory.index_directory)

        #user_id-->index
        n = get_uindex(user_ids)

        _df,numbers,history = get_df(data_csv)

        save_data_content_info(name,content_ids_index,_df,numbers,modb.cont[modb.CONTENT_DB],data_content_ids)

        save_user_info(name,n,modb.cont[modb.USER_DB],history)

        # 转换原始数据的data_content_id和user_id，遍历方式(待优化)，pandas的replace方法存在问题
        data_content_ids_list = np.array(data_csv['data_content_id'], dtype=np.str).tolist()
        user_ids = np.array(data_csv['user_id'], dtype=np.str).tolist()

        data_csv['data_content_id'] = [content_ids_index[data_content_id] for data_content_id in data_content_ids_list]
        data_csv['user_id']= [n[user_id] for user_id in user_ids]

        data_csv.to_csv(dst_path, columns=['user_id','data_content_id','counts','create_time','data_title'], index=False)
        print("Calculate %s Success"%(path))
    return

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def Min_Max_Normalization(x):
    min_x = np.min(x)
    max_x = np.max(x)
    y = (x - min_x) * 1.0 / (max_x - min_x)
    return y

def linear(x, a, b):
    return a * x + b

def extractor_data(event,modb,directory,valid_records):

    customer_ids = read_customer_ids('customer_ids.yml')

    if not os.path.exists(directory.data_directory):
        os.makedirs(directory.data_directory)

    if not os.path.exists(directory.index_directory):
        os.makedirs(directory.index_directory)

    if not os.path.exists(directory.tmp_directory):
        os.makedirs(directory.tmp_directory)

    count = 1
    while True:
        event.clear()
        sys.stdout.write("Extract Data,Execute the %s times\n" % (count))
        db = modb.cont[modb.DST_DB]
        start_time = time()
        get_user_data(customer_ids,db,directory.tmp_directory,threshold=valid_records)
        elaps_t1 = (time() - start_time)
        H,M,S = Calculate_H_M_S(elaps_t1)
        sys.stdout.write("Get login user data elaps times:%s:%s:%s\n"%(H,M,S))

        start_time = time()

        Cal_Rating(glob.glob(os.path.join(directory.tmp_directory,"*.csv")),directory,modb)
        elaps_t2 = (time() - start_time)
        H, M, S = Calculate_H_M_S(elaps_t2)
        sys.stdout.write("Calculate Rating Elaps times:%s:%s:%s\n" % (H, M, S))
        sys.stdout.write("Extractor Data Success\n")
        count = count + 1
        event.set()
        sleep(10)

if __name__ == '__main__':
    pass


