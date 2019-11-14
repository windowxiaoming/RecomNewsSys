#-*-coding:utf-8-*-
from configparser import ConfigParser
from pymongo import MongoClient
import sys
import redis

class Mongo_Object(object):
    def __init__(self,IP_ADDR="127.0.0.1",
                 PORT="27017",
                 SRC_DB="",
                 SRC_COLL="",
                 DST_DB="",
                 USER_DB="",
                 CONTENT_DB="",
                 ALL_CONTENT_DB=""
                 ):
        self.IP_ADDR = IP_ADDR
        self.PORT = PORT
        self.SRC_DB = SRC_DB
        self.SRC_COLL = SRC_COLL
        self.DST_DB = DST_DB
        self.USER_DB = USER_DB
        self.CONTENT_DB = CONTENT_DB
        self.ALL_CONTENT_DB = ALL_CONTENT_DB
        self.flag = self.check()

    def check(self):
        try:
            conn_str = "mongodb://{}:{}/".format(self.IP_ADDR,self.PORT)
            cont = MongoClient(conn_str)
            db = cont[self.SRC_DB]
            co = db[self.SRC_COLL]
            test = co.find_one({})
            if test:
                self.cont = cont
                return True
            else:
                return None
        except:
            cont.close()
            return False

def check_redis(host,port,password,db):
    redis_ip = host
    redis_port = port
    redis_pass = password
    redis_db = db
    try:
        pedis = redis.Redis(host=redis_ip, port=redis_port, password=redis_pass, db=redis_db)
        pedis.set("username", "hoge")
        pedis.get("username")
        pedis.delete("username")
    except Exception as e:
        print(e)
        pedis = None
        pass
    return pedis

class Parameter:
    def __init__(self,loss,\
                 n_iter,\
                 batch_size,\
                 learning_rate,\
                 l2,\
                 use_cuda,\
                 valid_records,\
                 model_directory,\
                 data_directory
                 ):
        self.loss = loss
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self.use_cuda = use_cuda
        self.valid_threshold = valid_records
        self.model_directory = model_directory
        self.data_directory = data_directory

    def get_hyperparameter(self):
        hparams = dict()
        hparams['loss'] = self.loss
        hparams['n_iter'] = self.n_iter
        hparams['batch_size'] = self.batch_size
        hparams['learning_rate'] = self.learning_rate
        hparams['l2'] = self.l2
        hparams['use_cuda'] = self.use_cuda
        hparams['data_directory'] = self.data_directory
        return hparams

class Directory:
    def __init__(self,data_directory,\
                 tmp_directory,\
                 index_directory,\
                 cb_directory
                 ):
        self.data_directory = data_directory
        self.tmp_directory = tmp_directory
        self.index_directory = index_directory
        self.cb_directory = cb_directory

def check_sys():
    conf = ConfigParser()
    conf.read("Config.ini")

    mongo_ip = conf.get("Mongodb","ip")
    mongo_port = conf.get("Mongodb","port")

    mongo_src = conf.get("Mongodb","src_db")
    mongo_src_coll = conf.get("Mongodb","src_coll")
    mongo_dst = conf.get("Mongodb", "dst_db")
    mongo_user_db = conf.get("Mongodb","user_db")
    mongo_content_db = conf.get("Mongodb", "content_db")
    mongo_all_content_db = conf.get("Mongodb", "all_content_db")

    mongo = Mongo_Object(IP_ADDR=mongo_ip,
                               PORT=mongo_port,
                               SRC_DB=mongo_src,
                               SRC_COLL=mongo_src_coll,
                               DST_DB=mongo_dst,
                               USER_DB=mongo_user_db,
                               CONTENT_DB=mongo_content_db,
                               ALL_CONTENT_DB = mongo_all_content_db
                               )

    if not mongo.flag:
       raise Exception("Mongodb数据库配置异常")


    redis_ip = conf.get("Redis", "ip")
    redis_port = conf.get("Redis", "port")
    redis_pass = conf.get("Redis", "password")
    redis_db = conf.get("Redis", "db")
    pyredis = check_redis(redis_ip, redis_port, redis_pass, redis_db)

    if not pyredis:
        raise Exception("Redis数据库配置异常")

    loss = conf.get("Hyperparameter", "loss")
    n_iter = int(conf.get("Hyperparameter", "n_iter"))
    batch_size = int(conf.get("Hyperparameter", "batch_size"))
    learning_rate = float(conf.get("Hyperparameter", "learning_rate"))
    l2 = float(conf.get("Hyperparameter", "l2"))
    use_cuda = bool(conf.get("Hyperparameter", "use_cuda"))
    model_directory = conf.get("Hyperparameter","model_directory")

    valid_records = int(conf.get("Control", "n_records"))
    merge_time = int(conf.get("Control", "merge_time"))

    data_directory = conf.get("Directory", "data")
    tmp_directory = conf.get("Directory", "tmp")
    index_directory = conf.get("Directory", "index")
    cb_directory = conf.get("Directory", "cb")

    hparams = Parameter(loss,n_iter,batch_size,learning_rate,l2,use_cuda,valid_records,model_directory,data_directory)

    directory = Directory(data_directory,tmp_directory,index_directory,cb_directory)

    sys_values = (mongo,pyredis,merge_time,hparams,directory,valid_records)

    print("系统检查完成，无异常发生!")
    return sys_values

if __name__ == '__main__':
    check_sys()
