#-*-coding:utf-8-*-
import threading
import sys,os
from check_system import check_sys
from hogedata.init_partition import init
from hogedata.extractor import extractor_data
from hogedata.resume_partition import resume
from hoge.train import start_engine
from hogedata.usercf_recommend_list import start_queue
from hogedata.statistic import calculate_corpus
from hogedata.contentcf_recommend_list import content_recommend
from time import sleep
event = threading.Event()
if __name__ == '__main__':
    (pymodb,pyredis,merge_time,hparams,directory,valid_records) = check_sys()

    if not os.path.exists("customer_ids.yml"):
        init(pymodb)

    extractor = threading.Thread(target=extractor_data, args=(event, pymodb,directory,valid_records))
    extractor.setDaemon(True)
    extractor.start()

    calculator = threading.Thread(target=start_engine,args=(event,pymodb,hparams.get_hyperparameter()))
    calculator.setDaemon(True)
    calculator.start()

    content_recommend = threading.Thread(target=content_recommend,args=(event,pymodb))
    content_recommend.setDaemon(True)
    content_recommend.start()

    calculator = threading.Thread(target=start_queue,args=(pyredis,pymodb,event))
    calculator.setDaemon(True)
    calculator.start()

    while True:
        calculate_corpus(pymodb)
        #sleep(merge_time)
        #resume(pymodb,pyredis)





