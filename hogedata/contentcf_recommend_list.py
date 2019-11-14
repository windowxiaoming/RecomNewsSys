from bson import ObjectId
import numpy as np
import sys
import codecs
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

def similarities(customer_id,pymongodb):
    u = pymongodb.cont[pymongodb.USER_DB][customer_id]
    c = pymongodb.cont[pymongodb.CONTENT_DB][customer_id]

    c_count = c.count()
    documents = [' '] * c_count

    c_results = c.aggregate(
        [
            {
                "$match":{"content_id":{"$exists":1}}
            }
        ]
    )

    for record in c_results:
        documents[record['index']-1] = record['content']

    u_results = u.aggregate(
        [
            {
                "$match":{"user_id":{"$exists":1}}
            }
        ]
    )

    tfidf_model = TfidfVectorizer(lowercase=False)
    X = tfidf_model.fit_transform(documents)
    for record in u_results:
        contents = []
        selected = record['history']
        selected_len = len(record['history'])
        if selected_len < 5:
            seed = selected_len
        else:
            seed = 5
        results = random.sample(selected,seed)
        for history in results:
            t = c.find_one({'content_id':history})
            #f = cont['test']['db'].find({}).sort([("_id", -1)]).limit(2)
            contents.append(t['content'])

        Y = tfidf_model.transform(contents)
        distances = cosine_similarity(Y,X)#越相似值越大
        indexs = np.argsort(-distances)[:,:100]

        similarities = list(set(indexs.T.ravel().tolist()))
        record['similarityval'] = similarities#没有转换
        u.update_one({"_id":ObjectId(record["_id"])},{"$set":record})

    # path = os.path.join(directory.cb_directory,"{}.pkl".format((customer_id)))
    # data = {"user_cb":users}
    # with open(path,'wb') as fp:
    #     pickle.dump(data,fp)
    sys.stdout.write("similarity has processd \n")

def content_recommend(event,pymongodb):
    #event.wait()
    sys.stdout.write("Start Content Recommend Calculate!\n")
    customer_ids = pymongodb.cont[pymongodb.USER_DB].list_collection_names()
    while True:
        for customer_id in customer_ids:
            if customer_id.isdigit():
                similarities(customer_id,pymongodb)

if __name__ == '__main__':
    pass

