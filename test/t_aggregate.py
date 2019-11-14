from pymongo import MongoClient
from bson import ObjectId
db = MongoClient("mongodb://10.0.1.110:27017")
'''results = db['region']['127'].aggregate(
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
                    "counts":{"$sum":1}

                }
            },
            {
                "$project": {
                    "data_content_id" :'$_id.data_content_id',
                    "user_ids" :"$user_ids",
                    "counts":"$counts",
                    "_id": 0
                }
            }
        ],
        session=None,allowDiskUse=True
    )'''
'''
{'user_ids': [1033], 'counts': 4, 'data_content_id': '161314'}
{'user_ids': [0], 'counts': 2, 'data_content_id': '323197'}
{'user_ids': [1033], 'counts': 4, 'data_content_id': '322409'}
{'user_ids': [1033], 'counts': 2, 'data_content_id': '161360'}
{'user_ids': [1033], 'counts': 2, 'data_content_id': '322507'}
{'user_ids': [0], 'counts': 1, 'data_content_id': '161347'}
{'user_ids': [1033, 0], 'counts': 6, 'data_content_id': '161317'}
{'user_ids': [1033, 0], 'counts': 4, 'data_content_id': '322488'}
{'user_ids': [0], 'counts': 1, 'data_content_id': '161396'}
{'user_ids': [1032], 'counts': 1, 'data_content_id': '161319'}
{'user_ids': [1033], 'counts': 2, 'data_content_id': '161363'}
{'user_ids': [1032], 'counts': 1, 'data_content_id': '322420'}
'''
'''for cursor in results:
    print(cursor)
    #print(max(len(cursor['user_ids']),1))
'''
# r = db['region']['127'].find_one({"_id" : ObjectId("5d22bfafe3a42916c95ae56b")})
# print(r)
# for i in r:
#     print(i)

c_results = db['content_db']['165'].aggregate(
        [
            {
                "$match":{"content_id":{"$exists":1}}
            }
        ]
    )

l = db['content_db']['165'].count()
print(l)
documents = [""] * l
for record in c_results:
    print(record['content'])
    documents[record['index'] - 1] = record['content']

