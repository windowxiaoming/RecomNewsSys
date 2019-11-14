from pymongo import MongoClient
from bson import ObjectId
conn_str = "mongodb://{}:{}/".format("10.0.1.110",27017)
cont = MongoClient(conn_str)
info = {
	"abc" : 2,
	"age" : 18,
	"name" : "gugo",
	"aaa" : 1236668
}
#r = cont['test']['db'].update_one({"_id":ObjectId("5db10d430cc5e192fa3fc0aa")},{"$set":{"aaa":123}})
# r = cont['test']['db'].update_one({"age":18},{"$set":info})
# print(r.raw_result)
# print(r.raw_result['updatedExisting'] == False)

f = cont['test']['db'].find({}).sort([("_id", -1)]).limit(2)
for i in f:
    print(i)