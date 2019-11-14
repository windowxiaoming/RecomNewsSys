import pandas as pd
pd_type = {'user_id':str,'data_content_id':str,'counts':int,'create_time':int,'data_title':str}
data_csv = pd.read_csv("165.csv",header=0,dtype=pd_type)
result = data_csv.groupby(['data_content_id','data_title'])#data_csv['counts'].groupby(['data_content_id'])
m = result.sum()
tt = dict(m['counts'])
print(tt)

index_dict = dict()
# content_id_dict = []
# count = 1
for index,content in enumerate(tt):
    print(content)
    exit(0)
    index_dict[content[0]] = index + 1
    if content[0] not in index_dict:
        index_dict[content[0]] = index + 1
    else:
        print("{}.{}".format(count,content[0]))
        count = count + 1
    #content_id_dict.append(content[0])
print(len(index_dict))
