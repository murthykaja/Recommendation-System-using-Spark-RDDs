import random
import numpy as np
import math
import hashlib
import csv
import sys
import json
from datetime import datetime
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("appName")
sc = SparkContext(conf=conf)
target_users = {"PomQayG1WhMxeSl1zohAUA":1,"uEvusDwoSymbJJ0auR3muQ":1,"q6XnQNNOEgvZaeizUgHTSw":1,"n00anwqzOaR52zgMRaZLVQ":1,"qOdmye8UQdqloVNE059PkQ":1}
b_target_users = sc.broadcast(target_users)
input_rdd = sc.textFile(sys.argv[1], 32)
input_rdd = input_rdd.map(json.loads)
def helper(x):
    res={}
    for k,v in x.items():
        if k  in ["date","user_id","business_id"]:
            res[k]=v
    return res
test_rdd = input_rdd.map(lambda x: helper(x))
sparse_rdd = input_rdd.map(lambda x : [x["user_id"],x["business_id"],x["stars"],datetime.strptime(x["date"], '%Y-%m-%d %H:%M:%S')])
per_date_rdd = sparse_rdd.map(lambda x : ((x[0],x[1]),[x[2],x[3]])).reduceByKey(lambda x,y: x if x[1]>y[1] else y).map(lambda x : (x[0][0],x[0][1],x[1][0],x[1][1]))
item_30_distinct_rdd_1 = per_date_rdd.map(lambda x: (x[1],[[x[0],x[2]]])).reduceByKey(lambda x,y: x+y).map(lambda x : (x[0],(x[1],len(x[1])))).filter(lambda x: x[1][1]>=30)
distinct_business = item_30_distinct_rdd_1.map(lambda x : x[0]).distinct().collect()
b_distinct_business = sc.broadcast(sorted(distinct_business))
distinct_business_count = len(distinct_business)
item_30_distinct_rdd = item_30_distinct_rdd_1.flatMap(lambda x : [(i[0],x[0],i[1]) for i in x[1][0]])
users_5_distinct_rdd_1 = item_30_distinct_rdd.map(lambda x: (x[0],[[x[1],x[2]]])).reduceByKey(lambda x,y:x+y).map(lambda x: (x[0],(x[1],len(x[1])))).filter(lambda x: x[1][1]>=5)
users_distinct_rdd = users_5_distinct_rdd_1.map(lambda x: (x[0],x[1][0]))
users_5_distinct_rdd = users_5_distinct_rdd_1.flatMap(lambda x: [(x[0],i[0],i[1]) for i in x[1][0]])


target_users_rdd = users_5_distinct_rdd.filter(lambda x : x[0] in b_target_users.value).map(lambda x: (x[0],[[x[1],x[2]]])).reduceByKey(lambda x,y:x+y)
checkpoint_2_1 = target_users_rdd.collect()
checkpoint_2_1_dic = {}
print("**********************************************")
print("Step 2.1: Create a utility matrix, represented in sparse format as an RDD:")
for x in checkpoint_2_1:
    print("User ID : "+str(x[0])+" & 10 Business IDs:")
    temp = sorted(x[1],key=lambda y:y[0])
    print(temp[:10])
    checkpoint_2_1_dic[str(x[0])]=temp
b_target_users_business = sc.broadcast(checkpoint_2_1_dic)

def matrix_formation(x):
    res=[0]*distinct_business_count
    val=0
    n=len(x[1])
    for j in x[1]:
        val+=float(j[1])
    m = val/n
    for j in x[1]:
        res[b_distinct_business.value.index(j[0])] = j[1]-m
    return (x[0],res)
def mean_centric(x):
    val=0
    n=len(x[1])
    for j in x[1]:
        val+=float(j[1])
    m = val/n
    for j in x[1]:
        j[1] = [j[1],j[1]-m]
    return x


def key_value_conversion(x):
    res={}
    for j in x[1]:
        res[j[0]]=j[1]
    return (x[0],res)

def havecommon_2(x,y):
    item_list_1=x.keys()
    item_list_2=y.keys()
    i=0
    for ele in  item_list_1:
        if ele in item_list_2:
            i+=1
        if i==2:
            return True
    return False

def cosine_sim(x,y):
    res={}
    for k,v in x.items():
        if k not in res.keys():
            res[k]=[]
        res[k].append([1,v])
    for k,v in y.items():
        if k not in res.keys():
            res[k]=[]
        res[k].append([2,v])
    n=0
    d1=0
    d2=0
    for k,v in res.items():
        if len(v)==2:
            n+=v[0][1][1]*v[1][1][1]
            d1+=v[0][1][1]*v[0][1][1]
            d2+=v[1][1][1]*v[1][1][1]
        else:
            if v[0][0]==1:
                d1+=v[0][1][1]*v[0][1][1]
            else:
                d2+=v[0][1][1]*v[0][1][1]
    return n/(math.sqrt(d1)*math.sqrt(d2)) if d1!=0 and d2!=0 else 0

def cal_t_user_p_val(x):
    n=0
    d=0
    for i in x[1]:
        n+=i[0]*i[1]
        d+=i[1]
    return (x[0],n/d if d!=0 else 0)


target_users_dict = target_users_rdd.map(lambda x: mean_centric(x)).map(lambda x: key_value_conversion(x)).collect()
# Part 2 
print("**********************************************")
print("Step 2.2: Perform user-user collaborative filtering:")
user_user_collaborative_filtering_rdd = users_distinct_rdd.map(lambda x: mean_centric(x)).map(lambda x: key_value_conversion(x))

for key,val in target_users_dict:
    checkpoint_2_2_1 = user_user_collaborative_filtering_rdd.filter(lambda x: havecommon_2(x[1],val)).filter(lambda x:  x[0]!=key)
    checkpoint_2_2_1_rdd = checkpoint_2_2_1.map(lambda x: (x[0],(x[1],cosine_sim(val,x[1])))).filter(lambda x: x[1][1]>0)
    checkpoint_2_2_2_rdd = sc.parallelize(checkpoint_2_2_1_rdd.sortBy(lambda x: x[1][1],ascending=False).take(50))
    busid_not_in_target = list(set(distinct_business)-set(val.keys()))
    checkpoint_2_2_3_rdd = checkpoint_2_2_2_rdd.flatMap(lambda x: [(x[0],i,v,x[1][1]) for i,v in x[1][0].items()]).filter(lambda x: x[1] in busid_not_in_target)
    checkpoint_2_2_3_rdd = checkpoint_2_2_3_rdd.map(lambda x: (x[1],[[x[2][0],x[3]]])).reduceByKey(lambda x,y: x+y).filter(lambda x: len(x[1])>2).map(lambda x: cal_t_user_p_val(x))
    checkpoint_2_2_3 = checkpoint_2_2_3_rdd.collect()
    temp = sorted(checkpoint_2_2_3,key=lambda x:x[0])
    if len(temp)>10:
        temp=temp[:10]
    print("User ID:"+key+" Below are 10 predicted Business ID Rating:")
    print(temp)

print("**********************************************")
print("Step 2.3: Perform item-item collaborative filtering:")
def hasTargetUser(x):
    for k,v in x[1].items():
        if k in b_target_users.value:
            return True
    return False

def anytarget(x):
    for tu in target_users.keys():
        if tu in x[1].keys():
            return True
    return False

item_rdd = users_5_distinct_rdd.map(lambda x: (x[1],[[x[0],x[2]]])).reduceByKey(lambda x,y:x+y)
item_item_collaborative_filtering_rdd = item_rdd.map(lambda x: mean_centric(x)).map(lambda x: key_value_conversion(x))
res={}
for key,val in target_users_dict:
    list_B_rdd = item_item_collaborative_filtering_rdd.filter(lambda x: key not in x[1].keys()).filter(anytarget)
    list_B = sorted(list_B_rdd.map(lambda x: x[0]).collect())
    res[key]=[]
    for it in list_B:
        if len(res[key]) ==10:
            print(key)
            print(res[key])
            break
        v=list_B_rdd.filter(lambda x: x[0] ==it).map(lambda x: x[1]).collect()[0]
        cos_rdd = list_B_rdd.filter(lambda x: x[0]!=it).map(lambda x: (x[0],x[1],cosine_sim(x[1],v))).sortBy(lambda x: x[2],ascending=False)
        cos_rdd_top50 = sc.parallelize(cos_rdd.take(50))
        cos_rdd_top50_tuser = cos_rdd_top50.flatMap(lambda x:[(k,[[v[0],x[2]]]) for k,v in x[1].items()]).reduceByKey(lambda x,y:x+y).filter(lambda x: len(x[1])>=3).map(lambda x: cal_t_user_p_val(x))
        ans = cos_rdd_top50_tuser.filter(lambda x: x[0]==key).collect()
        if len(ans)>0:
            res[key].append(ans)
for key,val in res:
    print("User ID:"+str(key))
    print(sorted(val,key=lambda x: x[0])[:10])

