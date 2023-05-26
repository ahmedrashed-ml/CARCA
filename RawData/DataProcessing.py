import gzip
from collections import defaultdict
from datetime import datetime
import array
import numpy as np
import pickle
import pandas as pd


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x= pickle.load(f)
    except:
        x = []
    return x

def save_data(data,filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

dataset_name = 'Video_Games'
f = open('reviews_' + dataset_name + '.txt', 'w')
for l in parse('reviews_' + dataset_name + '.json.gz'):
    line += 1
    f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    countU[rev] += 1
    countP[asin] += 1
f.close()


usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
User = dict()
for l in parse('reviews_' + dataset_name + '.json.gz'):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    if countU[rev] < 5 or countP[asin] < 5:
        continue

    if rev in usermap:
        userid = usermap[rev]
    else:
        usernum += 1
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemnum += 1
        itemid = itemnum
        itemmap[asin] = itemid
    User[userid].append([time, itemid])
# sort reviews in User according to time

for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])

print (usernum, itemnum)

f = open(dataset_name + '_cxt.txt', 'w')
for user in User.keys():
    for i in User[user]:
        f.write('%d %d %s\n' % (user, i[1], i[0]))
f.close()

f = open(dataset_name + '.txt', 'w')
for user in User.keys():
    for i in User[user]:
        f.write('%d %d\n' % (user, i[1]))
f.close()


#### Reading and writing features
itemfeat_dict = {}
counter = 0
for l in parse('meta_' + dataset_name + '.json.gz'):
    line += 1
    asin = l['asin']

    title = ""
    if 'description' in l.keys():
        title = l['description']

    price = 0.0
    if 'price' in l.keys():
        price = float(l['price'])       

    brand = ""
    if 'brand' in l.keys():
        brand = l['brand']

    categories = l['categories'][0]
    #print(price , "-",brand , "-",categories , "-" )
    if asin in itemmap.keys():
        itemid = itemmap[asin]
        itemfeat_dict[itemid] = [title,price,brand,categories]
        counter = counter + 1

features_list = list()
templist = ["",0.0,"",[]]
for item_id in range(1,itemnum+1):
    if item_id in itemfeat_dict.keys():
        features_list.append(itemfeat_dict[item_id])
    else:
        features_list.append(templist)


df = pd.DataFrame(features_list, columns=['title','price','brand','categories'])

del df['title']
df['categoriesstring'] = [' '.join(map(str, l)) for l in df['categories']]
df=pd.concat([df,df['categoriesstring'].str.get_dummies(sep=' ').add_prefix('cat_').astype('int8')],axis=1) 
del df['categories']
del df['categoriesstring']
df=pd.get_dummies(df,dummy_na=True)

print(df.head())
print(df.dtypes)
save_data(df.values,dataset_name+'_feat.dat')


###
