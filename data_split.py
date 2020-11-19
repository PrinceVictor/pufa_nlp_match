import os
import json
import random
filename = "data/train.json"
all_data = json.loads(open(filename,'r',encoding='utf-8').read())
length = len(all_data)
test_length = int(length*0.20)
test = random.sample(all_data,test_length)
ids = set([i['question_id'] for i in test])
train = []
for d in all_data:
    if d['question_id'] not in ids:
        train.append(d)
test_nosql = [{'question_id':i['question_id'],'db_name':i['db_name'],'question':i['question']} for i in test]
with open("data/train/train.json",'w') as f:
    json.dump(train,f,ensure_ascii=False,indent=2)
with open("data/val/val.json",'w') as f:
    json.dump(test,f,ensure_ascii=False,indent=2)
with open("data/val/val_nosql.json",'w') as f:
    json.dump(test_nosql,f,ensure_ascii=False,indent=2)