
import json
with open('datasets\data_intermediate.json') as fp:
    data1=json.load(fp)
print(data1[0])
# if data1["dialog"]["id"]==0:
#     print(data1["dialog"]["id"])
