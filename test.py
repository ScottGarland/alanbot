
import json
with open('datasets\data_intermediate.json') as fp:
    data1=json.load(fp)
# print(data1[0])
# print("\n\n")
# # if data1["dialog"]["id"]==0:
# #     print(data1["dialog"]["id"])
# print(data1[1]["dialog"])
# print("\n\n")
# print(data1[1]["dialog"][0]["text"])
# print("\n\n")
# print(data1[1]["user_profile"][1])
# print(data1[1]["bot_profile"][1])
#
# print(data1[1]["user_profile"])
# print(data1[1]["bot_profile"])
# # print(data1[3]["user_profile"])
# # print(data1[4]["bot_profile"])
# print("\n\n")
# print("USER")
# print(data1[5]["user_profile"])
#
# print("\n\n BOT")
# print(data1[5]["bot_profile"])
#
# print("\n\n")
# print("USER")
# print(data1[8]["user_profile"])
#
# print("\n\n BOT")
# print(data1[9]["bot_profile"])
# print(data1[0]["dialog"][0]["sender_class"])
# if data1[0]["dialog"][0]["sender_class"] =="Human":
#     print(data1[0]["dialog"][0]["text"])
# print(data1[1]["dialog"][0]["sender_class"])
# if data1[1]["dialog"][0]["sender_class"] =="Bot":
#     print(data1[1]["dialog"][0]["text"])
# print(data1[5]["dialog"][0]["sender_class"])
# if data1[5]["dialog"][0]["sender_class"] =="Human":
#     print(data1[5]["dialog"][0]["text"])
# print("FUCK")
# if data1[4]["dialog"][0]["sender_class"] =="Bot":
#     print(data1[4]["dialog"][0]["text"])
# print(data1[8]["dialog"][0]["sender_class"])
# if data1[6]["dialog"][0]["sender_class"] =="Human":
#     print(data1[6]["dialog"][0]["text"])
# print("FUCK")
# if data1[7]["dialog"][0]["sender_class"] =="Bot":
#     print(data1[7]["dialog"][0]["text"])
# print(data1[1])
# print(data1[2])
# print(data1[0]["dialog"][0]["text"])
# print(data1[0]["dialog"][0]["text"])
print(data1[3]["dialog"])
for x in data1[3]["dialog"]:
    print (x["text"])
