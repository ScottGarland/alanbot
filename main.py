#Importing librarys
import discord
from discord.ext import commands

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
nltk.download()
import numpy
import tflearn
from tensorflow.python.framework import ops
import random
import json
import pickle
import tkinter
from tkinter import  *

def send():
    msg =EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg !='':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END,"You: "+msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = start(msg)
        ChatLog.insert(END,"Alan: "+ res +'\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

def bagOWords(string,words):
    bag=[0 for _ in range(len(words))]
    stringWords = nltk.word_tokenize(string)
    stringWords =[stemmer.stem(word.lower()) for word in stringWords]

    for x in stringWords:
        for i, y in enumerate(words):
            if y == x:
                bag[i]=1
    return numpy.array(bag)



def start(msg):
    # print('Welcome to Alan, Here you can begin your converstaion.(Enter "exit" to quit)')
    # while True:
    #     inp =input("User: ")
    #     if inp.lower()=="exit":
    #         break
    #
    #     conclusion=model.predict([bagOWords(inp,words)])
    #     conclusionIndex = numpy.argmax(conclusion)
    #     tag = labels[conclusionIndex]
    #     for x in data["Library"]:
    #         if x['tag'] == tag:
    #             responses =x['responses']
    #
    #     print(random.choice(responses))
    msg = msg.lower()
    conclusion=model.predict([bagOWords(msg,words)])[0]
    conclusionIndex = numpy.argmax(conclusion)
    tag = labels[conclusionIndex]
    print(conclusion[conclusionIndex])
    if conclusion[conclusionIndex]>0.7:
        for x in data["Library"]:
            if x['tag'] == tag:
                responses =x['responses']
        return random.choice(responses)
    else:
        return "Sorry I do not understand"


#Open json file
with open('library.json') as fp:
    data=json.load(fp)

try:
    with open("pickle.pickle","rb") as f:
        words, labels, learning, output = pickle.load(f)
except:
    #Creating arrays for  words labels and docx and docy
    #docx list of patternes
    #docy tag for words
    words =[]
    labels = []
    docsx =[]
    docsy=[]

    #Stemming words for json library
    for library in data["Library"]:
        for pattern in library["patterns"]:
            keyword = nltk.word_tokenize(pattern)
            words.extend(keyword)
            docsx.append(keyword)
            docsy.append(library["tag"])

            #if tag is in labels array do not add duplicates
            if library["tag"] not in labels:
                labels.append(library["tag"])
    words = [stemmer.stem(single.lower()) for single in words if single not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    learning = []
    output = []
    emptyOut = [0 for _ in range(len(labels))]

    # for loop over enumerate(docsx) to create a bag of words
    for x, doc in enumerate(docsx):
        bag = []

        keyword = [stemmer.stem(single) for single in doc]
        for single in words:
            if single in keyword:
                bag.append(1)
            else:
                bag.append(0)

        outputRow = emptyOut[:]
        outputRow[labels.index(docsy[x])] = 1
        learning.append(bag)
        output.append(outputRow)
        with open("pickle.pickle","wb") as f:
            pickle.dump((words, labels, learning, output),f)


learning = numpy.array(learning)
output = numpy.array(output)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(learning[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(learning, output, n_epoch=1000, batch_size=8, show_metric=True)

model.save("ChatbotModel.tflearn")

#start()

client = commands.Bot(command_prefix = '.')

@client.event
async def on_ready():
    print('Bot is ready.')

@client.command(aliases=['speak'])
async def talk(ctx, *, question):
    msg =question
    if msg !='':
        res = start(msg)
        await ctx.send(f'{res}')


client.run('ODIyMjU5MzY2OTE2MTI4Nzc4.YFPqtQ.nghNJQa-VeHJf63vLZiDF_aNVOY')
