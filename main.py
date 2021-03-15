#Importing librarys
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

#Open json file
with open("library.json") as file:
    data=json.load(file)

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
    for library in data["library"]:
        for pattern in library["patterns"]:
            keyword = nltk.word_tokenize(pattern)
            words.extend(keyword)
            docsx.append(keyword)
            docsy.append(libray["tag"])

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

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(learning[0])])
net = tflearnfor.fully_connected(net,8) #hidden layer
net = tflearnfor.fully_connected(net,8) #hidden layer
net = tflearnfor.fully_connected(net,len(output[0]), activation="softmax")
model = tflearn.DNN(net)
try:
    model.load("ChatbotModel.tflearn")
except:
    model.fit(learning, output, n_epoch=4000, batch_size=8, show_metric=True)

    model.save("ChatbotModel.tflearn")

def bagOWords(string,words):
    bag=[0 for _ in range(len(words))]
    stringWords = nltk.word_tokenize(string)
    stringWords =[stemmer.stem(word.lower()) for word in stringWords]

    for x in stringWords:
        for i, y in enumerate(words):
            if y == x:
                bag[i]=1
    return numpy.array(bag)



def start():
    print('Welcome to Alan, Here you can begin your converstaion.(Enter "exit" to quit)')
    while True:
        input =input("User: ")
        if input.lower()=="exit":
            break

        conclusion=model.predict([bagOWords(input,words)])
        conclusionIndex = numpy.argmax(conclusion)
        tag = labels[conclusionIndex]
        for x in data["libray"]:
            if x['tag'] == tag:
                responses =x['responses']

        print(random.choice(responses))


start()
