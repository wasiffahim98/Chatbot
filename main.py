import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as fil:
        words, labels, train, out = pickle.load(fil)
except:
    words = []
    labels = []
    docs1 = []
    docs2 = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            docs1.append(pattern)
            docs2.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(i.lower()) for i in words if i != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
    train = []
    out = []
    emp = [0 for _ in range(len(labels))]

    for i, doc in enumerate(docs1):
        bag = []
        w = [stemmer.stem(j) for j in doc]
        for k in words:
            if k in w:
                bag.append(1)
            else:
                bag.append(2)
        output_row = emp[:]
        output_row[labels.index(docs2[i])] = 1

        train.append(bag)
        out.append(output_row)

    train = numpy.array(train)
    out = numpy.array(out)

    with open("data.pickle", "wb") as fil:
        pickle.dump((words, labels, train, out), fil)

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(out[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(train, out, n_epoch=10000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, w):
    bag = [0 for _ in range(len(w))]
    s_w = nltk.word_tokenize(s)
    s_w = [stemmer.stem(word.lower()) for word in s_w]

    for se in s_w:
        for i, j in enumerate(w):
            if j == se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("Start talking!")
    while True:
        user = input("You: ")
        if user.lower() == "quit":
            break

        results = model.predict([bag_of_words(user, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for t in data["intents"]:
            if t['tag'] == tag:
                responses = t['responses']
        
        print(random.choice(responses))
chat()