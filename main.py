import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)

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

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(out[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.fit(train, out, n_epoch=10000, batch_size=8, show_metric=True)
model.save("model.tflearn")