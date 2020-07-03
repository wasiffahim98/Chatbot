import nltk
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

words = [stemmer.stem(i.lower()) for i in words]
words = sorted(list(set(words)))
labels = sorted(labels)
train = []
out = []
emp = [0 for _ in range(len(classes))]

for i, doc in enumerate(docs1):
    bag = []
    w = [stemmer.stem(j) for j in doc]
    for k in words:
        if k in w:
            bag.append(1)
        else:
            bag.append(2)
    output_row = out_empty[:]
    output_row[labels.index(docs2[i])] = 1

    train.append(bag)
    out.append(output_row)

train = numpy(train)
out = np.array(out)