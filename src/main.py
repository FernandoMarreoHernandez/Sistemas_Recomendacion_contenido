'''
El software debe proporcionar como salida lo siguiente:
Para cada documento, tabla con las siguientes columnas:
Índice del término.
Término.
TF.
IDF.
TF-IDF.
Similaridad coseno entre cada par de documentos.
'''

import numpy as np
import heapq
import sys
import copy
import argparse
import json

def count_words(documents):
    words = {}
    for i, row in enumerate(documents):
        for word in row.split(" "):
            if word == "":
                continue
            if word not in words:
                words[word] = [0] * len(documents)
            words[word][i] += 1
    return words

parser = argparse.ArgumentParser(description='Process filename.')
parser.add_argument('-f','--filename', type=str, help='filename', required=True)
parser.add_argument('-s','--stopwords', type=str, help='stopword', required=True)
parser.add_argument('-c','--corpus', type=str, help='corpus', required=True)

args = parser.parse_args()

with open("data/documents/" + args.filename + ".txt", "r") as f:
    documents = f.readlines()

with open("data/stop-words/" + args.stopwords + ".txt", "r") as f:
    stopwords = f.readlines()

with open("data/corpus/" + args.corpus + ".txt", "r") as f:
    corpus = json.loads(f.read())

stopwords = [x.strip() for x in stopwords]

documents = [x.strip().replace(",","").replace(".","").replace(";","").replace("?","").replace("!","").replace("¡","").replace("¿","").replace("\"","").replace("\'","").lower() for x in documents]
for i, row in enumerate(documents):
    words = row.split(" ")
    for j, word in enumerate(words):
        if word in corpus:
            words[j] = corpus[word]
        if word in stopwords:
            words[j] = ""
    documents[i] = " ".join(words)


matrix = count_words(documents)
sys.stdout = open("results/" + args.filename + "-result.txt", "w")
for word in matrix:
    print(word, end="\t")
    for count in matrix[word]:
        print(count, end="\t")
    print("\n")