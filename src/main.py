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

def indices_count(palabra, documents):
    indices = []
    for i in range(len(documents)):
        if documents[i] == palabra:
            indices.append(i)
    return indices

#funcion que me dice en que posicion de la linea esta la palabra y si esta en varias lineas me dice todas
def pos_words(words_doc):
    #recorremos words_doc
    matrix = {}
    docinfo = {}
    helpval = 0
    for doc in words_doc:
        for word in doc:
            #si la palabra no esta en el diccionario la añadimos
            if word not in docinfo:
                #metemos la palabra en el diccionario
                docinfo[word] = [0]
                docinfo[word][0] = indices_count(word, words_doc[helpval])
        matrix[helpval] = copy.deepcopy(docinfo)
        helpval += 1
        docinfo = {}
    return matrix

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
    words = [x for x in words if x != ""]
    documents[i] = words
    
matrix = pos_words(documents)
#calculamos el tf
for doc in matrix:
    for word in matrix[doc]:
        matrix[doc][word].append(len(matrix[doc][word][0])/len(documents[doc]))

#calculamos el idf
for doc in matrix:
    for word in matrix[doc]:
        matrix[doc][word].append(np.log10(len(documents)/len(matrix[doc][word][0])))
    
#calculamos el tf-idf
for doc in matrix:
    for word in matrix[doc]:
        matrix[doc][word].append(matrix[doc][word][1]*matrix[doc][word][2])

#calculamos la similaridad coseno entre las filas
similarity = []
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        similarity.append([i,j,0])
        for word in matrix[i]:
            if word in matrix[j]:
                similarity[-1][2] += matrix[i][word][3]*matrix[j][word][3]
        similarity[-1][2] = similarity[-1][2]/(np.sqrt(sum([matrix[i][word][3]**2 for word in matrix[i]]))*np.sqrt(sum([matrix[j][word][3]**2 for word in matrix[j]])))




sys.stdout = open("results/" + args.filename + "-result.txt", "w")
helpval = 0
#recorremos la matriz
for doc in matrix:
    print ("Documento " + str(helpval))
    print ("\t Palabra \t Indice \t TF \t IDF \t TF-IDF")
    #recorremos las palabras de cada documento
    for word in matrix[doc]:
        #recorremos los indices de cada palabra y los mostramos sin duplicados
        print ("\t",word,"\t", str(list(set(matrix[doc][word][0]))), "\t", matrix[doc][word][1], "\t", matrix[doc][word][2], "\t", matrix[doc][word][3])
    helpval += 1
    print ("\n")
#mostramos la similaridad porfilas
print ("Similaridad entre filas")
print ("Documento \t Documento \t Similaridad")
for i in range(len(similarity)):
    print (str(similarity[i][0]) + "\t\t" + str(similarity[i][1]) + "\t\t" + str(similarity[i][2]))

