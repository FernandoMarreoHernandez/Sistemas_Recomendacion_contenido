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

#calculamos el idf comparando entre todos los documentos
for lines_num in matrix:
    #buscamos cuantas veces aparece la palabra en todos los documentos
    for word in matrix[lines_num]:
        print (word)
        #buscamos cuantas veces aparece la palabra en todos los documentos
        count = 0
        for lines_num2 in matrix:
            if word in matrix[lines_num2]:
                count += 1
        #calculamos el idf
        matrix[lines_num][word].append(np.log10(len(matrix)/count))

    
#calculamos el tf-idf
for doc in matrix:
    for word in matrix[doc]:
        matrix[doc][word].append(matrix[doc][word][1]*matrix[doc][word][2])

#sacamos los 5 mayores idfs
max_idf = []
for doc in matrix:
    for word in matrix[doc]:
        max_idf.append(matrix[doc][word][2])
max_idf = heapq.nlargest(5, max_idf)








sys.stdout = open("results/" + args.filename + "-result.txt", "w")
helpval = 0
#recorremos la matriz
for doc in matrix:
    print ("Documento " + str(helpval))
    print ("\t Palabra \t Indice \t TF \t IDF \t TF-IDF")
    #recorremos las palabras de cada documento
    for word in matrix[doc]:
        #recorremos los indices de cada palabra y los mostramos sin duplicados redondeando los valores a 3 decimales
        print ("\t " + str(word) + "\t\t" + str(matrix[doc][word][0]) + "\t\t" + str(round(matrix[doc][word][1],3)) + "\t" + str(round(matrix[doc][word][2],3)) + "\t" + str(round(matrix[doc][word][3],3)))
    helpval += 1
    print ("\n")
#mostramos la similaridad porfilas
print ("Similaridad entre filas")
print ("Documento \t Documento \t Similaridad")
for i in range(len(similarity)):
    print (str(similarity[i][0]) + "\t\t" + str(similarity[i][1]) + "\t\t" + str(similarity[i][2]))