import numpy as np
import sys
import argparse
import json

#funcion que cuenta cuantas veces aparece una palabra en un documento
def indices_count(palabra, documents):
    indices = []
    for i in range(len(documents)):
        if documents[i] == palabra:
            indices.append(i)
    return indices

#funcion que me dice en que posicion de la linea esta la palabra y si esta en varias lineas me dice todas
def pos_words(words_doc):
    matrix = {}
    for i, doc in enumerate(words_doc):
        docinfo = {}
        for word in doc:
            docinfo[word] = [indices_count(word, doc)]
        matrix[i] = docinfo
    return matrix

#argumentos que introducimos por linea de comando
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

#limpiamos los documentos quitando todos los caracteres que no sean letras y pasandolo a minusculas
documents = [x.strip().replace(",","").replace(".","").replace(";","").replace("?","").replace("!","").replace("¡","").replace("¿","").replace("\"","").replace("\'","").replace(":","").lower() for x in documents]
#eliminamos las stopwords y reemplazamos las palabras por su indice en el corpus
for i, row in enumerate(documents):
    words = row.split(" ")
    for j, word in enumerate(words):
        if word in corpus:
            words[j] = corpus[word]
        if word in stopwords:
            words[j] = ""
    words = [x for x in words if x != ""]
    documents[i] = words

#invocamos a la funcion para sacar cuantas veces aparece cada palabra en los documentos    
matrix = pos_words(documents)

#calculamos el tf
for doc in matrix:
    for word in matrix[doc]:
        tf = 1 + np.log10(len(matrix[doc][word][0]))
        matrix[doc][word].append(tf)

#calculamos el idf comparando entre todos los documentos
for lines_num in matrix:
    for word in matrix[lines_num]:
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


#calculamos la longitud de cada documento
vector_length = []
for doc in matrix:
    sum = 0
    for word in matrix[doc]:
        sum += matrix[doc][word][3]**2
    vector_length.append(np.sqrt(sum))

#ordenamos los tf-idf de mayor a menor valor
for doc in matrix:
    matrix[doc] = {k: v for k, v in sorted(matrix[doc].items(), key=lambda item: item[1][3], reverse=True)}
    

# normalizamos los tf-idf
for i, doc in enumerate(matrix):
    for word in matrix[doc]:
        matrix[doc][word].append(matrix[doc][word][3]/vector_length[doc])

#calculamos la similaridad entre filas
similarity = []
for i in range(len(matrix)):
    for j in range(i+1, len(matrix)):
        sum = 0
        for word in range(5):
            #multiplicamos los tf-idf de de iguales posicion dentro de cada documento
            sum += matrix[i][list(matrix[i].keys())[word]][4] * matrix[j][list(matrix[j].keys())[word]][4]
        similarity.append([i, j, sum])
    

#creamos el documento resultado y guardamos los resultados
sys.stdout = open("results/" + args.filename + "-result.txt", "w")
helpval = 0
#recorremos la matriz
for doc in matrix:
    print ("Documento " + str(helpval))
    print ("\t Palabra \t Indice \t TF \t IDF \t TF-IDF")
    #recorremos las palabras de cada documento
    for word in matrix[doc]:
        #recorremos los indices de cada palabra y los mostramos redondeando los valores a 5 decimales
        print ("\t " + str(word) + "\t\t" + str(matrix[doc][word][0]) + "\t\t" + str(round(matrix[doc][word][1],5)) + "\t" + str(round(matrix[doc][word][2],5)) + "\t" + str(round(matrix[doc][word][3],5)))
    helpval += 1
    print ("\n")
#mostramos la similaridad porfilas
print ("Similaridad entre filas")
print ("Documento \t Documento \t Similaridad")
for i in range(len(similarity)):
   print ("[" + str(similarity[i][0]) + "]\t\t[" + str(similarity[i][1]) + "]\t\t" + str(similarity[i][2]))