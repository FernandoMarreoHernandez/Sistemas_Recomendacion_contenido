# Sistemas de recomendación. Modelos basados en el contenido

## Índice

- [Instrucciones de instalación](#instrucciones-de-instalación)
- [Descripción del código](#descripción-del-código)
  - [Limpieza del documento](#limpieza-del-documento)
  - [Conteo de palabras por fila](#conteo-de-palabras-por-fila)
  - [Cálculo del TF](#cálculo-del-tf)
  - [Cálculo del IDF](#cálculo-del-idf)
  - [Cálculo del TF-IDF](#cálculo-del-tf-idf)
  - [Cálculo de similitudes](#cálculo-de-similitudes)
  - [Imprimir los resultados](#imprimir-los-resultados)
- [Ejecución del programa](#ejecución-del-programa)
- [Ejemplo de ejecución](#ejemplo-de-ejecución)

En esta práctica vamos a implementar un sistema de recomendación siguiendo el modelo basado en el contenido.

## Instrucciones de instalación

Las librerías necesarias para la ejecución del programa son las siguientes:

- numpy
- sys
- argparse

En caso de no contar con alguna de estas librerías, se debe instalar haciendo uso de **pip**

```
pip install <package>
```

## Descripción del código

### Limpieza del documento

En primer lugar, necesitamos eliminar del documento los caractéres y palabras que no nos interese. Esto son símbolos de puntuación, palabras de parada, sinónimos o derivados de otras palabras, etc.

```python
documents = [x.strip().replace(",","").replace(".","").replace(";","").replace("?","").replace("!","").replace("¡","").replace("¿","").replace("\"","").replace("\'","").replace(":","").lower() for x in documents]
```

En primer lugar, eliminamos por cada fila del documento los signos de puntuación que puedan aparecer.

```python
for i, row in enumerate(documents):
    words = row.split(" ")
    for j, word in enumerate(words):
        if word in corpus:
            words[j] = corpus[word]
        if word in stopwords:
            words[j] = ""
    words = [x for x in words if x != ""]
    documents[i] = words
```

Luego, usaremos los ficheros de palabras de parada y corpus para sustituirlas por caracteres vacíos. De esta forma, tenemos en cada fila únicamente las palabras relevantes para la recomendación.

### Conteo de palabras por fila

Para calcular cuántas veces aparece una palabra en una fila y en qué posición, hemos creado la función `pos_words`

```python
def pos_words(words_doc):
    matrix = {}
    for i, doc in enumerate(words_doc):
        docinfo = {}
        for word in doc:
            docinfo[word] = [indices_count(word, doc)]
        matrix[i] = docinfo
    return matrix
```

La función llama a `indices_count` para contar las veces que aparece un término en una fila. Llamaremos a esta función por cada palabra y almacenaremos su información en una matriz.

```python
def indices_count(palabra, documents):
    indices = []
    for i in range(len(documents)):
        if documents[i] == palabra:
            indices.append(i)
    return indices
```

En cuanto a la anterior mencionada función, esta se encarga de recorrer la fila y buscar la palabra en cuestión. Por cada coincidencia, añade el índice a una lista.

De esta forma, hemos conseguido almacenar las veces que aparece los términos en cada una de las filas.

### Cálculo del TF

Para calcular el TF de cada termino en cada fila, tendremos que hacer uso de la siguiente fórmula:

$$ TF*{ij} = 1 + log10(x*{ij})$$

Siendo $x_{ij}$ el número de veces que aparece el término $i$ en el documento $j$.

```python
for doc in matrix:
    for word in matrix[doc]:
        tf = 1 + np.log10(len(matrix[doc][word][0]))
        matrix[doc][word].append(tf)
```

Siguiendo la fórmula descrita anteriormente, almacenamos cada valor obtenido en la matriz de resultados junto al término y fila al que corresponde.

### Cálculo del IDF

Para calcular el IDF, tendremos que seguir la siguiente fórmula para cada término:

$$IDF_i = log10(N/df_i)$$

Siendo $N$ el número total de documentos y $df_i$ el número de documentos en los que aparece el término $i$.

```python
for lines_num in matrix:
    for word in matrix[lines_num]:
        count = 0
        for lines_num2 in matrix:
            if word in matrix[lines_num2]:
                count += 1
        matrix[lines_num][word].append(np.log10(len(matrix)/count))
```

En primer lugar, recorre cada una de las filas para contar el número de veces que aparece cada término. Luego, almacena en la matriz el IDF en función a lo anterior mencionado y al número total de documentos.

### Cálculo del TF-IDF

Para calcular el TF-IDF, se sigue la siguiente fórmula

$$TF-IDF_{ij} = TF_{ij} \cdot IDF_i$$

Siendo $TF_{ij}$ el TF del término $i$ en el documento $j$ y $IDF_i$ el IDF del término $i$.

```python
for doc in matrix:
    for word in matrix[doc]:
        matrix[doc][word].append(matrix[doc][word][1]*matrix[doc][word][2])

```

Este fragmento de código simplemente sigue la fórmula antes descrita y la añade a la matriz de resultados.

### Cálculo de similitudes

Para el cálculo de simililtudes tendremos que, en primer lugar, hallar la longitud de cada documento.

```python
vector_length = []
for doc in matrix:
    sum = 0
    for word in matrix[doc]:
        sum += matrix[doc][word][3]**2
    vector_length.append(np.sqrt(sum))
```

Para cada documento, tendremos que calcular la raíz de la suma de los cuadrados de cada uno de los TF en una misma fila. Esto nos resulta en una lista con cada longitud de documento. Con estos valores, podremos normalizar la matriz de TF.

```python
for i, doc in enumerate(matrix):
    for word in matrix[doc]:
        matrix[doc][word].append(matrix[doc][word][3]/vector_length[doc])
```

Para ello, dividimos cada valor de TF entre su respectiva longitud de documento. Con esto, hemos conseguido normalizar la matriz.

```python
similarity = []
for i in range(len(matrix)):
    for j in range(i+1, len(matrix)):
        sum = 0
        for word in matrix[i]:
            if word in matrix[j]:
                sum += matrix[i][word][4]*matrix[j][word][4]
        similarity.append([i, j, sum])
```

Finalmente, para hallar las similitudes de cada fila con las otras, se calcula la suma de los productos entre cada valor normalizado correspondiente al mismo término en distintas filas. El resultado lo almacenaremos en una matriz de similitudes.

### Imprimir los resultados

Por último, vamos a almacenar los resultados en un fichero txt con:

- Índices en los que aparece cada término por documento
- TF de cada término
- IDF de cada término
- TF-IDF de cada término
- Similiradidad coseno de cada fila

````python
sys.stdout = open("results/" + args.filename + "-result.txt", "w")
helpval = 0
for doc in matrix:
    print ("Documento " + str(helpval))
    print ("\t Palabra \t Indice \t TF \t IDF \t TF-IDF")
    for word in matrix[doc]:
        print ("\t " + str(word) + "\t\t" + str(matrix[doc][word][0]) + "\t\t" + str(round(matrix[doc][word][1],5)) + "\t" + str(round(matrix[doc][word][2],5)) + "\t" + str(round(matrix[doc][word][3],5)))
    helpval += 1
    print ("\n")
print ("Similaridad entre filas")
print ("Documento \t Documento \t Similaridad")
for i in range(len(similarity)):
   print ("[" + str(similarity[i][0]) + "]\t\t[" + str(similarity[i][1]) + "]\t\t" + str(similarity[i][2]))```
````

De esta forma imprimimos los resultados en un fichero dentro de la carpeta results.

## Ejecución del programa

Para ejecutar el programa, se debe ejecutar el siguiente comando:

```
python main.py -f <filename> -s <stopwords> -c <corpus>
```

Donde:

- filename: Nombre del fichero a procesar
- stopwords: Nombre del fichero de palabras de parada
- corpus: Nombre del fichero de corpus

## Ejemplo de ejecución

Si ejecutamos el programa con los siguientes parámetros:

```
python main.py -f documents01 -s stop-words-en -c corpus-en
```

Se almacenarán los resultados en el fichero `documents01-result.txt` dentro de la carpeta results. En el fichero se puede observar la matriz de resultados y la matriz de similitudes.
