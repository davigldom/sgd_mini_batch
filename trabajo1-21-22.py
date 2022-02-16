# ==========================================================
# Aprendizaje automático
# Máster en Ingeniería Informática - Universidad de Sevilla
# Curso 2021-22
# Primer trabajo práctico
# ===========================================================

# --------------------------------------------------------------------------
# APELLIDOS: Iglesias Domínguez
# NOMBRE: David
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo que
# debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite (e incluso se
# recomienda), pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de
# terceros, OBTENIDO A TRAVÉS DE LA RED o cualquier otro medio, se considerará
# plagio.

# Cualquier plagio o compartición de código que se detecte significará
# automáticamente la calificación de CERO EN LA ASIGNATURA para TODOS los
# estudiantes involucrados. Por tanto, NO se les conservará, para
# futuras convocatorias, ninguna nota que hubiesen obtenido hasta el
# momento. SIN PERJUICIO DE OTRAS MEDIDAS DE CARÁCTER DISCIPLINARIO QUE SE
# PUDIERAN TOMAR.
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES Y MÉTODOS
# QUE SE PIDEN





# ========================
# IMPORTANTE: USO DE NUMPY
# ========================

# SE PIDE USAR NUMPY EN LA MEDIDA DE LO POSIBLE.

import numpy as np
import warnings

# SE PENALIZARÁ el uso de bucles convencionales si la misma tarea se puede
# hacer más eficiente con operaciones entre arrays que proporciona numpy.

# PARTICULARMENTE IMPORTANTE es el uso del método numpy.dot.
# Con numpy.dot podemos hacer productos escalares de pesos por características,
# y extender esta operación de manera compacta a dos dimensiones, cuando tenemos
# varias filas (ejemplos) e incluso varios varios vectores de pesos.

# En lo que sigue, los términos "array" o "vector" se refieren a "arrays de numpy".

# NOTA: En este trabajo NO se permite usar scikit-learn (salvo en el código que
# se proporciona para cargar los datos).

# -----------------------------------------------------------------------------

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aa.zip y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn). Todos los datos se cargan en arrays de numpy.

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresista (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos.

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.

# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn. Como vocabulario,
#   se han usado las 609 palabras que ocurren más frecuentemente en las distintas
#   críticas. Los datos se cargan finalmente en las variables X_train_imdb,
#   X_test_imdb, y_train_imdb,y_test_imdb.

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles).




# ===========================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA (HOLDOUT)
# ===========================================================

# Definir una función

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test, y conservando la correspondencia
# original entre los ejemplos y sus valores de clasificación.
# La división ha de ser ALEATORIA y ESTRATIFICADA respecto del valor de clasificación.

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# In[1]: Xe_votos,Xp_votos,ye_votos,yp_votos
#            =particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# In[2]: y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
# Out[2]: (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# In[3]: np.unique(y_votos,return_counts=True)
# Out[3]: (array(['democrata', 'republicano'], dtype='<U11'), array([267, 168]))
# In[4]: np.unique(ye_votos,return_counts=True)
# Out[4]: (array(['democrata', 'republicano'], dtype='<U11'), array([178, 112]))
# In[5]: np.unique(yp_votos,return_counts=True)
# Out[5]: (array(['democrata', 'republicano'], dtype='<U11'), array([89, 56]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con más de dos clases:

# In[6]: Xe_credito,Xp_credito,ye_credito,yp_credito
#              =particion_entr_prueba(X_credito,y_credito,test=0.4)

# In[7]: np.unique(y_credito,return_counts=True)
# Out[7]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([202, 228, 220]))

# In[8]: np.unique(ye_credito,return_counts=True)
# Out[8]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([121, 137, 132]))

# In[9]: np.unique(yp_credito,return_counts=True)
# Out[9]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([81, 91, 88]))
# ------------------------------------------------------------------

# Primero se cargan todas las variables con los datos, para ser utilizadas a lo largo de los ejercicios
from carga_datos import *


def particion_entr_prueba(X, y, test=0.20):
    classes = np.unique(y) # Se obtiene las clases del conjunto de datos (ya sean 2 o más)
    test_indices = np.empty(0, int) # Se crea un array vacío de tipo int para almacenar los índices de los ejemplos para test
    rng = np.random.default_rng() # Se inicializa un generador de números random de numpy

    for cl in classes: # Por cada una de las clases encontradas
        cl_array = np.where(y == cl)[0] # Se obtienen los valores que sean igual a la clase a iterar
        test_examples = round(len(cl_array) * test) # Se calcula cuantos ejemplos de esta clase hay que escoger para mantener la proporción

        # Se escogen aleatoriamente los índices del array de la clase que se van a almacenar para test
        random_indexes = rng.choice(len(cl_array), test_examples, replace=False, shuffle=False)

        # Se obtienen los índices del conjunto completo que van para test, se ordenan y se añaden al array correspondiente
        test_indices = np.sort(np.append(test_indices, cl_array[random_indexes]))

    # Se borran los ejemplos con los índices que van para test, tanto en X como en y
    X_train = np.delete(X, test_indices, axis=0)
    y_train = np.delete(y, test_indices, axis=0)

    # Se obtienen los ejemplos de test, a partir de sus índices
    X_test = np.array(X)[test_indices]
    y_test = np.array(y)[test_indices]

    return X_train, X_test, y_train, y_test


# ===========================================
# EJERCICIO 2: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# Se pide implementar el clasificador de regresión logística mini-batch
# a través de una clase python, que ha de tener la siguiente estructura:

# class RegresionLogisticaMiniBatch():

#    def __init__(self,normalizacion=False,
#                 rate=0.1,rate_decay=False,batch_tam=64,
#                 pesos_iniciales=None):

#          .....

#    def entrena(self,entr,clas_entr,n_epochs=1000,
#                reiniciar_pesos=False):

#         ......

#     def clasifica_prob(self,E):


#         ......

#     def clasifica(self,E):


#         ......


# Explicamos a continuación cada uno de los métodos:


# * Constructor de la clase:
# --------------------------

#  El constructor debe tener los siguientes argumentos de entrada:


#  - El parámetro normalizacion, que puede ser True o False (False por
#    defecto). Indica si los datos se tienen que normalizar, tanto para el
#    entrenamiento como para la clasificación de nuevas instancias.  La
#    normalización es una técnica que suele ser útil cuando los distintos
#    atributos reflejan cantidades numéricas de muy distinta magnitud.
#    En ese caso, antes de entrenar se calcula la media m_i y la desviación
#    típica d_i en CADA COLUMNA i (es decir, en cada atributo) de los
#    datos del conjunto de entrenamiento.  A continuación, y antes del
#    entrenamiento, esos datos se transforman de manera que cada componente
#    x_i se cambia por (x_i - m_i)/d_i. Esta MISMA transformación se realiza
#    sobre las nuevas instancias que se quieran clasificar.

#  - rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  - rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula:
#       rate_n= (rate_0)*(1/(1+n))
#    donde n es el número de epoch, y rate_0 es la cantidad
#    introducida en el parámetro rate anterior.

#  - batch_tam: indica el tamaño de los mini batches (por defecto 64)
#    que se usan para calcular cada actualización de pesos.

#  - pesos_iniciales: Si es None, los pesos iniciales se inician
#    aleatoriamente. Si no, debe proporcionar un array de pesos que se
#    tomarán como pesos iniciales.

#

# * Método entrena:
# -----------------

#  Este método es el que realiza el entrenamiento del clasificador.
#  Debe calcular un vector de pesos, mediante el correspondiente
#  algoritmo de entrenamiento basado en ascenso por el gradiente mini-batch,
#  para maximizar la log verosimilitud. Describimos a continuación los parámetros de
#  entrada:

#  - entr y clas_entr, son los datos del conjunto de entrenamiento y su
#    clasificación, respectivamente. El primero es un array (bidimensional)
#    con los ejemplos, y el segundo un array (unidimensional) con las clasificaciones
#    de esos ejemplos, en el mismo orden.

#  - n_epochs: número de pasadas que se realizan sobre todo el conjunto de
#    entrenamiento.

#  - reiniciar_pesos: si es True, se reinicia al comienzo del
#    entrenamiento el vector de pesos de manera aleatoria
#    (típicamente, valores aleatorios entre -1 y 1).
#    Si es False, solo se inician los pesos la primera vez que se
#    llama a entrena. En posteriores veces, se parte del vector de
#    pesos calculado en el entrenamiento anterior. Esto puede ser útil
#    para continuar el aprendizaje a partir de un aprendizaje
#    anterior, si por ejemplo se dispone de nuevos datos.

#  NOTA: El entrenamiento en mini-batch supone que en cada epoch se
#  recorren todos los ejemplos del conjunto de entrenamiento,
#  agrupados en grupos del tamaño indicado. Por cada uno de estos
#  grupos de ejemplos se produce una actualización de los pesos.
#  Se pide una VERSIÓN ESTOCÁSTICA, en la que en cada epoch se asegura que
#  se recorren todos los ejemplos del conjunto de entrenamiento,
#  en un orden ALEATORIO, aunque agrupados en grupos del tamaño indicado.


# * Método clasifica_prob:
# ------------------------

#  Método que devuelve el array de correspondientes probabilidades de pertenecer
#  a la clase positiva (la que se ha tomado como clase 1), para cada ejemplo de un
#  array E de nuevos ejemplos.



# * Método clasifica:
# -------------------

#  Método que devuelve un array con las correspondientes clases que se predicen
#  para cada ejemplo de un array E de nuevos ejemplos. La clase debe ser una de las
#  clases originales del problema (por ejemplo, "republicano" o "democrata" en el
#  problema de los votos).


# Si el clasificador aún no ha sido entrenado, tanto "clasifica" como
# "clasifica_prob" deben devolver una excepción del siguiente tipo:

class ClasificadorNoEntrenado(Exception): pass


# Ejemplos de uso:
# ----------------



# CON LOS DATOS VOTOS:

#

# En primer lugar, separamos los datos en entrenamiento y prueba (los resultados pueden
# cambiar, ya que esta partición es aleatoria)


# In [1]: Xe_votos,Xp_votos,ye_votos,yp_votos
#            =particion_entr_prueba(X_votos,y_votos)

# Creamos el clasificador:

# In [2]: RLMB_votos=RegresionLogisticaMiniBatch()

# Lo entrenamos sobre los datos de entrenamiento:

# In [3]: RLMB_votos.entrena(Xe_votos,ye_votos)

# Con el clasificador aprendido, realizamos la predicción de las clases
# de los datos que estan en test:

# In [4]: RLMB_votos.clasifica_prob(Xp_votos)
# array([3.90234132e-04, 1.48717603e-11, 3.90234132e-04, 9.99994374e-01, 9.99347533e-01,...])

# In [5]: RLMB_votos.clasifica(Xp_votos)
# Out[5]: array(['democrata', 'democrata', 'democrata','republicano',... ], dtype='<U11')

# Calculamos la proporción de aciertos en la predicción, usando la siguiente
# función que llamaremos "rendimiento".

def rendimiento(clasif, X, y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]

# In [6]: rendimiento(RLMB_votos,Xp_votos,yp_votos)
# Out[6]: 0.9080459770114943

# ---------------------------------------------------------------------

# CON LOS DATOS DEL CÀNCER

# Hacemos un experimento similar al anterior, pero ahora con los datos del
# cáncer de mama, y usando normalización y disminución de la tasa

# In[7]: Xe_cancer,Xp_cancer,ye_cancer,yp_cancer
#           =particion_entr_prueba(X_cancer,y_cancer)


# In[8]: RLMB_cancer=RegresionLogisticaMiniBatch(normalizacion=True,rate_decay=True)

# In[9]: RLMB_cancer.entrena(Xe_cancer,ye_cancer)

# In[9]: RLMB_cancer.clasifica_prob(Xp_cancer)
# Out[9]: array([9.85046885e-01, 8.77579844e-01, 7.81826115e-07,..])

# In[10]: RLMB_cancer.clasifica(Xp_cancer)
# Out[10]: array([1, 1, 0,...])

# In[11]: rendimiento(RLMB_cancer,Xp_cancer,yp_cancer)
# Out[11]: 0.9557522123893806


def sigmoid(z):
    e_z = np.exp(-z)
    sig = 1 / (1 + e_z)

    return sig


class RegresionLogisticaMiniBatch():

    def __init__(self, normalizacion=False, rate=0.1, rate_decay=False, batch_tam=64, pesos_iniciales=None):
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.pesos_iniciales = pesos_iniciales
        self.rng = np.random.default_rng() # Generador de número aleatorios que se utilizará
        self.weights = None  # Variable para almacenar los pesos tras entrenamiento
        self.classes = None # Variable para almacenar las clases originales del conjunto "y"

    def entrena(self, entr, clas_entr, n_epochs=1000, reiniciar_pesos=False):
        self.classes = np.unique(clas_entr) # Se almacenan las clases que componen el conjunto de datos
        pesos_iniciales = None # Se inicializa la variable para los pesos_iniciales

        if self.pesos_iniciales is not None: # Si se han introducido unos pesos iniciales al construir el clasificador
            pesos_iniciales = np.array(self.pesos_iniciales) # Se guardan los pesos iniciales en su variable

            if len(pesos_iniciales) != len(entr[0]) + 1: # Se comprueba que tenga la longitud adecuada
                raise ValueError("La longitud del array de pesos iniciales debe ser igual al número de características más 1 (para w0).")

            if len(pesos_iniciales[(pesos_iniciales > 1) | (pesos_iniciales < -1)]) != 0: # Se comprueba que estén entre los valores adecuados
                raise ValueError("Los valores de los pesos iniciales deben estar entre -1 y 1.")

        if isinstance(clas_entr[0], str): # Si las clases no son numéricas
            clas_entr = np.where(clas_entr==self.classes[0], 0, 1) # Se cambia la clase negativa a 0 y la positiva a 1

        if self.normalizacion is True: # Si se ha indicado que se normalicen los datos
            if not isinstance(entr[0][0], int) and not isinstance(entr[0][0], float): # Se comrpueba que los valores a normalizar sean números
                raise ValueError("Los datos del conjunto de entrenamiento no son numéricos: no se puede aplicar normalización.")

            std = np.std(entr, axis=0) # Se calcula la desviación típica de cada una de las características
            mean = np.mean(entr, axis=0) # Se calcula la media de cada una de las características
            entr = np.divide(np.subtract(entr, mean), std) # Al array con los ejemplos se le resta el array con las medias y se divide entre el de las desviaciones

        if reiniciar_pesos is True or self.weights is None: # Si se ha indicado que se reinicien los pesos o es la primera vez que se entrena
            # Se generan unos pesos aleatorios entre -1 y 1, o se cogen los pesos iniciales que se hayan introducido
            weights = 2 * self.rng.random(len(entr[0]) + 1) - 1 if pesos_iniciales is None else pesos_iniciales

        else: # En caso contrario, se empieza con los pesos anteriores
            weights = self.weights

        for epoch in np.arange(n_epochs): # Por cada uno de los epochs que se ha indicado
            # Se actualiza la tasa de aprendizaje si se ha indicado que se realice un descenso
            rate = self.rate * (1 / (1 + epoch)) if self.rate_decay is True else self.rate
            perm_index = self.rng.permutation(len(entr)) # Se realiza una permutación de los índices del conjunto de entrenamiento

            # Se dividen los índices en batches del tamaño indicado
            batches = np.split(perm_index, np.arange(self.batch_tam, len(entr), self.batch_tam))

            for batch in batches: # Por cada uno de los batches anteriores
                # Se obtienen los ejemplos correspondientes de "X" a partir de los índices, insertando un 1 como x0 de cada ejemplo (para w0)
                X_batch = np.insert(entr[batch], 0, 1, axis=1)
                y_batch = clas_entr[batch] # Se obtienen los ejemplos correspondientes de "y" a partir de los índices

                # Se realiza la actualización de los pesos según la fórmula: wi <- wi + rate * sumatorio[(yj - sigmoide(w * xj)) * xij]
                dot = X_batch.dot(weights)
                sigma = y_batch - sigmoid(dot)
                summation = sigma.dot(X_batch)
                weights = weights + rate * summation

        self.weights = weights # Tras el entrenamiento, se almacenan los pesos para utilizarlos en los métodos de clasificación

    def clasifica_prob(self, E):
        if self.weights is None: # Si no hay pesos almacenados, significa que no se ha entrenado el modelo
            raise ClasificadorNoEntrenado()

        if self.normalizacion is True: # Si se ha indicado que se quiere normalizar los datos
            if not isinstance(E[0][0], int) and not isinstance(E[0][0], float): # Se comprueban que sean numéricos
                raise ValueError("Los datos del conjunto a clasificar no son numéricos: no se puede aplicar normalización.")

            # Se realiza el mismo procedimiento de normalización que en el método entrena()
            std = np.std(E, axis=0)
            mean = np.mean(E, axis=0)
            E = np.divide(np.subtract(E, mean), std)

        # Se inserta un 1 como x0 de cada ejemplo, para que al realizar w * x quede como w0 * 1 + w1 * x1...
        E = np.insert(E, 0, 1, axis=1)

        # Para calcular las probabilidades de pertenecer a la clase positiva
        dot = np.array(E).dot(self.weights) # Se realiza el producto escalar w * x de cada ejemplo
        probs_array = sigmoid(dot) # Se aplica la función sigmoide

        return probs_array

    def clasifica(self, E):
        if self.weights is None: # Si no hay pesos almacenados, significa que no se ha entrenado el modelo
            raise ClasificadorNoEntrenado()

        probs_array = self.clasifica_prob(E) # Se obtiene el array con las probabilidades de pertenecer a la clase positiva

        # Se crea el array de prediciones con la clase positiva donde la probabilidad sea mayor o igual que 0.5, y la negativa en el resto de los casos
        y_pred = np.where(probs_array >= 0.5, self.classes[1], self.classes[0])

        return y_pred


Xe_votos, Xp_votos, ye_votos, yp_votos = particion_entr_prueba(X_votos, y_votos)

RLMB_votos = RegresionLogisticaMiniBatch()
RLMB_votos.entrena(Xe_votos, ye_votos)

votos_probs_array = RLMB_votos.clasifica_prob(Xp_votos)
print(votos_probs_array)

votos_y_pred = RLMB_votos.clasifica(Xp_votos)
print(votos_y_pred)

votos_score = rendimiento(RLMB_votos, Xp_votos, yp_votos)
print(votos_score) # OUT: 0.9770114942528736


Xe_cancer, Xp_cancer, ye_cancer, yp_cancer = particion_entr_prueba(X_cancer, y_cancer)

RLMB_cancer = RegresionLogisticaMiniBatch(normalizacion=True, rate_decay=True)
RLMB_cancer.entrena(Xe_cancer, ye_cancer)

cancer_probs_array = RLMB_cancer.clasifica_prob(Xp_cancer)
print(cancer_probs_array)

cancer_y_pred = RLMB_cancer.clasifica(Xp_cancer)
print(cancer_y_pred)

cancer_score = rendimiento(RLMB_cancer, Xp_cancer, yp_cancer)
print(cancer_score) # OUT: 0.9823008849557522


# =================================================
# EJERCICIO 3: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================

# Este ejercicio vale 2 PUNTOS (SOBRE 10) pero se puede saltar, sin afectar
# al resto del trabajo. Puede servir para el ajuste de parámetros en los ejercicios
# posteriores, pero si no se realiza, se podrían ajustar siguiendo el método "holdout"
# implementado en el ejercicio 1.

# La técnica de validación cruzada que se pide en este ejercicio se explica
# en el tema "Evaluación de modelos".

# Definir una función:

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador. El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cáncer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad,
# no tiene por qué coincidir exactamente el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#             {"batch_tam":16,"rate_decay":True},Xe_cancer,ye_cancer,n=5)
# 0.9121095227289917


# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones deben ser aleatorias y estratificadas.

# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> LR16=RegresionLogisticaMiniBatch(batch_tam=16,rate_decay=True)
# >>> LR16.entrena(Xe_cancer,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(LR16,Xp_cancer,yp_cancer)
# 0.9203539823008849

#------------------------------------------------------------------------------


def particion_validacion_cruzada(X, y, n):
    # Listado para almacenar los listados con los índices de cada fold (no puede ser array de numpy ya que los folds pueden no tener la misma longitud)
    folds_indexes = []

    rng = np.random.default_rng() # Se inicializa el generador de números aleatorios
    classes = np.unique(y) # Se obtienen las distintas clases del conjunto de datos

    for cl in classes: # Por cada una de las clases
        cl_array = np.where(y == cl)[0] # Se obtiene un array con los ejemplos de dicha clase
        perm_indexes = rng.permutation(cl_array) # Se realiza una permutación aleatoria de dicho array

        cl_folds = np.array_split(perm_indexes, n) # Se divide en tantas partes como se haya indicado (n) para la validación cruzada

        if len(folds_indexes) == 0: # Si es la primera clase
            folds_indexes = cl_folds # Se almacena tal cual en la variable correspondiente

        else:
            # IMPORTANTE: Debido a que los folds pueden tener un número distinto de ejemplos (resultado de no ser divisible los ejemplos del entrenamiento entre n)
            # el array o list de arrays de numpy pasa a ser de tipo "objeto", ya que los arrays multidimensionales de numpy deben tener la misma longitud para todas
            # sus filas. Debido a que es del tipo "objeto", el atributo "shape" solo devuelve el número de filas, por lo que no se puede utilizar los métodos como
            # concatenate o sort en el axis=1. Por tanto, la única solución es iterar con un bucle for convencional por cada una de las filas
            folds_indexes = [np.sort(np.concatenate((folds_indexes[row], cl_folds[row]))) for row in range(len(cl_folds))]

    # Se obtienen los listados ("X" e "y") con los ejemplos divididos entre los distintos folds
    X_folds = [X[fold] for fold in folds_indexes]
    y_folds = [y[fold] for fold in folds_indexes]

    return X_folds, y_folds


def rendimiento_validacion_cruzada(clase_clasificador, params, X, y, n=5):
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) # Se desactiva los warnings de creación de arrays 2D con distintos tamaños

    scores = np.array([]) # Se crea un array para guardar los rendimientos obtenidos
    RLMB_cv = clase_clasificador(**params) # Se crea el clasificador indicado con los parámetros indicados

    X_folds, y_folds = particion_validacion_cruzada(X, y, n) # Se obtienen los ejemplos dividos entre los distintos folds

    for iter in np.arange(n):
        # IMPORTANTE: Al igual que en el método anterior, aquí también se tiene que iterar por cada fila mediante un for clásico, debido a que cabe la posibilidad
        # de que las filas de este array multidimensional de numpy tengan distinto tamaño. Esto sería avisado mediante un warning por consola, al llamar al método
        # np.delete sobre "X_folds" o "y_folds", pero se han deshabilitado dichos warnings para no ensuciar la salida por consola.
        X_train_folds = np.delete(X_folds, iter, axis=0) # Se borra el fold que se utilizará en esta iteración para las pruebas
        X_train = X_train_folds[0] # Se almacena el primer fold tal cual
        for t_fold in X_train_folds[1:]: # Por cada uno de los otros folds
            X_train = np.concatenate((X_train, t_fold)) # Se concatenan en X_train (el objetivo final es pasar de un conjunto de arrays 2D a uno solo)

        # Se realiza el mismo procedimiento para el conjunto "y"
        y_train_folds = np.delete(y_folds, iter, axis=0)
        y_train = y_train_folds[0]
        for t_fold in y_train_folds[1:]:
            y_train = np.concatenate((y_train, t_fold))

        # Se obtienen los ejemplos (tanto el conjunto "X" como el "y") para pruebas que corresponden a esta iteración
        X_test = X_folds[iter]
        y_test = y_folds[iter]

        RLMB_cv.entrena(X_train, y_train) # Se entrena el clasificador creado con los conjuntos de entrenamiento de esta iteración

        score = rendimiento(RLMB_cv, X_test, y_test) # Se calcula el rendimiento del conjunto de pruebas de esta iteración
        scores = np.append(scores, score, axis=None) # Se almacena el rendimiento en el array correspondiente

    return np.mean(scores) # Se devuelve la media de los rendimientos obtenidos


# ===================================================
# EJERCICIO 4: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando los dos modelos implementados en el ejercicio 3, obtener clasificadores
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama
# - Críticas de películas en IMDB

# Ajustar los parámetros para mejorar el rendimiento. Si se ha hecho el ejercicio 3,
# usar validación cruzada para el ajuste (si no, usar el "holdout" del ejercicio 1).

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos.

# Para que el tiempo de ejecución no sea demasiado excesivo, y considerando que no se especifica qué parámetros hay que ajustar y que
# el rate se va a ajustar en el ejercicio 6.2, se va a probar con combinar "rate_decay" True o False con "batch_tam" entre 16, 32 y 64
params_grid = [
    {"batch_tam": 16, "rate_decay": True},
    {"batch_tam": 32, "rate_decay": True},
    {"batch_tam": 64, "rate_decay": True},
    {"batch_tam": 16, "rate_decay": False},
    {"batch_tam": 32, "rate_decay": False},
    {"batch_tam": 64, "rate_decay": False},
]

# Votos de congresistas US ("Xe_votos", "ye_votos", "Xp_votos" e "yp_votos" se han obtenido en el Ejercicio 2)
cv_scores = np.array([])

for params in params_grid:
    cv_score = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, params, Xe_votos, ye_votos, n=5)
    cv_scores = np.append(cv_scores, cv_score, axis=None)

print(cv_scores) # OUT: [0.95983193 0.95109244 0.95689076 0.95117647 0.93689076 0.94806723]

# Se puede observar que el rendimiento medio más alto lo hemos conseguido (en esta ejecución) para la combinación 1,
# es decir, "batch_tam" igual a 16 y "rate_decay" igual a True. Vamos a enternar ahora el clasificador y ver su rendimiento
# sobre el conjunto de pruebas
RLMB_cv_votos = RegresionLogisticaMiniBatch(batch_tam=16, rate_decay=True)
RLMB_cv_votos.entrena(Xe_votos, ye_votos)
cv_votos_score = rendimiento(RLMB_cv_votos, Xp_votos, yp_votos)
print(cv_votos_score) # OUT: 0.9770114942528736

# Cáncer de mama ("Xe_cancer", "ye_cancer", "Xp_cancer" e "yp_cancer" se han obtenido en el Ejercicio 2)
cv_scores = np.array([])

for params in params_grid:
    cv_score = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, params, Xe_cancer, ye_cancer, n=5)
    cv_scores = np.append(cv_scores, cv_score, axis=None)

print(cv_scores) # OUT: [0.91889632 0.91227903 0.90590062 0.88380315 0.90355948 0.91877688]

# Se puede observar que el rendimiento medio más alto lo hemos conseguido (en esta ejecución) también para la combinación 1,
# es decir, "batch_tam" igual a 16 y "rate_decay" igual a True. Vamos a enternar ahora el clasificador y ver su rendimiento
# sobre el conjunto de pruebas
RLMB_cv_cancer = RegresionLogisticaMiniBatch(batch_tam=16, rate_decay=True)
RLMB_cv_cancer.entrena(Xe_cancer, ye_cancer)
cv_cancer_score = rendimiento(RLMB_cv_cancer, Xp_cancer, yp_cancer)
print(cv_cancer_score) # OUT: 0.929203539823008

# Críticas de películas en IMDB
cv_scores = np.array([])

for params in params_grid:
    cv_score = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, params, X_train_imdb, y_train_imdb, n=5)
    cv_scores = np.append(cv_scores, cv_score, axis=None)

print(cv_scores) # OUT: [0.80903698 0.81951327 0.81651449 0.77502816 0.7665606  0.77202068]

# Se puede observar que el rendimiento medio más alto lo hemos conseguido (en esta ejecución) para la combinación 2,
# es decir, "batch_tam" igual a 32 y "rate_decay" igual a True. Vamos a enternar ahora el clasificador y ver su rendimiento
# sobre el conjunto de pruebas
RLMB_cv_imdb = RegresionLogisticaMiniBatch(batch_tam=32, rate_decay=True)
RLMB_cv_imdb.entrena(X_train_imdb, y_train_imdb)
cv_imdb_score = rendimiento(RLMB_cv_imdb, X_test_imdb, y_test_imdb)
print(cv_imdb_score) # OUT: 0.765


# =====================================
# EJERCICIO 5: CLASIFICACIÓN MULTICLASE
# =====================================

# Técnica "One vs Rest" (Uno frente al Resto)
# -------------------------------------------


# Se pide implementar la técnica "One vs Rest" (Uno frente al Resto),
# para obtener un clasificador multiclase a partir del clasificador
# binario definido en el apartado anterior.


#  En concreto, se pide implementar una clase python
#  RegresionLogisticaOvR con la siguiente estructura, y que implemente
#  el entrenamiento y la clasificación siguiendo el método "One vs
#  Rest" tal y como se ha explicado en las diapositivas del módulo.



# class RegresionLogisticaOvR():

#    def __init__(self,normalizacion=False,rate=0.1,rate_decay=False,
#                 batch_tam=64):

#          .....

#    def entrena(self,entr,clas_entr,n_epochs=1000):

#         ......

#    def clasifica(self,E):


#         ......



#  Los parámetros de los métodos significan lo mismo que en el
#  apartado anterior.

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------

# In[1] Xe_iris,Xp_iris,ye_iris,yp_iris
#            =particion_entr_prueba(X_iris,y_iris,test=1/3)

# >>> rl_iris=RL_OvR(rate=0.001,batch_tam=20)

# >>> rl_iris.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9797979797979798

# >>> rendimiento(rl_iris,Xp_iris,yp_iris)
# >>> 0.9607843137254902
# --------------------------------------------------------------------

class RegresionLogisticaOvR():
    def __init__(self, normalizacion=False, rate=0.1, rate_decay=False, batch_tam=64):
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.rng = np.random.default_rng() # Generador de números aleatorios
        self.classes = None # Variable para almacenar las clases originales en el conjunto de datos
        self.classifiers = None # Variable para almacenar cada uno de los clasificadores que se van a entrenar

    def entrena(self, entr, clas_entr, n_epochs=1000, reiniciar_pesos=False):
        classes = np.unique(clas_entr) # Se obtienen las clases del conjunto de datos
        self.classes = np.unique(clas_entr) # Se almacenan también en la variable del modelo
        self.classifiers = np.array([]) # Se inicializa un array para almacenar los distintos clasificadores

        if isinstance(y_credito[0], str): # Si las clases vienen dadas por strings
            classes = np.arange(len(self.classes)) # Se almacenan como valores numéricos en la variable correspondiente
            condlist = [cl == clas_entr for cl in self.classes] # Se crea un listado de condiciones para buscar los valores en string
            clas_entr = np.select(condlist, classes) # Se cambia cada string por su valor numérico

        if self.normalizacion is True: # Si se ha indicado que se realice una normalización de los datos
            if not isinstance(entr[0][0], int) and not isinstance(entr[0][0], float): # Se comrpueba que sean numéricos
                raise ValueError("Los datos del conjunto de entrenamiento no son numéricos: no se puede aplicar normalización.")

            # Se realiza el mismo procedimiento de normalización que en el ejercicio 2
            std = np.std(entr, axis=0)
            mean = np.mean(entr, axis=0)
            entr = np.divide(np.subtract(entr, mean), std)

        for cl in classes: # Por cada una de las clases (valores numéricos) que tenemos en el conjunto de datos
            k_clas_entr = np.where(clas_entr==cl, 1, 0) # Se pone como 1 (positiva) la clase a iterar y el resto como 0 (negativa)
            k_RLMB = RegresionLogisticaMiniBatch(False, self.rate, self.rate_decay, self.batch_tam, None) # Se crea un clasificador binario
            k_RLMB.entrena(entr, k_clas_entr, n_epochs, reiniciar_pesos) # Se entrena con el array de clasificaciones modificado
            self.classifiers = np.append(self.classifiers, k_RLMB) # Se almacena en el array de clasificadores

    def clasifica(self, E):
        # Si el array de clasificadores es None o está vacío, significa que no se ha entrenado el modelo
        if self.classifiers is None or self.classifiers.size == 0:
            raise ClasificadorNoEntrenado()

        probs_matrix = np.array([]) # Se crea un array para almacenar la probabilidad de pertenecer a la clase positiva en cada uno de los clasificadores

        for classifier in self.classifiers: # Por cada clasificador binario entrenado
            probs_array = classifier.clasifica_prob(E) # Se obtiene el array con las probabilidades de pertenecer a la clase positiva

            if probs_matrix.size == 0: # Si es el primer array a almacenar
                probs_matrix = probs_array # Se guarda directamente en la variable

            else: # En caso contrario
                probs_matrix = np.vstack((probs_matrix, probs_array)) # Se utiliza el método vstack para crear una nueva fila en la matriz correspondiente

        # El array con las prediciones vendrá dado por el índice del clasificador que mayor probabilidad ha devuelto de pertenecer a su clase positiva
        y_pred = np.argmax(probs_matrix, axis=0)

        if isinstance(self.classes[0], str): # Si las clases originalmente eran string
            condlist = [cl == y_pred for cl in np.arange(len(self.classes))] # Se crea un listado de condiciones para buscar cada valor numérico de las clases
            y_pred = np.select(condlist, self.classes) # Se cambian los valores numéricos de las clases por sus string

        return y_pred


Xe_iris, Xp_iris, ye_iris, yp_iris = particion_entr_prueba(X_iris, y_iris, test=1/3)

RLOVR_iris = RegresionLogisticaOvR(rate=0.001, batch_tam=20)

RLOVR_iris.entrena(Xe_iris, ye_iris)

e_iris_score = rendimiento(RLOVR_iris, Xe_iris, ye_iris)
print(e_iris_score) # OUT: 0.9595959595959596

p_iris_score = rendimiento(RLOVR_iris, Xp_iris, yp_iris)
print(p_iris_score) # OUT: 0.9803921568627451


# ==============================================
# EJERCICIO 6: APLICACION A PROBLEMAS MULTICLASE
# ==============================================


# ---------------------------------------------------------
# 6.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación del apartado anterior, para obtener un
# clasificador que aconseje la concesión, estudio o no concesión de un préstamo,
# basado en los datos X_credito, y_credito. Ajustar adecuadamente los parámetros.

# NOTA IMPORTANTE: En este caso concreto, los datos han de ser transformados,
# ya que los atributos de este conjunto de datos no son numéricos. Para ello, usar la llamada
# "codificación one-hot", descrita en el tema "Preprocesado e ingeniería de características".
# Se pide implementar esta transformación (directamete, SIN USAR Scikt Learn ni Pandas).


def one_hot(array):
    # Por cada una de las filas del array traspuesto (de esta forma, cada fila sería una característica) que se quiere codificar como one_hot
    for char_values in array.T:
        # IMPORTANTE: Al ser string, numpy no compara adecuadamente la igualdad si invocas a np.unique de todo el array traspuesto, indicando
        # axis=1. Es por ello que se itera mediante un for, además que facilita el realizar las siguientes operaciones
        # Obtenemos los valores únicos que tiene dicha característica, así como los índices para volver a construir esta fila (de esta forma,
        # tendremos un valor numérico que nos indicará qué valor único tenía cada elemento)
        unique, unique_inverse = np.unique(char_values, return_inverse=True)

        # Creamos una matriz identidad con el número de columnas y filas igual al número de valores únicos de la característica y obtenemos la
        # fila que corresponda al valor numérico anterior (por ejemplo, si es 2 y teníamos 3 valores únicos obtendremos la última fila: [0 0 1])
        char_onehot = np.eye(unique.shape[0])[unique_inverse]

        # Añadimos cada codificación one_hot al final de su fila en el array que queríamos transformar, y borramos el primer elemento (funcionaría
        # parecido a una pila: cogemos la primera característica, la codificamos en one_hot, la ponemos al final y borramos su valor inicial)
        array = np.hstack((array, char_onehot))
        array = np.delete(array, 0, axis=1)

    return array.astype(float) # Devolvemos el array con los valores como números


Xe_credito, Xp_credito, ye_credito, yp_credito = particion_entr_prueba(X_credito, y_credito)

Xe_credito = one_hot(Xe_credito)
Xp_credito = one_hot(Xp_credito)

cv_scores = np.array([])

for params in params_grid:
    cv_score = rendimiento_validacion_cruzada(RegresionLogisticaOvR, params, Xe_credito, ye_credito, n=5)
    cv_scores = np.append(cv_scores, cv_score, axis=None)

print(cv_scores) # OUT: [0.74043842 0.74445084 0.72893483 0.71963084 0.70770349 0.70565392]

# Se puede observar que el rendimiento medio más alto lo hemos conseguido (en esta ejecución) para la combinación 2,
# es decir, "batch_tam" igual a 32 y "rate_decay" igual a True. Vamos a enternar ahora el clasificador y ver su rendimiento
# sobre el conjunto de pruebas, modificando el "rate" de inicio para intentar mejorar el rendimiento
RLMB_cv_credito = RegresionLogisticaOvR(batch_tam=32, rate_decay=True)
RLMB_cv_credito.entrena(Xe_credito, ye_credito)
cv_credito_score = rendimiento(RLMB_cv_credito, Xp_credito, yp_credito)
print(cv_credito_score) # OUT: 0.7615384615384615

# Como los valores han sido muy bajos, podemos probar a modificar el rate inicial para ver si conseguimos una mejora
RLMB_cv_credito2 = RegresionLogisticaOvR(batch_tam=32, rate_decay=True, rate=0.01)
RLMB_cv_credito2.entrena(Xe_credito, ye_credito)
cv_credito_score2 = rendimiento(RLMB_cv_credito2, Xp_credito, yp_credito)
print(cv_credito_score2) # OUT: 0.7307692307692307
# Al parecer, no obtenemos mejora al cambiar la tasa de aprendizaje inicial. La falta de rendimiento se podría deber
# a la separación realizada para entrenamiento y pruebas, o tal vez debido a que todas las características han sido
# codificadas como one_hot y esta implementación de este clasificador no alcanza buen rendimiento en este caso, además
# que se podría realizar un preprocesado para mejorar el rendimiento


# ---------------------------------------------------------
# 6.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación o implementaciones del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador.

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test).

import zipfile

def extract_zip(file_name, dest_path):
    zip = zipfile.ZipFile(file_name, "r") # Se inicializa un objeto de tipo ZipFile, de solo lectura, a partir de la ruta indicada del archivo

    try:
        zip.extractall(path=dest_path) # Se extraen todos los archivos del zip en la ruta especificada

    except:
        # Si hay un error (habitualmente porque no existe el archivo o directorio) se lanza una excepción
        raise FileNotFoundError("No se ha podido extraer correctamente el fichero digitdata.zip")

    zip.close() # Se cierra el archivo .zip


def extract_X_data(file_name):
    data_file = open(file_name, "r") # Se abre el fichero correspondiente

    file_content = data_file.read() # Se lee su contenido
    # Se sustituyen los espacios en blanco por 0 y los "+" y "#" por 1
    file_content = file_content.replace(" ", "0").replace("+", "1").replace("#", "1")
    # Se separa el contenido por sus lineas y cada linea se separa en los distintos números que la componen
    lines_list = [list(line) for line in file_content.splitlines()]

    data_array = np.array(lines_list).astype(int) # Se crea un array de numpy (de tipo numérico) con los datos obtenidos

    # Se redimensiona el array para que cada ejemplo lo compongan 28 filas con sus 28 valores en cada una (cada imagen es de 28x28)
    data_array = data_array.reshape(int(data_array.shape[0]/28), 28*28)

    data_file.close() # Se cierra el fichero

    return data_array


def extract_y_data(file_name):
    data_file = open(file_name, "r") # Se abre el fichero correspondiente

    file_content = data_file.read() # Se lee su contenido
    lines_list = file_content.splitlines() # Se divide en cada una de las filas

    data_array = np.array(lines_list).astype(int) # Se crea un array de numpy (de tipo numérico) con las filas anteriores

    data_file.close() # Se cierra el archivo

    return data_array


# Extracción de los distintos conjuntos de datos utilizando los métodos anteriores
# extract_zip("datos/digitdata.zip", "datos/digitdata/")
X_train_digits = extract_X_data("datos/digitdata/trainingimages")
y_train_digits = extract_y_data("datos/digitdata/traininglabels")
X_test_digits = extract_X_data("datos/digitdata/testimages")
y_test_digits = extract_y_data("datos/digitdata/testlabels")
X_valid_digits = extract_X_data("datos/digitdata/validationimages")
y_valid_digits = extract_y_data("datos/digitdata/validationlabels")

# Como el tiempo de ejecución es muy grande, se va a probar a realizar una única ejecución con el conjunto de datos completo
RLMB_digits = RegresionLogisticaOvR()
RLMB_digits.entrena(X_train_digits, y_train_digits)

valid_digits_score = rendimiento(RLMB_digits, X_valid_digits, y_valid_digits)
print(valid_digits_score) # OUT: 0.82

test_digits_score = rendimiento(RLMB_digits, X_test_digits, y_test_digits)
print(test_digits_score) # OUT: 0.77

# Entonces, para ajustar los parámetros, se va a utilizar un conjunto de datos menor (solo con el 10% de los ejemplos)
X_train_digits = X_train_digits[:500]
y_train_digits = y_train_digits[:500]
X_test_digits = X_test_digits[:100]
y_test_digits = y_test_digits[:100]
X_valid_digits = X_valid_digits[:100]
y_valid_digits = y_valid_digits[:100]

# Primero vamos a probar a entrenar un modelo con sus parámetros por defecto
RLMB_digits2 = RegresionLogisticaOvR()
RLMB_digits2.entrena(X_train_digits, y_train_digits)

valid_digits_score2 = rendimiento(RLMB_digits2, X_valid_digits, y_valid_digits)
print(valid_digits_score2) # OUT: 0.86

test_digits_score2 = rendimiento(RLMB_digits2, X_test_digits, y_test_digits)
print(test_digits_score2) # OUT: 0.78

# Con este primer intento, ya hemos conseguido alcanzar un 0,75 de rendimiento sobre pruebas, pero probemos ahora a cambiar
# el tamaño de los batches y a que la tasa de aprendizaje decaiga en cada epoch
RLMB_digits3 = RegresionLogisticaOvR(batch_tam=32, rate_decay=True)
RLMB_digits3.entrena(X_train_digits, y_train_digits)

valid_digits_score3 = rendimiento(RLMB_digits3, X_valid_digits, y_valid_digits)
print(valid_digits_score3) # OUT: 0.85

test_digits_score3 = rendimiento(RLMB_digits3, X_test_digits, y_test_digits)
print(test_digits_score3) # OUT: 0.78

# En este caso, no hemos conseguido una mejora en el conjunto de validación, aunque sí una muy leve sobre
# el de pruebas. Vamos ahora a probar a variar la tasa de aprendizaje inicial, manteniendo los otros parámetros
RLMB_digits4 = RegresionLogisticaOvR(batch_tam=32, rate_decay=True, rate=0.01)
RLMB_digits4.entrena(X_train_digits, y_train_digits)

valid_digits_score4 = rendimiento(RLMB_digits4, X_valid_digits, y_valid_digits)
print(valid_digits_score4) # OUT: 0.7

test_digits_score4 = rendimiento(RLMB_digits4, X_test_digits, y_test_digits)
print(test_digits_score4) # OUT: 0.67

# Ambos rendimientos han decaído bastante, por lo que ha sido una mala opción entrenar el modelo con estos parámetros.
# Como último intento, veamos qué ocurre si cambiamos la tasa inicial, pero con rate_decay a False y un tam_batch de 64
RLMB_digits5 = RegresionLogisticaOvR(batch_tam=64, rate_decay=False, rate=0.01)
RLMB_digits5.entrena(X_train_digits, y_train_digits)

valid_digits_score5 = rendimiento(RLMB_digits5, X_valid_digits, y_valid_digits)
print(valid_digits_score5) # OUT: 0.84

test_digits_score5 = rendimiento(RLMB_digits5, X_test_digits, y_test_digits)
print(test_digits_score5) # OUT: 0.8

# Al parecer este sí ha sido un buen ajuste, ya que hemos conseguido el mejor rendimiento hasta el momento para el
# conjunto de pruebas, al mismo tiempo que obteniendo un rendimiento para validación casi como el mejor obtenido
