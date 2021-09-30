#!/usr/bin/env python
# coding: utf-8

# # <img src="imagenes\logo uem.png" width="300" height="600" align="left "><b>PROYECTO OPEN DATA II     -          *Universidad Europea de Madrid*
# 
#    > ## <font color='blue'>Predictor de precios de venta de viviendas en Ames, Iowa, EEUU</font>  
# <div style=" color:#000000; font-style: normal; font-family: Georgia;">
#     Alumnos: Carlos García y Víctor Salvador

# <img src="imagenes\ames.jpg" width="1000" height="600">

# *** 

# ## 1. Importación y limpieza de datos

# ### 1.1 Librerías importadas 

# In[1]:


import warnings   #Control de advertencias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# ### 1.2 Datos importados

# In[2]:


train=pd.read_csv('./train.csv',index_col=['Id'])
test_ns=pd.read_csv('./test.csv',index_col=['Id'])
SalePrice=pd.read_csv('./sample_submission.csv',names=['Id','SalePrice'],skiprows=1,index_col=['Id'])
test=pd.concat((test_ns,SalePrice),axis=1) #distinguiremos el train y el test más adelante

house=pd.concat((train,test),sort=False).reset_index(drop=True)
print(f"Total size is {house.shape}")
print(house.columns)


# ### 1.3 Limpieza de datos <img src="imagenes\limpieza datos 3.png" width="80" height="100" align="left ">

# #### Eliminación de campos demasiado vacíos

# In[3]:


h=house.dropna(thresh=len(house)*0.8,axis=1)
print(f'{house.shape[1]-h.shape[1]} campos eliminados del dataset') 


# In[4]:


allna = (h.isnull().sum() / len(h))*100             
allna = allna.drop(allna[allna == 0].index).sort_values()
NA=h[allna.index.to_list()]


# #### Relleno de valores vacíos

# In[5]:


NAcat=NA.select_dtypes(include='object') #no que no es numérico
NAnum=NA.select_dtypes(exclude='object') #lo que es numérico
print(f'Tenemos {NAcat.shape[1]} campos categóricos con valores vacíos')
print(f'Tenemos {NAnum.shape[1]} campos numéricos con valores vacíos')


# In[6]:


NAnum.columns


# In[7]:


NCol=[['MasVnrArea'],['BsmtFinSF2'],['BsmtFullBath'],['BsmtHalfBath'],
      ['BsmtUnfSF'],['TotalBsmtSF'],['BsmtFinSF1'],['GarageCars'],['GarageArea']]

for col in NCol:
    h[col]= h[col].fillna(0) #Rellenamos los vacíos con 0 porque Na significa la ausencia de estas características
#media
h['LotFrontage']=h['LotFrontage'].fillna(h.LotFrontage.mean()) #Rellenamos los valores vacíos con la media
h['GarageYrBlt']=h["GarageYrBlt"].fillna(h.GarageYrBlt.median()) #Rellenamos los varloes vacíos con la mediana


# In[8]:


def filling_NA(data, columns, METHOD='ffill'): #Función para rellenar columnas 
    fill_cols = columns
    
    for col in data[fill_cols]:
        data[col]= data[col].fillna(method=METHOD) 
        #ffill significa 'forward fill' y propagará la última observación válida hacia adelante.
    
    return data

ffill_cols = ['Electrical', 'SaleType', 'KitchenQual', 'Exterior1st',
             'Exterior2nd', 'Functional', 'Utilities', 'MSZoning']


hh=filling_NA(h, ffill_cols)


# In[9]:


NAcols=hh.columns
for col in NAcols:
    if hh[col].dtype == "object":
        hh[col] = hh[col].fillna("None")


# In[10]:


hh.isnull().sum().sort_values(ascending=False).head()


# Ningun valor vacío en nuestro dataset

# #### Agrupación de campos relacionados

# In[11]:


hh['TotalArea'] = hh['TotalBsmtSF'] + hh['1stFlrSF'] + hh['2ndFlrSF'] + hh['GrLivArea'] +hh['GarageArea']

hh['Bathrooms'] = hh['FullBath'] + hh['HalfBath']*0.5 

hh['YearAverage']= (hh['YearRemodAdd']+hh['YearBuilt'])/2


# In[12]:


hh['PrecioVenta']=hh['SalePrice']


# In[13]:


hh=hh.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea',
        'FullBath','HalfBath','YearRemodAdd','YearBuilt','SalePrice'], axis=1)


# #### Eliminación de las columnas de tipo objeto

# Por la naturaleza de los algoritmos de regresión que utilizaremos más adelante, prescindiremos de las columnas tipo object. A excepción del campo **Neighborhood**, debido a su relevancia ya demostrada en la parte exploratoria de este proyecto; la cual será **discretizada** a continuación.

# In[14]:


hh


# Como vamos a transformar la variable Neighborhood de categórica a numérica, ordenamos el campo por el precio de venta para facilitar el entrenamiento de los modelos posteriores.

# In[15]:


aa=pd.DataFrame(hh['Neighborhood'])
aa['PrecioVenta']=hh['PrecioVenta']
aa.groupby(['Neighborhood']).mean().sort_values('PrecioVenta')


# In[16]:


hh['Neighborhood']=hh['Neighborhood'].replace({'BrDale':1,'MeadowV':2,'BrkSide':3,
                                                              'IDOTRR':4,'Blueste':5,'Edwards':6,
                                                              'OldTown':7,'NPkVill':8,'SWISU':9,
                                                              'Sawyer':10,'NAmes':11,'Mitchel':12,
                                                              'Blmngtn':13,'SawyerW':14,'NWAmes':15,
                                                              'Gilbert':16,'CollgCr':17,'Somerst':18,
                                                              'Crawfor':19,'ClearCr':20,'Veenker':21,
                                                              'Timber':22,'NridgHt':23,'StoneBr':24,
                                                              'NoRidge':25})


# In[17]:


hh.Neighborhood.dtype


# In[18]:


vars_train=hh.select_dtypes(exclude='object')


# ## 2.  Importar pyspark

# Apache Spark es una plataforma de **computación distribuida** de código abierto, algunas de las ventajas que justifican su uso son:
# * **Velocidad en materia de aprendizaje automático**: permite a los programadores realizar operaciones sobre un gran volumen de datos en clústeres de forma rápida y con tolerancia a fallos
# * **Distintas plataformas** para gestionar y procesar datos, como Spark SQL, Spark Streaming, Mlib o Graph X.
# <img src="imagenes\pyspark2.png" width="600" height="300" align="center ">

# ### 2.1 Preparación de las variables

# In[19]:


import pyspark.sql.types as typ


# In[20]:


vars_train.dtypes


# In[21]:


vars_train.head(5)


# A continuación creamos un **label personalizado** con los tipos asignados manualmente y con los campos de tipo object excluidos salvo la variable barrio la cual ha sido discretizada.

# In[22]:


labels = [
    ('TipoDeVienda', typ.IntegerType()),
    ('LongitudDeLaCalle', typ.FloatType()),
    ('AreaDelTerreno', typ.IntegerType()),
    ('Barrio', typ.IntegerType()),
    ('CalidadDeLaVivienda', typ.IntegerType()),
    
    ('CondicionesDeLaVivienda', typ.IntegerType()),
    ('PiesCuadradosDeFachada', typ.FloatType()),
    ('PiesCuadradosDeSotanoTerminados', typ.FloatType()),
    ('PiesCuadradosDeSotano2Terminados', typ.FloatType()),
    ('PiesCuadradosDeSotanoNoTerminados', typ.FloatType()),
    
    ('PiesCuadradosDeBajaCalidad', typ.IntegerType()),
    ('BaniosEnterosEnSotano', typ.FloatType()),
    ('BaniosPequeniosEnSotano', typ.FloatType()),
    ('DormitoriosSobreSuelo', typ.IntegerType()),
    ('CocinasSobresSuelo', typ.IntegerType()),
    
    ('habitacionesSobreSueloNoBanios', typ.IntegerType()),
    ('Chimeneas', typ.IntegerType()),
    ('AnioDeConstruccionDelGaraje', typ.FloatType()),
    ('CochesDelGaraje', typ.FloatType()),
    ('PiesCuadradosDeTerrazaDeMaderaSobreSuelo', typ.IntegerType()),
    
    ('PiesCuadradosDePorcheAbierto', typ.IntegerType()),
    ('PiesCuadradosDePorcheCerrado', typ.IntegerType()),
    ('PorcheThreeSeasson', typ.IntegerType()),
    ('PorcheAcristalado', typ.IntegerType()),
    ('AreaDePiscina', typ.IntegerType()),
    
    ('PrecioDeLosMiscelaneos', typ.IntegerType()),
    ('MesDeVenta', typ.IntegerType()),
    ('AnioDeVenta', typ.IntegerType()),
    ('AreaHabitableTotal', typ.FloatType()),
    ('Banios', typ.FloatType()),
    
    ('EdadMediaCasa', typ.FloatType()),
    ('PrecioDeLaVivenda', typ.FloatType())
]


schema = typ.StructType([
    typ.StructField(e[0], e[1], False) for e in labels
])


# ### 2.2 SparkSession

# Inicializamos la sesión en spark, este es el punto de entrada a todas las funciones de Spark

# In[23]:


import findspark
findspark.init()

from pyspark.sql import *
from pyspark import SparkContext

spark=SparkSession.builder.getOrCreate()
sc=spark.sparkContext
sc


# **Spark SQL** es un módulo de Apache Spark para el procesamiento de datos estructurados. Uno de sus usos es ejecutar consultas SQL, aunque también se puede utilizar para leer datos de una instalación Hive existente, devueltos en un **DataFrame Spark**.
# 
# A diferencia de **Spark API RDD** (desde la versión 1.0 en Spark), sus interfaces proporcionan información adicional (puesto que reciben información sobre la estructura de los datos), lo cual aplica a más funcionalidades.
# 
# La mayor abstracción en la API de Spark SQL es el DataFrame, el cual conserva caracerísticas de los RDD: inmutabilidad, resiliencia y computación distribuida.

# In[24]:


from pyspark.sql import SQLContext
sqlContext = SQLContext(sc) 
#SQLContext permite conectar el motor con diferentes fuentes de datos. Se utiliza para iniciar las funcionalidades de Spark SQL

spark.createDataFrame(vars_train)
vars_train = sqlContext.createDataFrame(vars_train,schema)


# In[25]:


vars_train.show(1)


# ## 3. Estimador y modelos sin mejoras

# In[26]:


# separamos el dataset en train y test (70 y 30 respectivamente)
(trainingData, testData) = vars_train.randomSplit([0.7, 0.3], seed=666)


# ### 3.1 Vector assembler

# VectorAssembler es un transformador que convierte datos de varias columnas en una **columna vectorial de una sola columna**.

# Buscamos todas las columnas del dataset que no sean la etiqueta (precio de vivienda).
# Lo realizamos mediante un select, con esto hacemos más preciso al modelo a la hora de añadir nuevas columnas

# In[27]:


import pyspark.ml.feature as ft
featureCreator = ft.VectorAssembler(
    inputCols=[col for col in vars_train.select("*").columns if col!='PrecioDeLaVivenda'], #todas menos el target 
    outputCol='features'
)


# ### 3.2 Modelos de regresión 

# Realizamos el modelo con un **Random Forest** y un **Gradient Boosting** para ver cual devuelve un menor **RMSE** (Raíz del MSE), y por lo tanto un mejor rendimiento de regresión

# * ### 3.2.1 Random Forest sin mejoras

# El Random Forest es un modelo formado por muchos **árboles de decisión**. De manera que el resultado devuelto es obtenido tras **promediar las predicciones** de los árboles.
# 
# Dichos árboles utilizan **muestreos aleatorios** del train durante el entrenamiento y están formados por **subconjuntos** aleatorios de características consideradas al dividir nodos. La clave es la **baja correlación entre los modelos** (árboles).
# 
# Los motivos por los que hemos seleccionado este modelo como el primero a implementar son los siguientes:
# * **Requiere muy pocas suposiciones**, por lo que la preparación de los datos es mucho más leve en comparación con otros algoritmos; por ejemplo: no necesita estandarización.
# * **Poco afectados por valores atípicos** al ponderar con medias o modas en la resolución.
# * Tiene un método efectivo para estimar datos faltantes; además predice **bien para grandes cantidades de datos** al utilizar varios árboles y reducir el riesgo de overfiting.
# <img src="imagenes\randomforest.png" width="700" height="1000" align="center ">

# In[28]:


from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline


# In[29]:


# modelo de random forest en este caso solo hace falta especificar el label
rf = RandomForestRegressor(labelCol='PrecioDeLaVivenda')
#el pipeline es un camino que indica los pasos que tiene que hacer el modelo en este caso
#crea una columna vector que agrupa el resto de columnas y efectúa el random forest
pipeline = Pipeline(stages=[featureCreator,rf])


# In[30]:


#guarda el modelo entrenado con los datos de training en pModel
pModel = pipeline.fit(trainingData)


# In[31]:


#guarda el test en pTest
pTest = pModel.transform(testData)


# In[32]:


#se tiene que crear un evaluador de regresión para usarlo después
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(
    #la métrica que sale de este evaluador es la raiz cuadrada del error cuadrático medio(rsme),
    #osea el error medio
    labelCol="PrecioDeLaVivenda", predictionCol="prediction", metricName="rmse")


# In[33]:


#Con este código se imprime el valor de la raiz cuadrada del error cuadrático medio
rmse = evaluator.evaluate(pTest)
print("Error cuadratico medio Random Forest (RMSE) = %g" % rmse)


# Obteniendo un **RMSE de menor de 50.000 dólares**, nos indica que, en promedio, nuestro modelo ha obtenido resultados que difieren en esa cifra de dólares del precio real. 
# 
# Hay que tener en cuenta que estas medidas provienen del promedio de las realizaciones de la prueba. Esto implica que cuando predecimos resultados sesgados (precios, ingresos, etc) lo más probable es que el **error también sea sesgado**.
# Esto puede ser debido a que en la mayoría de los casos el error es muy pequeño, pero que existen ejemplos con errores extremadamente grandes. Un error demasiado sesgado puede invalidar el resultado del promedio.
# 
# Por otro lado, cuanto **mayor es el precio de venta, menor significación** tienen 50.000 dólares de diferencia; y viceversa.
# 

# * ### 3.2.2 Gradient Boosting sin mejoras

# Gradient Boosting es una familia de algoritmos basados en la **secuencia de modelos predictivos** débiles que en nuestro caso serán los árboles de decisión. La generación de dichos árboles se crea de forma que cada uno corrija los errores del anterior. Estos **"weak lerners"** suelen ser árboles poco profundos con no más de 3 o 4 niveles de profundidad. 
# 
# La razón por la que queremos comparar su resultado con el del Random Forest es debido a que, pese a tener características ventajosas muy similares que se adaptan a las del problema, iteran disminuyendo su error con **enfoques distintos**: 
# El Gradient Boosting utiliza árboles débiles (alto sesgo, baja variación), por lo que el algoritmo se limita principalmente a **reducir el sesgo**. Por otro lado, los diversos árboles completamente desarrollados del RF (bajo sesgo, alta varianza); por lo que procuran reducir el error de la forma contraria: **reduciendo la varianza**.
# <img src="imagenes\gbm.png" width="700" height="1000" align="center ">

# In[34]:


from pyspark.ml.regression import GBTRegressor
#lo mismo pero con otro tipo de modelo
gbt = GBTRegressor(featuresCol="features",labelCol='PrecioDeLaVivenda')#, maxIter=10)
pipeline = Pipeline(stages=[featureCreator,gbt])


# In[35]:


gbtModel = pipeline.fit(trainingData)


# In[36]:


gbtTest = gbtModel.transform(testData)


# In[37]:


rmse = evaluator.evaluate(gbtTest)
print("Error cuadratico medio gradient boosting (RMSE) = %g" % rmse)


# **Random Forest** devuelve resultados ligeramente mejores en el RMSE, por lo que será el **modelo elegido para la optimización**. Además, es mucho más fácil de sintonizar que GBM. Por lo general, hay dos hiperparámetros en RF: número de árboles y número de campos que se utilizarán para entrenar cada nodo. Aunque, por otro lado, está demostrado que GBM suele funcionar mejor que RF si los parámetros se ajustan con cuidado.
# 
# La realidad es que los resultados son destacablemente similares precisamente debido a que ambos modelos trabajan sobre árboles de decisión, diferenciandose en el orden y la forma en el que estos se combinan.
# 

# ## 4. Ajuste del modelo

# ### 4.1 Discretización

# Discretizar una variable significa **convertir un grupo de valores continuos en una segmentación discreta** (mediante intervalos). Precisamente el RF puede beneficiarse de esta conversión puesto que utiliza la minimización de la **entropía de la información heurística** para seleccionar puntos de corte.
# 
# 
# Hemos seleccionado la variable "PiesCuadradosDeFachada" puesto que es una de las más correlacionadas con el precio de venta de la vivienda.
# <img src="imagenes\discretizacion.png" width="700" height="1000" align="center ">

# In[38]:


import pyspark.ml.feature as ft

discretizer = ft.QuantileDiscretizer(
    numBuckets=20, 
    inputCol='PiesCuadradosDeFachada', 
    outputCol='PiesCuadradosDeFachada_discretized')


# ### 4.2 Normalización

# Normalizar significa **tipificar las escalas de las variables en una sola escala común**, es decir: extender o comprimir valores de la variable en un rango definido; con la intención de evitar así relaciones y dependencias no deseadas entre datos. Suele ser utilizada previamente a una realización de promedios
# 
# 
# Random Forest presenta un **carácter invariante** a transformaciones de características individuales debido a que los campos **no son comparados en magnitud con otros**, sino en los rangos de una característica segmentada por el modelo. No obstante procederemos a la normalizacion del modelo siguiendo las instrucciones generales del proyecto.
# <img src="imagenes\ejemplo normalizacion.png" width="500" height="7000" align="center ">

# #### 4.2.1 Normalización 1

# En primer lugar vamos a normalizar mediante el **Standard Scaler**, dicho algoritmo elimina la media y escala la varianza de la unidad utilizando estadísticos de la muestra de entrenamiento. La "unidad estándar" se calcula con la desviación estándar de la muestra corregida.

# In[39]:


#definimos la función ya parametrizada 
normalizer_features = ft.StandardScaler(
    inputCol='features', 
    outputCol='normalized_features', 
    withMean=True, #Obtiene el valor de withMean o su valor predeterminado.
    withStd=True  #Obtiene el valor de withStd o su valor predeterminado.
)


# In[40]:


#especificamos que las features son las normalizadas
rf = RandomForestRegressor(featuresCol='normalized_features',labelCol='PrecioDeLaVivenda')
# también adaptamos el vector asembler para que no pase por el input los piescuadradosdefachada porque ha sido discretizada
featureCreator_no_dis = ft.VectorAssembler(
    inputCols=[col for col in vars_train.select("*").columns if col!='PrecioDeLaVivenda' and col!='PiesCuadradosDeFachada]'], #todas menos el target 
    outputCol='features'
)


# In[41]:


pipeline = Pipeline(stages=[discretizer,featureCreator_no_dis,normalizer_features,rf])


# In[42]:


model = pipeline.fit(trainingData)


# In[43]:


test = model.transform(testData)


# In[44]:


rmse = evaluator.evaluate(test)
print("Error cuadratico medio (RMSE) = %g" % rmse)


# El RMSE no se reduce con este tipo de normalización en Random Forest debido a que altera la escala de las variables, no la segmentación de rangos; por lo que no ofrece ninguna mejoría

# #### 4.2.2 Normalización 2

# **MinMaxScaler** modifica la escala de cada variable a un rango común (mínimo, máximo) mediante estadísticos muestrales. Es común la práctica de normalizar en un rango (0,1), que son los límites por defecto.
# 
# A priori parece que, a diferencia de StandardScaler, **sí podría variar el RMSE**; debido a que podríamos alterar las particiones de los árboles de decisión. Aunque no esperamos grandes resultados.

# In[45]:


normalizer_features_MMS=ft.MinMaxScaler(
    inputCol='features', 
    outputCol='normalized_features', 
    #withMean=True,
    #withStd=True
)


# In[46]:


pipeline = Pipeline(stages=[discretizer,featureCreator_no_dis,normalizer_features_MMS,rf])


# In[47]:


model = pipeline.fit(trainingData)


# In[48]:


test = model.transform(testData)


# In[49]:


rmse = evaluator.evaluate(test)
print("Error cuadratico medio (RMSE) = %g" % rmse)


# En este caso parece que tampoco hemos logrado reducir el error, aunque en otras iteraciones sí que nos redujo despreciablemente el RMSE.

# ### 4.3 Hypertuning

# A diferencia de los parámetros del modelo, los cuales se aprenden durante el entrenamiento, los hiperparámetros deben establecerse **antes del entrenamiento**. En el caso del Random Forest, los hiperparámetros son el **número de árboles de decisión** y el **número de características** que evaluará cada árbol. 
# 
# Cada paradigma funciona mejor con unos u otros hiperparámetros; la tarea del científico de datos es encontrar la combinación que mejor aproxime las soluciones reales del problema.
# <img src="imagenes\hipertunning.png" width="500" height="500" align="center ">

# In[50]:


import pyspark.ml.tuning as tune


# #### 4.3.1 Pipeline

# Creamos varios modelos combinando hiperparámetros a través de pipelines. En total se crearán 9 modelos 3*3.
# 
# La finalidad es encontrar la **mejor combinación de los nueve modelos probados**.

# In[51]:


pipeline = Pipeline(stages=[discretizer,featureCreator,normalizer_features,rf])


# In[52]:


grid = tune.ParamGridBuilder().addGrid(rf.numTrees, [20,25,30]).addGrid(rf.maxDepth, [5,6,9]).build()


# Hay que poner un número reducido de valores porque muchos modelos sobrecargarían el ordenador; se suelen usar 3x3 o 10x10

# #### 4.3.2 Validación Cruzada

# **CrossValidator** divide el conjunto de datos en **"pliegues"** (*folds*) o conjuntos de datos separados: **validación** y **entrenamiento**. Habiendo que establecer mediante parámetro el número de pliegues que se deseen realizar. Su finalidad es la de evaluar la **solidez del modelo** y la **independiencia de la partición** entre datos de train y test, mediante el entrenamiento de varios subconjuntos de los datos de entrada y su evaluación con otro subconjunto complementario de los datos que fue ignorado durante el entrenamiento.
# 
# 
# Por ejemplo: numFolds=3; generaríamos 3 pares datasets (train 70% y test 30%). Para evaluar calcularíamos el RMSE de los 3 pliegues, ajustando los 3 pares de conjuntos diferentes
# 
# 
# Esta técnica tiene un alto **coste computacional**; sin embargo, también es un método estadísticamente más sólido que el ajuste manual heurístico a la hora de seleccionar los hiperparámetros.
# <img src="imagenes\validación cruzada.jpg" width="700" height="500" align="center ">

# In[53]:


cv = tune.CrossValidator(estimator=pipeline,estimatorParamMaps=grid,evaluator=evaluator,numFolds=3)


# In[54]:


cvModel = cv.fit(trainingData)


# In[55]:


results = cvModel.transform(testData)


# In[56]:


#comparación de errores
rmse = evaluator.evaluate(test)
print("Error cuadratico medio (RMSE) sin hypertuning en los datos test = %g" % rmse)
rmse = evaluator.evaluate(results)
print("Error cuadratico medio (RMSE) con hypertuning en los datos test = %g" % rmse)


# El RMSE del modelo se **reduce aproximadamente en unos 1.000 dólares**, lo cual es una mejora considerable de la predicción del modelo. 
# 
# Esto se debe a que se ha garantizado la independiencia de la partición entre datos de entrenamiento y prueba. Y a que se han optimizado los hiperparámetros del modelo, seleccionando los que se ajustan la predicción.

# In[57]:


bestModel = cvModel.bestModel


# In[58]:


#código que me imprime que hyperparámetros son los mejores para el modelo
print('El mejor número de árboles es: {}'.format(bestModel.stages[-1]._java_obj.getNumTrees()))
print('El mejor valor de profundidad es: {}'.format(bestModel.stages[-1]._java_obj.getMaxDepth()))


# El mejor modelo devuelto por el grid consta de los siguientes hiperparámetros:
# * Número de árboles: 30
# * Profundidad de árbol: 5

# #### 4.3.3 Selección de características

# La selección de características indentifica las **características más influyentes** durante el entrenamiento del modelo. Reduciendo el tamaño del espacio de funciones podemos mejorar el rendimiento del aprendizaje estadístico, además de su velocidad.
# <img src="imagenes\seleccion.jpg" width="400" height="300" align="center ">

# #### 4.3.3.1 Selección de características sin hypertuning

# In[59]:


from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

#me selecciona que variables son las más relevantes para el modelo
selector = ChiSqSelector(numTopFeatures=16, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="PrecioDeLaVivenda")


# In[60]:


#actualizo el normalizer porque ahora el input es diferente y quiero que funcione bien
normalizer_features_selected_features = ft.StandardScaler(
    inputCol='selectedFeatures', 
    outputCol='normalized_features', 
    withMean=True,
    withStd=True
)


# In[61]:


pipeline = Pipeline(stages=[discretizer,featureCreator,selector,normalizer_features_selected_features,rf])


# In[62]:


sModel = pipeline.fit(trainingData)


# In[63]:


sTest = sModel.transform(testData)


# In[64]:


#comparativa de errores
rmse = evaluator.evaluate(test)
print("Error cuadratico medio (RMSE) sin reducir variables = %g" % rmse)
rmse = evaluator.evaluate(sTest)
print("Error cuadratico medio (RMSE) reduciendo variables = %g" % rmse)


# Al utilizar solo las variables más importantes hemos quitado ruido al modelo y reducido el RMSE

# #### 4.3.3.2 Selección de características con hypertuning

# Como la selección de características y el hypertuning fueron los añadidos que más redujeron el RMSE, decidimos como punto extra juntarlos para conseguir el mejor modelo posible

# In[65]:


grid = tune.ParamGridBuilder().addGrid(rf.numTrees, [20,25,30]).addGrid(rf.maxDepth, [5,6,9]).build()


# In[66]:


cv = tune.CrossValidator(estimator=pipeline,estimatorParamMaps=grid,evaluator=evaluator,numFolds=3)


# In[67]:


cvModel = cv.fit(trainingData)


# In[68]:


results = cvModel.transform(testData)


# In[69]:


#comparativa de errores
rmse = evaluator.evaluate(test)
print("Error cuadratico medio (RMSE) sin reducir variables = %g" % rmse)
rmse = evaluator.evaluate(results)
print("Error cuadratico medio (RMSE) reduciendo variables y con hypertuning (mejor versión) = %g" % rmse)


# In[70]:


bestModel = cvModel.bestModel


# In[71]:


#código que me imprime que hyperparámetros son los mejores para el modelo
print('El mejor número de árboles es: {}'.format(bestModel.stages[-1]._java_obj.getNumTrees()))
print('El mejor valor de profundidad es: {}'.format(bestModel.stages[-1]._java_obj.getMaxDepth()))


# El RMSE del modelo se reduce a valores cercanos a **45.000 dólares**, que es el mejor resultado que hemos obtenido mediante el ajuste del modelo
# 
# Obteniendo este resultado, hemos combinado la selección de características actualizando los hiperparámetros adecuados para el problema. Es intuitivo deducir esta mejoría debido a que el número de árboles y sus complejidades no **deberían rendir igual para 16 variables que para 32** durante el entrenamiento.
# 
# Los nuevos hiperparámetros de este modelo son:
# * Número de árboles: 25
# * Profundidad de árbol: 6

# #### 4.3.4  Redución de dimensionalidad

# La reducción de dimensionalidad trata de **reducir el número de variables** consideradas en el modelo. Su funcionalidad radica en:
# * **Eliminación de la característica**: eliminación de variables redundantes o que no proporcionan suficiente información.
# * **Extracción de variables**: formar nuevas variables a partir de las antiguas
# 
# El análisis de componentes principales (**PCA**) consiste precisamente en la extracción de variables.
# <img src="imagenes\reduccion dimensionalidad.png" width="600" height="300" align="center ">

# In[72]:


from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors


# In[73]:


rf = RandomForestRegressor(featuresCol='pca_features',labelCol='PrecioDeLaVivenda',)
#código para la reducción de dimensionalidad
pca = PCA(k=9, inputCol="features", outputCol="pca_features")
pipeline = Pipeline(stages=[featureCreator,pca,rf])
modelPCA = pipeline.fit(trainingData)
testPCA=modelPCA.transform(testData)#.collect()[0].pca_features


# In[74]:


rmse = evaluator.evaluate(test)
print("Error cuadratico medio (RMSE) sin  pca = %g" % rmse)
rmse = evaluator.evaluate(testPCA)
print("Error cuadratico medio (RMSE) con pca = %g" % rmse)


# El error ha **aumentado ligeramente**, probablemente sea por comprimir demasiado las variables al reducir la dimensionalidad a 9.
