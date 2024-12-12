# Practica 1

**Autores** Ana Zabalo, Ana Salmerón 

**Correo electrónico:** ana.zabalo@cuenf.edu , ana.salmeron@cunef.edu

**Directorio GitHub:** https://github.com/ana282001/practica1
                        https://github.com/anazb001/practica2.git

## Notebooks Desarrollados

A lo largo de este EDA, hemos llevado a cabo distintos notebooks para el analisis

- Notebook 01_EDA_Exploracion_General : 

En el primer notebook, hemos realizado el análisis exploratorio inicial del dataset con el propósito de comprender su estructura general, los tipos de variables que contiene, y su distribución. En esta etapa, identificamos valores nulos y outliers, separando las variables en continuas y categóricas para facilitar su análisis. También analizamos la variable objetivo (TARGET), destacando su distribución desbalanceada, y generamos visualizaciones que mostraron relaciones relevantes entre las variables independientes y la variable objetivo. Este análisis permitió identificar patrones iniciales y definir estrategias preliminares para el tratamiento de los datos en etapas posteriores.

- Notebook 02_Tratamiento_correlaciones_missing_outliers

En el segundo notebook, nos centramos en limpiar y procesar los datos para garantizar su calidad. Abordamos los valores faltantes, identificando las columnas con porcentajes altos de nulos y aplicando estrategias de imputación o eliminación según su relevancia. También analizamos las correlaciones entre las variables numéricas, detectando redundancias que podrían simplificarse para reducir la dimensionalidad del dataset. Además, tratamos el problema de los outliers, decidiendo su manejo en función de su impacto en el análisis. El resultado de esta fase fue un dataset más limpio y organizado, preparado para ser transformado y escalado en la etapa siguiente.

- Notebook 03_Encoding_scaling_conclusion

Finalmente, en el tercer notebook, el objetivo fue preparar el dataset para su uso en modelos predictivos. Esto incluyó la codificación de variables categóricas en formatos numéricos mediante técnicas como Ordinal Encoding y Target Encoding, así como el escalado de las variables continuas para garantizar una escala uniforme entre ellas. Posteriormente, dividimos el dataset en conjuntos de train y test de forma estratificada, asegurando la representatividad de la variable objetivo en ambos subconjuntos. El trabajo realizado en este cuaderno concluyó con la generación de un dataset preprocesado y transformado, listo para la implementación de modelos predictivos en las etapas siguientes.

- Notebook 03_Feature_Procesing : 

En este notebook, hemos optimizado el modelo mediante selección de variables y reducción de dimensionalidad. Descartamos el modelo Random Forest por su tendencia a asignar demasiada importancia a variables aleatorias, y elegimos Boruta, que seleccionó 16 variables clave. A través de regresiones logísticas (Ridge, Lasso, ElasticNet), consolidamos las variables más relevantes. Aunque la reducción de dimensionalidad con PCA mostró patrones, no mejoró el rendimiento del modelo. En resumen, el enfoque más eficaz fue la combinación de Boruta y las regresiones logísticas para seleccionar variables predictivas.

- Notebook 04_Model_Selection

En este notebook, realizamos la selección y evaluación de varios modelos de clasificación. Comenzamos con un modelo base y luego probamos diferentes modelos, como regresión logística, árboles de decisión y modelos de boosting (como LGBMClassifier(), XGBClassifier() y GradientBoostingClassifier()). Evaluamos su rendimiento en términos de precisión y eficiencia, y optimizamos los mejores modelos mediante la búsqueda de hiperparámetros. Los resultados nos permitieron identificar los modelos más efectivos, equilibrando precisión y eficiencia computacional para el análisis futuro.

- Notebook 05_Model_Implementation

En este notebook, implementamos los dos mejores modelos con los parámetros optimizados y tomamos una decisión final.

- Notebook 06_Explainability

Utilizamos la librería SHAP para estudiar la explicabilidad del modelo