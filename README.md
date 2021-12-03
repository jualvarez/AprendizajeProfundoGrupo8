# AprendizajeProfundoGrupo8
Práctico aprendizaje profundo DiploDatos 2021 FaMAF

Integrantes:
- Alvarez, Juan
- Carrion, Nicolas
- Delgado, Gabriel
- Heredia, Gonzalo
- Ramos, Matias 

# Introduccion

En el repositorio se encuentra la resolucion del trabajo practico en el cual se presento
 como desafio, el dataset del meli challengue.
Este dataset contiene informacion sobre el titulo de productos que se encuentran a la venta en el marketplace de mercado libre
y la categoria a la que pertenecen (hay 632 posibles).
Cabe aclarar que nosotros a la hora de la experimentacion, utilizamos las 100 categorias mas populares y un conjunto reducido del dataset en español de mas de 1 millon de datos.
Nuestro aproach fue enfocarse en crear una estructura general, que posteriormente pudiera ser utilizada con la cantidad de datos y parametros que se necesite. 

# Preprocesamiento 


# Modelos 

Se utilizaron 2 arquitecturas a la hora de implementar los modelos. Primero se realizaron experimentos con un ****MLP*** con el fin de empezar a introducirnos en algunos conceptos basicos. Este modelo tenia varios parametros por defecto y nos dio resultados bastantes buenos.
Posteriormente y para cumplir con el enunciado del practico, se utilizo un ***CNN*** primero con parametos por defecto y luego fuimos experimentando con otros valores. En el codigo se dejo una version en la cual uno puedo setear los parametros con los cuales se puede probar ****checkear esto****.

# Evaluacion

Todos los resultados que fuimos obteniendo los almacenamos en ML-Flow para ir teniendo un seguimiento de los mismos. Algunas capturas de nuestro repo de ML-Flow:
- Este es el grafico del accuracy de nuestro conjunto de test con el mejor modelo
![WhatsApp Image 2021-11-30 at 23 09 33](https://user-images.githubusercontent.com/38257398/144618532-e9f5088d-e2d1-44d9-a50c-89e7d328a88f.jpeg)
- Este es el loss del conjunto de test
![WhatsApp Image 2021-11-30 at 23 10 15](https://user-images.githubusercontent.com/38257398/144618550-45f0ecd4-937d-42ee-af1d-5548622c2e98.jpeg)
Tambien almacenamos el mejor modelo obtenido, en un archivo de drive ****poner enlace**** para hacer pruebas con datos fuera de la muestra. Se creo un script para ir haciendo pruebas de que tan bien estaba funcionando el modelo. Eso se puede ejecutar desde el archivo cnn.py pasando el argumento --eval=True
![WhatsApp Image 2021-11-30 at 23 11 11](https://user-images.githubusercontent.com/38257398/144619089-b0aa4956-0225-466d-9915-e75247946b2b.jpeg)



