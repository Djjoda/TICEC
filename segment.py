# Importamos las librerías necesarias. En este caso, las dos más importantes son Numpy y OpenCV.
import argparse

import cv2
import numpy as np
import math


# Los argumentos de entrada definen la ruta a la imagen que será segmentada, así como el número de *clusters* o grupos
# a hallar mediante la aplicación de K-Means.
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', required=True, type=str, help='Ruta a la imagen a segmentar.')
argument_parser.add_argument('-k', '--num-clusters', default=3, type=int,
                             help='Número de clusters para K-Means (por defecto = 3).')
arguments = vars(argument_parser.parse_args())

# Cargamos la imagen de entrada.
image = cv2.imread(arguments['image'])

# Creamos una copia para poderla manipular a nuestro antojo.
image_copy = np.copy(image)

# Mostramos la imagen y esperamos que el usuario presione cualquier tecla para continuar.
#cv2.imshow('Imagen', image)
#cv2.waitKey(0)

# Convertiremos la imagen en un arreglo de ternas, las cuales representan el valor de cada pixel. En pocas palabras,
# estamos aplanando la imagen, volviéndola un vector de puntos en un espacio 3D.
pixel_values = image_copy.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Abajo estamos aplicando K-Means. Como siempre, OpenCV es un poco complicado en su sintaxis, así que vamos por partes.

# Definimos el criterio de terminación del algoritmo. En este caso, terminaremos cuando la última actualización de los
# centroides sea menor a *epsilon* (cv2.TERM_CRITERIA_EPS), donde epsilon es 1.0 (último elemento de la tupla), o bien
# cuando se hayan completado 10 iteraciones (segundo elemento de la tupla, criterio cv2.TERM_CRITERIA_MAX_ITER).
stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Este es el número de veces que se correrá K-Means con diferentes inicializaciones. La función retornará los mejores
# resultados.
number_of_attempts = 10

# Esta es la estrategia para inicializar los centroides. En este caso, optamos por inicialización aleatoria.
centroid_initialization_strategy = cv2.KMEANS_RANDOM_CENTERS

# Ejecutamos K-Means con los siguientes parámetros:
# - El arreglo de pixeles.
# - K o el número de clusters a hallar.
# - None indicando que no pasaremos un arreglo opcional de las mejores etiquetas.
# - Condición de parada.
# - Número de ejecuciones.
# - Estrategia de inicialización.
#
# El algoritmo retorna las siguientes salidas:
# - Un arreglo con la distancia de cada punto a su centroide. Aquí lo ignoramos.
# - Arreglo de etiquetas.
# - Arreglo de centroides.
_, labels, centers = cv2.kmeans(pixel_values,
                                arguments['num_clusters'],
                                None,
                                stop_criteria,
                                number_of_attempts,
                                centroid_initialization_strategy)



# Aplicamos las etiquetas a los centroides para segmentar los pixeles en su grupo correspondiente.
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# Debemos reestructurar el arreglo de datos segmentados con las dimensiones de la imagen original.
segmented_image = segmented_data.reshape(image_copy.shape)
fitrado = np.copy(segmented_image)
x,y,z = segmented_image.shape

#   0      1        2       3        4         5       6        7
# azul - verde -   rojo -  lila - amarillo - blanco - cian - tomate
colors = [
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [255,0,255],
    [0,255,255],
    [255,255,255],
    [255,255,0],
    [0,143,255]]


for i in range(x):
    for j in range(y):
        b,g,r = segmented_image[i,j]
        for m in range(len(centers)):
            #m=7
            if b == centers[m,0] and g == centers[m,1] and r == centers[m,2]:
                segmented_image[i,j] = colors[m]
                    

# Mostramos la imagen segmentada resultante.
cv2.imshow('Imagen original', image)
cv2.imshow('Imagen segmentada', segmented_image)
cv2.imwrite('segmentada.png',segmented_image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

l_colors = [0,0,0,0,0,0,0,0]

l_colors[0] = int(input("AZUL: "))
l_colors[1] = int(input("VERDE: "))
l_colors[2] = int(input("ROJO: "))
l_colors[3] = int(input("LILA: "))
l_colors[4] = int(input("AMARILLO: "))
l_colors[5] = int(input("BLANCO: "))
l_colors[6] = int(input("CIAN: "))
l_colors[7] = int(input("TOMATE: "))
print(f"Filtrando {l_colors}...")

for l in range(len(centers)):
    if l_colors[l] == 0:
        colors[l] = [0,0,0]
    else:
        colors[l] = [255,255,255]

for a in range(x):
    for b in range(y):
        b1,g1,r1 = fitrado[a,b]
        for n in range(len(centers)):
            if b1 == centers[n,0] and g1 == centers[n,1] and r1 == centers[n,2]:
                fitrado[a,b] = colors[n]  

cv2.imshow('Imagen 2', image)
cv2.imshow('Imagen 3', fitrado)
cv2.imwrite('filtrado.png',fitrado)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

#Detección de bordes

gray = cv2.cvtColor(fitrado,cv2.COLOR_BGR2GRAY)
_,th = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
cont,hierarchy1 = cv2.findContours(th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

areas = np.zeros(len(cont))
varianza = 0
for c in range(len(cont)):
    areas[c] = cv2.contourArea(cont[c])
    varianza = varianza + pow(areas[c],2)
    
#print(sum(areas)/len(cont),min(areas),max(areas),varianza/len(areas),math.sqrt(varianza/len(areas)))
umbral = math.sqrt(varianza/len(areas))
media = sum(areas)/len(cont)
for c2 in cont:
    if cv2.contourArea(c2) >= media:        
        cv2.drawContours(fitrado, c2, -1, (255,255,0), 3)


cv2.imshow('imagen',fitrado)
cv2.imwrite('contorno.png',fitrado)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
