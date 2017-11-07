#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:17:25 2017

@author: jorgemauricio
"""

#librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

#%% estilo de la grafica
plt.style.use('ggplot')

# - - - - - MAIN - - - - - 
def main():
	# función para generar el dict de colores (comentar si ya se tiene generado el diccionario)
	# generarDictColores()

	# leer csv totalDecolores
	data = pd.read_csv('resultados/totalDeColores.csv')

	# crear columna L
	data['L'] = data.apply(lambda x: generarValorL(x['color']), axis=1)

	# crear columna L
	data['a'] = data.apply(lambda x: generarValorA(x['color']), axis=1)

	# crear columna L
	data['b'] = data.apply(lambda x: generarValorB(x['color']), axis=1)

	# eliminar colores grises
	data = data.loc[(data['a'] <= -5) | (data['a'] >= 5) & (data['b'] <= -5) | (data['b'] >= 5)]

	# sumatoria de la frecuencia
	sumatoriaFrecuencia = data['frecuencia'].sum()

	# generar columna de porcentaje
	data['porcentaje'] = data['frecuencia'] / sumatoriaFrecuencia * 100

	# ordenar valores
	data = data.sort_values(by='porcentaje', ascending=False)

	# sumatoria de porcentajes
	data['sumatoriaPorcentaje'] = data['porcentaje'].cumsum()

	# tomar solo en cuenta el 80-20 de los datos
	data = data.loc[data['sumatoriaPorcentaje'] <= 80]

	# clasificacion de hoja de acuerdo al porcentaje de color verde
	data['clasificacionColor'] = data.apply(lambda x: clasificacionDecolor(x['L'],x['a'],x['b']), axis=1)

	# eliminar colores de fondo
	data = data.loc[data['clasificacionColor'] != 'f']

	# guardar csv
	data.to_csv('resultados/colores_clasificados.csv')

	
# funcion para determinar el porcentaje de color verde
def clasificacionDecolor(L,a,b):
	"""
	Determina la clasificacion del color mediante los espectros de color Lab
	param: L: valor L
	param: a: valor a
	param: b: valor b
	regresa v: verde, r: rojo, c: cafe, a: amarillo, n: naranja, az: azul, f: fondo
	"""

	if L >= 0 and L <= 88 and a >= -86 and a <= -20 and b >= 3 and b <= 128:
		return "v"
	elif L >= 9 and L <= 56 and a >= 30 and a <= 71 and b >= 15 and b <= 51:
		return "r"
	elif L >= 44 and L <= 64 and a >= 12 and a <=26 and b >= 53 and b <= 55:
		return "c"
	elif L >= 66 and L <= 98 and a >= -14 and a <= 14 and b >= 50 and b <= 70:
		return "a"
	elif L >= 41 and L <= 67 and a >= 44 and a <= 47 and b >= 52 and b <= 61:
		return "n"
	elif L >= 2 and L <= 34 and a >= -128 and a <= -78 and b >= -128 and b <= -29:
		return "az"
	else:
		return "f"

# función para generar la columna L
def generarValorL(valor):
	"""
	Genera el valor de L del string de color
	param: valor:  string de color
	"""
	L, a, b = valor.split("/")
	return float(L)

# función para generar la columna L
def generarValorA(valor):
	"""
	Genera el valor de L del string de color
	param: valor:  string de color
	"""
	L, a, b = valor.split("/")
	return float(a)

# función para generar la columna L
def generarValorB(valor):
	"""
	Genera el valor de L del string de color
	param: valor:  string de color
	"""
	L, a, b = valor.split("/")
	return float(b)

# función para generar el dict de colores
def generarDictColores():
	"""
	Genera un diccionario de colores de todas las imagenes que se encuentren en la carpeta images/train
	"""
	# generar una lista con todas las imagenes
	fileList = [x for x in os.listdir('images/train') if x.endswith('.jpg')]

	# declarar dict
	dictColores = {}

	# determinar los principales colores en las imágenes
	for file in fileList:

		# rpint
		print(file)

		# nombre temporal del archivo a evaluar
		nombreTemporalArchivo = "images/train/{}".format(file)

		# cargar la imagen
		im = Image.open(nombreTemporalArchivo)
		pix = im.load()

		# tamaño de la imagen
		x, y = im.size

		# ciclo for para contabilizar los colores
		for i in range(x):
			for j in range(y):
				vR, vG, vB = pix[i, j]
				valueL, valueA, valueB = convertirRGBtoLAB(vR, vG, vB)
				statusColor = validarArea(valueL, valueA, valueB)
				if statusColor:
					nombreTemporalClave = "{}/{}/{}".format(valueL, valueA, valueB)
					if nombreTemporalClave in dictColores:
						dictColores[nombreTemporalClave] += 1
					else:
						dictColores[nombreTemporalClave] = 1

	# dict to DataFrame
	data = pd.DataFrame()
	data['color'] = dictColores.keys()
	data['frecuencia'] = dictColores.values()

	# ordenar información
	data = data.sort_values(by='frecuencia')

	# save to csv
	data.to_csv('resultados/totalDeColores.csv')

# Function RGB to Lab
def convertirRGBtoLAB(vr, vg, vb):
    """
    Convertir colores del espectro RGB a Lab
    param: vr: valor espectro r
    param: vg: valor espectro g
    param: vb: valor espectro b
    """
    r = (vr + 0.0) / 255
    g = (vg + 0.0) / 255
    b = (vb + 0.0) / 255

    if (r > 0.04045):
        r = pow((r + 0.055) / 1.055, 2.4)
    else:
        r = r / 12.92
    if (g > 0.04045):
        g = pow((g + 0.055) / 1.055, 2.4)
    else:
        g = g / 12.92
    if (b > 0.04045):
        b = pow((b + 0.055) / 1.055, 2.4)
    else:
        b = b / 12.92

    r = r * 100.0
    g = g * 100.0
    b = b * 100.0

    var_x = r * 0.4124 + g * 0.3576 + b * 0.1805
    var_y = r * 0.2126 + g * 0.7152 + b * 0.0722
    var_z = r * 0.0193 + g * 0.1192 + b * 0.9505

    var_x = var_x / 95.047
    var_y = var_y / 100.00
    var_z = var_z / 108.883

    if (var_x > 0.008856):
        var_x = pow(var_x, (1.0 / 3.0))
    else:
        var_x = (7.787 * var_x) + (16.0 / 116.0)
    if (var_y > 0.008856):
        var_y = pow(var_y, (1.0 / 3.0))
    else:
        var_y = (7.787 * var_y) + (16.0 / 116.0)
    if (var_z > 0.008856):
        var_z = pow(var_z, (1.0 / 3.0))
    else:
        var_z = (7.787 * var_z) + (16.0 / 116.0)

    var_L = (116.0 * var_y) - 16.0
    var_a = 500.0 * (var_x - var_y)
    var_b = 200.0 * (var_y - var_z)
    if (var_L >= 0 and var_L <= 100 and var_a == 0 and var_b == 0):
    	return 0.0, 0.0, 0.0
    else:
    	return var_L, var_a, var_b 

def validarArea(vL, vA, vB):
    """
    Eliminar puntos grises y fondo
    param: vL: valor espectro L
    param: vA: valor espectro a
    param: vB: valor espectro b
    """
    # validate grayscale and mark points
    if vL >= 0 and vL <= 100 and vA > -5 and vA < 5 and vB > -5 and vB < 5:
        return False
    else:
        return True

if __name__ == "__main__":
	main()