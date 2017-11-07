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
	generarDictColores()
	

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