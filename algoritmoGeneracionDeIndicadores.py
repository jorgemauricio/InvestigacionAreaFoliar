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
	# leer archivo de analisis foliar
	dataFoliar = pd.read_csv('data/Foliares_Hass.csv')

	# estructura de archivo csv de resultados
	textToData = "Id,N,P,K,Ca,Mg,S,Fe,Cu,Mn,Zn,B,Mo,Na,Cl,pVerde,pCafeRojo,pCafe,pAmarillo,pAmarilloDorado,pCafeAmarillo\n"
	
	# obtener los archivos de imagen a clasificar
	# generar una lista con todas las imagenes
	fileList = [x for x in os.listdir('images/train') if x.endswith('.jpg')]
	
	# ciclo for para evaluar cada imagen
	for file in fileList:
		
		# declarar dict
		dictColores = {}

		# determinar los principales colores en las imágenes
		# print nombre de la imagen
		print("***** Procesando: {}".format(file))

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
		data = data.sort_values(by='frecuencia', ascending = False)

		# nombre archivo de colores para cada imagen
		nombreTemporalArchivoColores = 'resultados/totalDeColores_{}.csv'.format(file.split('.')[0])

		# save to csv
		data.to_csv(nombreTemporalArchivoColores, index=False)

		# crear columna L
		data['L'] = data.apply(lambda x: generarValorL(x['color']), axis=1)

		# crear columna L
		data['a'] = data.apply(lambda x: generarValorA(x['color']), axis=1)

		# crear columna L
		data['b'] = data.apply(lambda x: generarValorB(x['color']), axis=1)

		# eliminar colores grises
		# data = data.loc[(data['a'] <= -5) | (data['a'] >= 5) | (data['b'] <= -5) | (data['b'] >= 5)] 

		# clasificacion de hoja de acuerdo al porcentaje de color verde
		data['clasificacionColor'] = data.apply(lambda x: clasificacionDecolor(x['L'],x['a'],x['b']), axis=1)

		# eliminar colores de fondo
		data = data.loc[data['clasificacionColor'] != "f"]

		# sumatoria de la frecuencia
		sumatoriaFrecuencia = data['frecuencia'].sum()

		# generar columna de porcentaje
		data['porcentaje'] = data['frecuencia'] / sumatoriaFrecuencia * 100

		# sumatoria de porcentajes
		data['sumatoriaPorcentaje'] = data['porcentaje'].cumsum()

		# tomar solo en cuenta el 80-20 de los datos
		# data = data.loc[data['sumatoriaPorcentaje'] <= 80]

		# nombre archivo de colores clasificados para cada imagen
		nombreTemporalArchivoColoresClasificados = 'resultados/totalDecolores_clasificados_{}.csv'.format(file.split('.')[0])

		# guardar como csv
		data.to_csv(nombreTemporalArchivoColoresClasificados, index=False)

		# numero de registros por clasificacion
		pVerde = len(np.array(data.loc[data['clasificacionColor'] == 'v']))
		pCafeRojo = len(np.array(data.loc[data['clasificacionColor'] == 'cr']))
		pCafe = len(np.array(data.loc[data['clasificacionColor'] == 'c']))
		pAmarillo = len(np.array(data.loc[data['clasificacionColor'] == 'a']))
		pAmarilloDorado = len(np.array(data.loc[data['clasificacionColor'] == 'ag']))
		pCafeAmarillo = len(np.array(data.loc[data['clasificacionColor'] == 'ca']))

		print(pVerde, pCafeRojo, pCafe, pAmarillo, pAmarilloDorado, pCafeAmarillo)

		# numero total de registros
		numeroTotalDeRegistros = pVerde + pCafeRojo + pCafe + pAmarillo + pAmarilloDorado + pCafeAmarillo

		# numero de registros por clasificacion
		pVerde = pVerde / numeroTotalDeRegistros * 100
		pCafeRojo = pCafeRojo / numeroTotalDeRegistros * 100
		pCafe = pCafe / numeroTotalDeRegistros * 100
		pAmarillo = pAmarillo / numeroTotalDeRegistros * 100
		pAmarilloDorado = pAmarilloDorado / numeroTotalDeRegistros * 100
		pCafeAmarillo = pCafeAmarillo / numeroTotalDeRegistros * 100

		# agregar record al texto
		dataTemporalFoliar = dataFoliar.loc[dataFoliar['Id'] == int(file.split('.')[0])]
		N = np.array(dataTemporalFoliar['N'])
		P = np.array(dataTemporalFoliar['P'])
		K = np.array(dataTemporalFoliar['K'])
		Ca = np.array(dataTemporalFoliar['Ca'])
		Mg = np.array(dataTemporalFoliar['Mg'])
		S = np.array(dataTemporalFoliar['S'])
		Fe = np.array(dataTemporalFoliar['Fe'])
		Cu = np.array(dataTemporalFoliar['Cu'])
		Mn = np.array(dataTemporalFoliar['Mn'])
		Zn = np.array(dataTemporalFoliar['Zn'])
		B = np.array(dataTemporalFoliar['B'])
		Mo = np.array(dataTemporalFoliar['Mo'])
		Na = np.array(dataTemporalFoliar['Na'])
		Cl = np.array(dataTemporalFoliar['Cl'])

		print('***** N: {}'.format(N[0]))
		print('***** pVerde: {}'.format(pVerde))

		textToData += "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f}\n".format(file.split('.')[0],N[0],P[0],K[0],Ca[0],Mg[0],S[0],Fe[0],Cu[0],Mn[0],Zn[0],B[0],Mo[0],Na[0],Cl[0],pVerde,pCafeRojo,pCafe,pAmarillo,pAmarilloDorado,pCafeAmarillo)

		# guardar archivo con los resultados
		nombreTemporalDeArchivoFinal = 'data/Foliar_join_porcentajes.csv'
		archivoCompiladoFinal = open(nombreTemporalDeArchivoFinal, "w")
		archivoCompiladoFinal.write(textToData)
		archivoCompiladoFinal.close()

# funcion para determinar el porcentaje de color verde
def clasificacionDecolor(L,a,b):
	"""
	Determina la clasificacion del color mediante los espectros de color Lab
	param: L: valor L
	param: a: valor a
	param: b: valor b
	regresa v: verde, r: rojo, c: cafe, a: amarillo, n: naranja, az: azul, f: fondo
	"""

	if L >= 2 and L <= 73 and a >= -64 and a <= -2 and b >= 3 and b <= 72:
		return "v"
	elif L >= 74 and L <= 99 and a >= -66 and a <= -4 and b >= 5 and b <= 95:
		return "a"
	elif L >= 41 and L <= 94 and a >= -18 and a <= -10 and b >= 48 and b <= 80:
		return "ag"
	elif L >= 3 and L <= 67 and a >= 2 and a <= 42 and b >= 4 and b <= 75:
		return "c"
	elif L >= 10 and L <= 60 and a >= -14 and a <=-5 and b >= 15 and b <= 64:
		return "ca"
	elif L >= 2 and L <= 19 and a >= 11 and a <= 40 and b >= 4 and b <= 29:
		return "cr"
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