#from google.colab.patches import cv2_imshow
# Escuela Superior de Cómputo 
# Instituto Politécnico Nacional 

# Visión artificial
# Examen Práctico 2

# Luis Gerardo Ortiz Cruz 
# 5BM1 

# python examenPractico2.py 

# Bibliotecas 
import cv2 # Para leer y guardar imagenes
import numpy as np # Para manipular los arreglos
import math # Para las funciones matematicas
import random # Para los numeros pseudoaleatorios
import matplotlib.pyplot as plt # Para graficar las opercaiones de vectores

# Inicializa la semilla de los numeros pseudoaleatorios
random.seed(0)

# Obtiene los datos de la imagen
def leeImagen(nombreArchivo):
  # Lee la imagen
  imagen = cv2.imread(nombreArchivo, cv2.IMREAD_UNCHANGED)
  # Convierte la imagen
  imagen = imagen.tolist()
  return imagen

# Muestra la imagen y la guarda en un archivo PNG
def muestraYGuardaImagen(imagen, nombre):
  if type(imagen) == type([]): # Si la imagen es una lista
    # Convierte la imagen al tipo de opencv
    imagen = np.array(imagen, dtype=np.uint8)
  # Muestra los datos de la imagen
  filas = len(imagen)
  columnas = len(imagen[0])
  print()
  print("Imagen:", nombre)
  print("Filas:", filas)
  print("Columnas:", columnas)
  # Muestra la imagen
  #cv2_imshow(imagen)
  cv2.imshow(nombre, imagen)
  # Guarda la imagen
  cv2.imwrite(f"{nombre}.png", imagen)

# Obtiene los nuevos centroides de un conjunto de grupos
def obtenNuevosCentroidesColor(sumasComponentes, tamannosGrupos):
  # Obtiene el tamaño de los componentes y el numero de centroides
  totalComponentes = len(sumasComponentes[0])
  totalGrupos = len(tamannosGrupos)
  # Inicializa los nuevos centroides
  centroides = [[0] * totalComponentes for i in range(totalGrupos)]
  for centroide, sumasGrupo, tamannoGrupo in zip(centroides, sumasComponentes, tamannosGrupos): # Para cada centroide (grupo)
    for j, suma_j in enumerate(sumasGrupo): # Para cada componente
      if tamannoGrupo == 0: # Si el numero de elementos es 0
        # Evita la division entre 0
        tamannoGrupo = 1
      # Obtiene el nuevo valor de la componente del centroide
      centroide[j] = suma_j / tamannoGrupo
  return centroides

# Obtiene el nuevo grupo de un vector, con base en su distancia a los centroides de los grupos
def obtenNuevoGrupoColor(vector, centroides):
  # Inicializa la comparacion con menor distancia
  DISTANCIA_MAXIMA = 256 ** 2 * len(vector)
  menorComparacion = (DISTANCIA_MAXIMA, -1)
  for i, centroide in enumerate(centroides): # Para cada centroide
    # Obtiene la diastancia
    distancia = 0
    for vector_i, centroide_i in zip(vector, centroide): # Para cada componente
      distancia += (centroide_i - vector_i) ** 2
    distancia = math.sqrt(distancia)
    # Guarda la mejor comparacion
    menorComparacion = min(menorComparacion, (distancia, i))
  return menorComparacion[1]

def segmentaColorKMedias(imagen, totalGrupos):
  # Obtiene las dimensiones de la imagen
  filasImagen = len(imagen)
  columnasImagen = len(imagen[0])
  # Asignar el grupo inicial al azar
  gruposPixeles = [[random.randint(0, totalGrupos - 1)] * columnasImagen for i in range(filasImagen)]
  sumasComponentes = [[0, 0, 0] for i in range(totalGrupos)]
  tamannosGrupos = [0] * totalGrupos
  for filaImagen, filaGrupos in zip(imagen, gruposPixeles): # Para cada fila
    for pixel, grupo_pixel in zip(filaImagen, filaGrupos): # Para cada pixel
      for k, pixel_k in zip(range(3), pixel): # Para cada componente de color
        # Suma el valor de la componente al grupo correspondientes
        sumasComponentes[grupo_pixel][k] += pixel_k
      tamannosGrupos[grupo_pixel] += 1
  # Realiza la reasignacion de grupos
  hayReasignacion = True
  while hayReasignacion: # Mientras exista una reasignacion
    # Calcula los nuevos centroides
    centroides = obtenNuevosCentroidesColor(sumasComponentes, tamannosGrupos)
    # Revisa si existe reasignacion
    hayReasignacion = False
    for filaImagen, filaGrupos in zip(imagen, gruposPixeles): # Para cada fila
      for j, (pixel, grupo_pixel) in enumerate(zip(filaImagen, filaGrupos)): # Para cada pixel
        nuevoGrupo = obtenNuevoGrupoColor(pixel, centroides)
        if nuevoGrupo != grupo_pixel: # Si el nuevo grupo es diferente
          # Asigna el nuevo grupo
          filaGrupos[j] = nuevoGrupo
          hayReasignacion = True
          for k, pixel_k in zip(range(3), pixel): # Para cada componente de color
            # Actualiza las sumas
            sumasComponentes[grupo_pixel][k] -= pixel_k
            sumasComponentes[nuevoGrupo][k] += pixel_k
          # Actualiza los tamaños de los grupos
          tamannosGrupos[grupo_pixel] -= 1
          tamannosGrupos[nuevoGrupo] += 1 
  return gruposPixeles, centroides

def asignaValorGruposImagenSegmentada(gruposPixeles, centroides):
  # Obtiene las dimensiones de la imagen
  filasImagen = len(gruposPixeles)
  columnasImagen = len(gruposPixeles[0])
  # Segmenta la imagen
  imagenSegmentada = [[[0, 0, 0] for j in range(columnasImagen)] for i in range(filasImagen)]
  for filaImagen, filaGrupos in zip(imagenSegmentada, gruposPixeles): # Para cada fila
    for j, grupo_pixel in enumerate(filaGrupos): # Para cada pixel
      filaImagen[j] = centroides[grupo_pixel]
  return imagenSegmentada

# Obtiene la distancia entre 2 puntos en coordenadas homogeneas
def distanciaEntrePuntos(punto1, punto2):
  deltaX = punto1[0] - punto2[0]
  deltaY = punto1[1] - punto2[1]
  distancia = math.hypot(deltaX, deltaY)
  return distancia

# Convierte a escala de grises por NTSC
def convierteGrisesNTSC(imagen):
  if len(imagen[0][0]) > 2: # Si la imagen es a color
    # Obtiene las dimensiones
    filas = len(imagen)
    columnas = len(imagen[0])
    imagenGrises = [[0] * columnas for i in range(filas)]
    for i in range(filas): # Para cada fila
      for j in range(columnas): # Para cada fila
        # Obtiene los valores de los pixeles
        valorRojo = imagen[i][j][2]
        valorVerde = imagen[i][j][1]
        valorAzul = imagen[i][j][0]
        valorGris = 0.299 * valorRojo + 0.587 * valorVerde + 0.114 * valorAzul
        imagenGrises[i][j] = int(valorGris)
  else: # Si la imagen no es a color
    imagenGrises = imagen
  return imagenGrises

# Coloca los bordes para un filtro
def agregaBordesFiltroEscalaGrises(filasFiltro, columnasFiltro, imagen):
  # Variables de la imagen
  filasImagen = len(imagen)
  columnasImagen = len(imagen[0])
  # Variables de los bordes
  filasBorde = (filasFiltro - 1) // 2
  columnasBorde = (columnasFiltro - 1) // 2
  # Variables de la imagen con bordes
  filasImagenBordes = filasBorde + filasImagen + filasBorde;
  columnasImagenBordes = columnasBorde + columnasImagen + columnasBorde;
  imagenBordes = [[0] * columnasImagenBordes for i in range(filasImagenBordes)]
  for i in range(filasImagen): # Para cada fila
    for j in range(columnasImagen): # Para cada columna
      # Copia la imagen
      valorGris = imagen[i][j]
      imagenBordes[filasBorde + i][columnasBorde + j] = valorGris
  return imagenBordes

# Aplica un filtro en escala de grises por convolucion
def aplicaFiltroGrises(filtro, imagen):
  # Variables del filtro
  filasFiltro = len(filtro)
  columnasFiltro = len(filtro[0])
  # Variables de la imagen con filtro
  filasImagen = len(imagen) 
  columnasImagen = len(imagen[0])
  filasImagen -= filasFiltro - 1
  columnasImagen -= columnasFiltro - 1
  imagenFiltro = [[0] * columnasImagen for i in range(filasImagen)]
  for i in range(filasImagen): # Para cada fila
    for j in range(columnasImagen): # Para cada columna
      # Variables de la convolucion
      convolucion = 0
      for k in range(filasFiltro): # Para cada fila del filtro
        for l in range(columnasFiltro): # Para cada columna del filtro
          # Suma el valor del producto de los pixeles de la imagen con bordes y el filtro
          convolucion += filtro[k][l] * imagen[i + k][j + l] # i + a - a + k
      # Guarda el valor de la covolucion en el pixel
      imagenFiltro[i][j] = convolucion
  return imagenFiltro

# Binariza los pixeles de la imagen que crucen por 0
def binarizaImagenCrucesPorCero(imagen):
  # Constantes de los niveles de gris
  NEGRO = 0
  BLANCO = 255
  # Variables de la imagen binarizada
  filasImagen = len(imagen)
  columnasImagen = len(imagen[0])
  imagenBinarizada = [[NEGRO] * columnasImagen for i in range(filasImagen)]
  for i in range(1, filasImagen - 1): # Para cada fila con vecinos
    for j in range(1, columnasImagen - 1): # Para cada columna con vecinos
      # Obtiene el valor del pixel
      valorPixel = imagen[i][j]
      for k in range(-1, 1): # Para cada fila del vecindario
        for l in range(-1, 1): # Para cada columna del vecindario
          # Obtiene el valor del vecino
          valorVecino = imagen[i + k][j + l]
          if valorPixel * valorVecino <= 0: # Si hay un cruce por 0
            # Marca el borde
            imagenBinarizada[i][j] = BLANCO
            imagenBinarizada[i + k][j + l] = BLANCO
  return imagenBinarizada

# Binariza los pixeles de la imagen cuya diferencia es mayor a un umbral
def binarizaImagenUmbralDiferencias(imagen, UMBRAL):
  # Constantes de los niveles de gris
  NEGRO = 0
  BLANCO = 255
  # Variables de la imagen binarizada
  filasImagen = len(imagen)
  columnasImagen = len(imagen[0])
  imagenBinarizada = [[NEGRO] * columnasImagen for i in range(filasImagen)]
  for i in range(1, filasImagen - 1): # Para cada fila con vecinos
    for j in range(1, columnasImagen - 1): # Para cada columna con vecinos
      # Obtiene el valor del pixel
      valorPixel = imagen[i][j]
      for k in range(-1, 1): # Para cada fila del vecindario
        for l in range(-1, 1): # Para cada columna del vecindario
          # Obtiene el valor del vecino y la diferencia
          valorVecino = imagen[i + k][j + l]
          diferencia = abs(valorPixel - valorVecino)
          if diferencia > UMBRAL: # Si la diferencia es mayor al umbral
            # Marca el borde
            imagenBinarizada[i][j] = BLANCO
            imagenBinarizada[i + k][j + l] = BLANCO
  return imagenBinarizada

# Intersecta 2 imagenes binarias
def intersectaImagenesBinarias(imagen1, imagen2):
  # Constantes de los niveles de gris
  BLANCO = 255
  # Variables de la nueva imagen binarizada
  filasImagen = len(imagen1)
  columnasImagen = len(imagen1[0])
  imagenInterseccion = [[0] * columnasImagen for i in range(filasImagen)]
  for i in range(1, filasImagen - 1): # Para cada fila
    for j in range(1, columnasImagen - 1): # Para cada columna
      # Obtiene el valor de la interseccion (AND)
      valorInterseccion = imagen1[i][j] * imagen2[i][j]
      if valorInterseccion > BLANCO:
        valorInterseccion = BLANCO
      # Guarda el valor
      imagenInterseccion[i][j] = valorInterseccion
  return imagenInterseccion

# Crea la ventana del laplaciano del gaussiano
def creaVentanaFiltroLoG(filas, columnas, sigma):
  # Variables para la ventana
  ventanaLoG = [[0] * columnas for i in range(filas)]
  anchoBordeHorizontal = (filas - 1) // 2
  anchoBordeVertical = (columnas - 1) // 2
  # Variables para los coeficientes
  x = -anchoBordeVertical
  y = anchoBordeHorizontal
  for i in range(filas): # Para cada fila
    for j in range(columnas): # Para cada columna
      # Asigna los valores del filtro
      exponente = (x ** 2 + y ** 2) / (2 * sigma ** 2)
      valorCoeficiente = -(1 - exponente) * math.exp(-exponente) / (math.pi * sigma ** 4)
      ventanaLoG[i][j] = valorCoeficiente
      x += 1
    # Reinicializa la coordenada x
    x = -anchoBordeVertical;
    # Modifica la coordenada y
    y -= 1
  return ventanaLoG

# Devuelve una imagen a color, con el grupo seleccionado con sus colores originales y los demas con el color especificado
def seleccionaGrupoColor(imagen, gruposPixeles, grupoSeleccionado, colorOtrosGrupos=[0, 0, 0]):
  # Obtiene las dimensiones de la imagen
  filasImagen = len(imagen)
  columnasImagen = len(imagen[0])
  # Segmenta la imagen
  imagenSegmentada = [[[0, 0, 0] for j in range(columnasImagen)] for i in range(filasImagen)]
  for filaImagenSegmentada, filaGrupos, filaImagen in zip(imagenSegmentada, gruposPixeles, imagen): # Para cada fila
    for j, grupo_pixel in enumerate(filaGrupos): # Para cada pixel
      if grupo_pixel != grupoSeleccionado:
        filaImagenSegmentada[j] = colorOtrosGrupos
      else:
        filaImagenSegmentada[j] = filaImagen[j]
  return imagenSegmentada

# Obtiene las listas de los puntos de los bordes de los objetos en una imagen binarizada con los bordes en blanco
def obtenPuntosBordesObjetos(imagen):
  # Constantes de los valores binarizados
  BLANCO = 255
  # Obtiene las dimensiones de la imagen
  filasImagen = len(imagen)
  columnasImagen = len(imagen[0])
  # Obtiene los bordes
  adyacentes = [[-2, -2], [-2, -1], [-2,  0], [-2,  1], [-2,  2],
                [-1, -2], [-1, -1], [-1,  0], [-1,  1], [-1,  2],
                [ 0, -2], [ 0, -1], [ 0,  0], [ 0,  1], [ 0,  2],
                [ 1, -2], [ 1, -1], [ 1,  0], [ 1,  1], [ 1,  2],
                [ 2, -2], [ 2, -1], [ 2,  0], [ 2,  1], [ 2,  2]]
  pixelesVisitados = set()
  bordesObjetos = list()
  pila = list()
  for i in range(filasImagen): # Para cada fila
    for j in range(columnasImagen): # Para cada columna
      if imagen[i][j] == BLANCO and (i, j) not in pixelesVisitados: # Si es un pixel de borde de un nuevo objeto
        bordeObjeto = [(i, j)]
        pixelesVisitados.add((i, j))
        pila.append((i, j))
        while len(pila) > 0: # Mientras existan bordes sin procesar
          iActual, jActual = pila.pop()
          for movI, movJ in adyacentes: # Para cada vecino
            iNueva = iActual + movI
            jNueva = jActual + movJ
            if iNueva >= 0 and iNueva < filasImagen and jNueva >= 0 and jNueva < columnasImagen and imagen[iNueva][jNueva] == BLANCO and (iNueva, jNueva) not in pixelesVisitados: # Si es una casilla valida y un pixel de borde del objeto
              # Agrega el borde
              bordeObjeto.append((iNueva, jNueva))
              pixelesVisitados.add((iNueva, jNueva))
              pila.append((iNueva, jNueva))
        # Agrega el nuevo objeto
        bordesObjetos.append(bordeObjeto)
  return bordesObjetos

# Obtiene los extremos mas alejados de los bordes de los objetos ordenados por tamaño
def obtenMedidasObjetosPrincipales(bordesObjetos, numeroObjetos=1):
  extremosObjetos = list()
  for indice, bordeObjeto in enumerate(bordesObjetos): # Para cada objeto
    mejorExtremo = (0, (0, 0), (0, 0))
    for i in range(len(bordeObjeto) - 1): # Para cada punto del borde
      punto1 = bordeObjeto[i]
      for j in range(i + 1, len(bordeObjeto)): # Para cada punto contrario
        punto2 = bordeObjeto[j]
        distancia = distanciaEntrePuntos(punto1, punto2)
        mejorExtremo = max(mejorExtremo, (distancia, punto1, punto2))
    extremosObjetos.append((mejorExtremo, indice))
  # Valida el numero de objetos
  numeroObjetos = max(numeroObjetos, 0)
  numeroObjetos = min(numeroObjetos, len(bordesObjetos))
  # Determina si es un objeto relevante
  extremosObjetos.sort(reverse=True)
  totalObjetos = 0
  extremosObjetosPrincipales = list()
  while totalObjetos < numeroObjetos:
    extremosObjetosPrincipales.append((extremosObjetos[totalObjetos][1], extremosObjetos[totalObjetos][0]))
    totalObjetos += 1
  # Determina el indice del objeto
  extremosObjetosPrincipales.sort()
  totalObjetos = 0
  extremosObjetosPrincipalesObtenidos = list()
  while totalObjetos < numeroObjetos:
    extremosObjetosPrincipalesObtenidos.append((totalObjetos, extremosObjetosPrincipales[totalObjetos][1]))
    totalObjetos += 1
  return extremosObjetosPrincipalesObtenidos

# Convierte una imagen bgr a rgb
def convierteImagenBGRaRGB(imagenBGR):
  # Obtiene las dimensiones de la imagen
  filasImagen = len(imagenBGR)
  columnasImagen = len(imagenBGR[0])
  # Transforma la imagen
  imagenRGB = [[[0, 0, 0] for j in range(columnasImagen)] for i in range(filasImagen)]
  for filaImagenRGB, filaImagenBGR in zip(imagenRGB, imagenBGR): # Para cada fila
    for j in range(len(filaImagenRGB)): # Para cada pixel
      filaImagenRGB[j][0] = filaImagenBGR[j][2]
      filaImagenRGB[j][1] = filaImagenBGR[j][1]
      filaImagenRGB[j][2] = filaImagenBGR[j][0]
  return imagenRGB

# Obtiene el producto escalar entre 2 vectores
def productoEscalar(vector1, vector2):
  return sum(v1_i * v2_i for v1_i, v2_i in zip(vector1, vector2))

# Obtiene el producto vectorial entre 2 vectores en R3
def productoVectorial(vector1, vector2):
  v_1 = vector1[1] * vector2[2] - vector1[2] * vector2[1]
  v_2 = -(vector1[0] * vector2[2] - vector1[2] * vector2[0])
  v_3 = vector1[0] * vector2[1] - vector1[1] * vector2[0]
  return [v_1, v_2, v_3]

# Convierte un vector en R3 a coordenadas homogeneas
def convierteACoordenadasHomogeneas(vector):
  v_3 = vector[2]
  if v_3 != 0: # Si el punto existe
    v_1 = vector[0] / v_3
    v_2 = vector[1] / v_3
    v_3 = 1
    v = [v_1, v_2, v_3]
  else: # Si el punto no existe
    v = None
  return v

# Obtiene la distancia entre 2 puntos en coordenadas homogeneas
def distanciaEntrePuntos(punto1, punto2):
  deltaX = punto1[0] - punto2[0]
  deltaY = punto1[1] - punto2[1]
  distancia = math.hypot(deltaX, deltaY)
  return distancia

# Obtiene el punto de interseccion de 2 rectas
def intersectaRectas(vectorRecta1, vectorRecta2):
  p = productoVectorial(vectorRecta1, vectorRecta2) 
  p = convierteACoordenadasHomogeneas(p)
  return p

# Obtiene la recta que pasa por 2 puntos
def recta2Puntos(vectorPunto1, vectorPunto2):
  r = productoVectorial(vectorPunto1, vectorPunto2) 
  return r

# Grafica una recta con su vector
def graficaRecta(vectorRecta, minX, maxX, minY, maxY, color=None):
  [a, b, c] = vectorRecta
  x1, x2 = minX, maxX
  if b != 0: # ax + by + c = 0   =>   (ax + c) / -b = y
    y1 = (a * x1 + c) / -b
    y2 = (a * x2 + c) / -b
  else: # ax + c = 0   =>   x = -c / a
    x1 = x2 = -c / a
    y1 = minY
    y2 = maxY
  puntosX = [x1, x2]
  puntosY = [y1, y2]
  plt.plot(puntosX, puntosY, color=color)

# Grafica un punto con su vector
def graficaPunto(vectorPunto, color=None):
  if vectorPunto != None:
    [x, y, z] = vectorPunto
    plt.scatter(x, y, color=color)

# Grafica un conjunto de rectas y puntos
def graficaRectasYPuntos(rectas, puntos, minX, maxX, minY, maxY, coloresRectas=None, coloresPuntos=None):
  if coloresRectas == None:
    coloresRectas = [None] * len(rectas)
  if coloresPuntos == None:
    coloresPuntos = [None] * len(puntos)
  print("\nGrafica:")
  print("Rectas:")
  for recta, color in zip(rectas, coloresRectas):
    print(recta)
    graficaRecta(recta, minX, maxX, minY, maxY, color)
  print("Puntos:")
  for punto, color in zip(puntos, coloresPuntos):
    print(punto)
    graficaPunto(punto, color)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




















# Lee la imagen
imagenColor1 = leeImagen("Jit1.JPG") # 16 minutos
#imagenColor1 = leeImagen("Jit1.png") # 14 segundos

# Segmenta la imagen por color en 2 grupos
gruposPixeles1, centroides1 = segmentaColorKMedias(imagenColor1, 2)
imagenSegmentadaColor1 = asignaValorGruposImagenSegmentada(gruposPixeles1, centroides1)

# Obtiene la nueva imagen a color con el fondo oscuro
grupoRojo1 = obtenNuevoGrupoColor([0, 0, 255], centroides1)
imagenColor2 = seleccionaGrupoColor(imagenColor1, gruposPixeles1, grupoRojo1)

# Segmenta la imagen por color en 3 grupos
gruposPixeles2, centroides2 = segmentaColorKMedias(imagenColor2, 3)
imagenSegmentadaColor2 = asignaValorGruposImagenSegmentada(gruposPixeles2, centroides2)

# Obtiene la nueva imagen a color con el fondo oscuro
grupoRojo2 = obtenNuevoGrupoColor([0, 0, 255], centroides2)
imagenColor3 = seleccionaGrupoColor(imagenColor2, gruposPixeles2, grupoRojo2)

# Convierte la imagen en escala de grises
imagenGrises1 = convierteGrisesNTSC(imagenColor3)

# Crea los elementos necesarion para aplicar el filtro LoG
filasVentana1 = 5
columnasVentana1 = 5
sigmaVentana1 = 1.4
ventanaLoG1 = creaVentanaFiltroLoG(filasVentana1, columnasVentana1, sigmaVentana1)
imagenBordesFiltro1 = agregaBordesFiltroEscalaGrises(filasVentana1, columnasVentana1, imagenGrises1)

# Aplica el filtro LoG
imagenLoG1 = aplicaFiltroGrises(ventanaLoG1, imagenBordesFiltro1)
imagenBordesCeroLoG1 = binarizaImagenCrucesPorCero(imagenLoG1)
umbralLoG1 = 2
imagenBordesUmbralLoG1 = binarizaImagenUmbralDiferencias(imagenLoG1, umbralLoG1)
imagenBordesFinalLoG1 = intersectaImagenesBinarias(imagenBordesCeroLoG1, imagenBordesUmbralLoG1)

# Obtiene los bordes de todos los objetos
puntosBordes1 = obtenPuntosBordesObjetos(imagenBordesFinalLoG1)
print("Numero de objetos sin procesar", len(puntosBordes1))
print()

# Obtiene los diametros mayores de los objetos selecionados
extremosObjetos1 = obtenMedidasObjetosPrincipales(puntosBordes1, 4)

# Muestra los resultados
print("Numero de objetos despues de procesar", len(extremosObjetos1))
print()
print("Extremos de los objetos:")
for i, (distancia, punto1, punto2) in extremosObjetos1:
  print(f"Objeto {i + 1}:")
  print("Punto 1:", punto1)
  print("Punto 2:", punto2)
  print("Distancia:", distancia, "pixeles")
  print()

# Grafica los puntos y lineas
plt.figure(figsize=[10, 10])
plt.axis("off")
plt.imshow(convierteImagenBGRaRGB(imagenColor1))
for i, (distancia, (x1, y1), (x2, y2)) in extremosObjetos1:
  plt.plot([y1, y2], [x1, x2], "o-y")
  xm = (x1 + x2) / 2
  ym = (y1 + y2) / 2
  plt.text(ym, xm, f"{i+1}", size=20, ha="center", va="center", bbox=dict(boxstyle="round", fc=(1., 1., 1.)))

# Muestra y guarda las imagenes
muestraYGuardaImagen(imagenSegmentadaColor1, "colorSegmentada1")
muestraYGuardaImagen(imagenColor2, "colorFondoOscuro1")
muestraYGuardaImagen(imagenSegmentadaColor2, "colorSegmentada2")
muestraYGuardaImagen(imagenColor3, "colorFondoOscuro2")
muestraYGuardaImagen(imagenGrises1, "grisesFondoOscuro1")
muestraYGuardaImagen(imagenBordesFiltro1, "grisesBordesFiltro1")
muestraYGuardaImagen(imagenBordesCeroLoG1, "bordesLog1")
muestraYGuardaImagen(imagenBordesUmbralLoG1, "bordesLog2")
muestraYGuardaImagen(imagenBordesFinalLoG1, "bordesLog3")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
