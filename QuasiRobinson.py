import networkx as nx # type: ignore
import itertools
import numpy as np
import random
import heapq
import time
import os
from datetime import datetime
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt

import os
import pandas as pd



def disimilitud(n, m=None, inyectividad=False):
    """
    Crea 
    una disimilitud aleatoria 
                d : X x X -> [[m+1]]
    donde |X|=n, y d(x,y)=d[x][y]=d[y][x]. Esta función devuelve una matriz nxn cero-diagonal con esta disimilitud.
    Argumentos:
    - n: cantidad de elementos en X.
    - m: rango de valores para la disimilitud. Si no es dado, se asume igual a n.
    - inyectividad: indica si se quiere que la disimilitud devuelva valores distintos para cada par. Si no es dado, se asume Falso.
    """
    if m is None:
        m = n

    disimilaridad = np.zeros((n, n))

    if inyectividad:
        num_valores = n * (n - 1) // 2
        valores = np.random.permutation(np.arange(1, num_valores + 1))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                valor = valores[idx]
                disimilaridad[i, j] = valor
                disimilaridad[j, i] = valor
                idx += 1
    else:
        for i in range(n):
            for j in range(i + 1, n):
                valor = np.random.randint(1, m + 1)
                disimilaridad[i, j] = valor
                disimilaridad[j, i] = valor

    return disimilaridad

################################################################################################# 
# ROBINSON DISSIMILIRATY
def disimilitud_Rob(n):
    """
    Se genera una disimilitud Robinson simétrica 
              d_R : X x X -> [[m]]
    La salida es una matriz cuadrada cero-diagonal de orden n
    """

    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = int(abs(i-j))

    return D

#################################################################################################

def dis_euc(n):
    # Generar n puntos aleatorios en [0,1]x[0,1]
    points = np.random.rand(n, 2)   # matriz n x 2
    # A
    
    #  matriz de distancias
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(points[i] - points[j])
    
    return D, points

#################################################################################################

def dis_poincare(n):
    '''
    Dado n natural, se retorna una matriz nxn cuyas entradas representan la disimilitud entre dos elementos cualesquiera en un espacio topológico de Poincaré, además de la lista de puntos obtenida. 
    '''
    points = sample_ball(n)
    D = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            iso_inv = (2*np.linalg.norm(points[i]-points[j])**2)/((1-np.linalg.norm(points[i]))*(1-np.linalg.norm(points[j])))
            D[i,j] = np.arccosh(1+iso_inv)

    return D, points

###################################################################################################

def sample_ball(n, m=2):
    """
    Genera n puntos aleatorios en la bola unitaria de R^m.

    Parámetros:
    -----------
    n : int
        Número de puntos a generar.
    m : int, opcional
        Dimensión del espacio (por defecto 2).
    Retorna:
    --------
    np.ndarray o list
        Conjunto de puntos en la bola unitaria.
    """
    points = []
    for _ in range(n):
        # Dirección uniforme en la esfera
        direction = np.random.normal(0, 1, m)
        direction /= np.linalg.norm(direction)
        
        # Radio con distribución correcta
        r = np.random.rand() ** (1/m)
        
        # Punto en la bola
        points.append(r * direction)
    
    return np.array(points)

#################################################################################################



def rowandcol(M, i, j):
    """
    Se permutan simultáneamente las filas y columnas i y j de la matriz M
    """
    # Crear una copia de la matriz para no modificar la original
    M_permutada = M.copy()
    
    # Permutar las filas i y j
    M_permutada[[i, j], :] = M_permutada[[j, i], :]
    
    # Permutar las columnas i y j
    M_permutada[:, [i, j]] = M_permutada[:, [j, i]]
    
    return M_permutada 

#################################################################################################

def crear_hip(n):
    """
    # Para crear hipergrafos maunalmente, la funcion crear_hip 
    # devolverá la matriz de incidencia del hipergrafo construido
    # con n vértices
    """
    file1 = open('myhyp.txt','w')
    M=np.zeros(n)
    hedgelist = str(input('arista '))
    while hedgelist != '-1':
        file1.writelines(hedgelist+'\n')
        M = np.vstack((M,AddHEdge(hedgelist,n)))
        hedgelist = str(input('arista '))
    file1.close()
    return np.delete(M,0,0).T

#################################################################################################

def AddHEdge(l,n):
    """
    La función AddHEdge añadirá una hiperarista, dada una matriz de incidencia, una lista l de los vértices a unir, y la cantidad total de vértices n
    """
    zz = np.zeros(n)
    aux = list()  # guardamos los índices de los vértices de la hiperarista
    for x in list(l):
        aux.append(x)
        for i in aux:
            zz[i]=1
    return zz
  
#################################################################################################

def hyp(n,hyperedges):
    """
    Dada una lista de listas 'hyperedges', se construye una matriz de incidencia del hipergrafo H=([[n]],hyperedges)
    """
    M=np.zeros(n)
    for hedge in hyperedges:
        M=np.vstack((M,AddHEdge(hedge,n)))
    return np.delete(M,0,0).T
    # return M

#################################################################################################

def crear_hip_arx(n):
    """
    La funcion crear_hip_arx crea la matriz de incidencia de un hipergrafo
a partir de una lista de hiperaristas dada en un archivo llamado 'myhyp.txt'
de n vértices
    """
    with open('myhyp.txt') as LL:
        M=np.zeros(n)
        for line in LL:
            if line!='': M=np.vstack((M,AddHEdge(M,line[:-1],n)))
    return np.delete(M,0,0).T

#################################################################################################

def Kn(n,d):
    '''
    Dados n natural y d disimilitud sobre [[n]]x[[n]], Kn(n,d) retorna el grafo ponderado G_d=([[n]],E_d),
    donde para cada arista xy se cumple que w(xy)=d(x,y).
    '''
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1,n):
            G.add_edge(i,j, weight=d[i,j])
    return G

#################################################################################################

def all_minimum_spanning_trees(G):
    '''
    Dado un grafo completo G, la función retorna una lista de todos los árboles recubridores de peso mínimo de G,
    asumiendo que G es un grafo no dirigido con pesos.
    '''
    # Peso total del MST
    mst_edges = list(nx.minimum_spanning_edges(G, data=True))
    print(mst_edges)
    min_weight = sum(e[2]['weight'] for e in mst_edges)
    n = len(G.nodes)

    all_trees = []

    # Consideramos todas las combinaciones de n-1 aristas
    for edges in itertools.combinations(G.edges(data=True), n - 1):
        H = nx.Graph()
        H.add_nodes_from(G.nodes)
        H.add_edges_from([(u, v, {'weight': d['weight']}) for u, v, d in edges])
        if nx.is_connected(H) and len(H.edges) == n - 1:
            total_weight = sum(d['weight'] for u, v, d in H.edges(data=True))
            if np.isclose(total_weight, min_weight):
                all_trees.append(H.copy())

    return all_trees

#################################################################################################

# RANDOM DECREASING SEQUENCE
def rand_dec_seq(n):
    i=0
    orden = np.zeros(n)
    orden[0] = np.random.rand()
    while i!=n:
        z=np.random.rand()
        if z < orden[i]:
            orden[i+1] = z
            i=i+1
    return orden

#################################################################################################

def contar_hojas(arbol):
    # Contar los nodos que tienen grado 1 (son hojas)
    hojas = [nodo for nodo in arbol.nodes() if arbol.degree(nodo) == 1]
    return len(hojas)

#################################################################################################



#################################################################################################

def contiene_subcamino(camino_largo, subcamino):
    """
    Retorna True si subcamino (o su reverso) es una subsecuencia contigua de camino_largo.
    """
    n, m = len(camino_largo), len(subcamino)
    subcamino_rev = list(reversed(subcamino))
    
    for i in range(n - m + 1):
        ventana = camino_largo[i:i+m]
        if list(ventana) == list(subcamino) or list(ventana) == list(subcamino_rev):
            return True
    return False

#################################################################################################

def reconocer_caminos_robinson(T, D):
    """
    T: árbol (grafo de networkx)
    D: disimilitud (diccionario de pares de nodos)
    Retorna una tupla:
    .- PR_D(T)
    .- Lista de caminos que cumplen la condición Robinson
    .- Extremos caminos Robinson
    .- RE_D(T)
    """
    RE = 0
    diam = nx.diameter(T)
    caminos_por_k = caminos_por_longitud(T)

    caminos_robinson = []
    # Considerar caminos de largo 1. Puede ser añadiendo todas las aristas del árbol a la lista de salida
    descartados = set()

    for k in range(2, diam + 1):
        if k not in caminos_por_k:
            continue
        for camino in caminos_por_k[k]:
            if camino in descartados:
                continue

            if es_camino_robinson(camino, D):
                caminos_robinson.append(camino)
                RE += k
            else:
                # Descartar todos los caminos más largos que contengan este
                for m in range(k+1, diam+1):
                    if m not in caminos_por_k:
                        continue
                    for p_m in caminos_por_k[m]:
                        if contiene_subcamino(p_m, camino):
                            descartados.add(p_m)
                            
    return len(caminos_robinson)+(len(D)-1), caminos_robinson, [(c[0], c[-1]) for c in caminos_robinson], RE + (len(D)-1)


#################################################################################################

def caminos_por_longitud(T):
    """
    Dado un árbol T, retorna un diccionario {k: lista_de_caminos_de_longitud_k}
    donde cada camino es una tupla de nodos (v0, v1, ..., vk)
    """
    from itertools import combinations
    caminos_por_k = dict()

    # Recorremos todos los pares de nodos (como T es árbol, hay un único camino entre cada par)
    for u, v in combinations(T.nodes, 2):
        camino = nx.shortest_path(T, source=u, target=v)
        k = len(camino) - 1  # largo del camino = número de aristas
        if k not in caminos_por_k:
            caminos_por_k[k] = []
        caminos_por_k[k].append(tuple(camino))

    return caminos_por_k

#################################################################################################

def generar_arboles_reconectando_hojas(T):
    '''
    Dado un árbol T, esta función retorna una lista de todos los árboles resultantes tras reubicar cada hoja del árbol T original adyacente a cada uno del resto de nodos.
    '''
    arboles_generados = []

    # Identificamos las hojas (grado 1)
    hojas = [n for n in T.nodes if T.degree[n] == 1]

    for h in hojas:
        l = next(T.neighbors(h))  # l es el único vecino de h

        # Creamos una copia del árbol sin la hoja h
        T_sin_h = T.copy()
        T_sin_h.remove_node(h)

        # Reconectamos h a cada otro nodo distinto de l
        for nuevo_vecino in T_sin_h.nodes:
            if nuevo_vecino == l:
                continue  # no reconectamos con el original

            nuevo_arbol = T_sin_h.copy()
            nuevo_arbol.add_node(h)
            nuevo_arbol.add_edge(h, nuevo_vecino)
            arboles_generados.append(nuevo_arbol)

    return arboles_generados

#################################################################################################

def generar_arboles_reconectando_aristas(T):
    """
    Genera árboles reconectando componentes tras eliminar cada arista de T.

    Parámetro:
    - T: árbol (grafo no dirigido acíclico conectado) en formato NetworkX.

    Retorna:
    - lista de nuevos árboles generados por la operación descrita.
    """
    nuevos_arboles = set()

    for u, v in T.edges():
        # se crea una copia del árbol y se elimina la arista (u, v)
        T_sin_e = T.copy()
        T_sin_e.remove_edge(u, v)

        # Obtenemos las dos componentes conexas resultantes
        componentes = list(nx.connected_components(T_sin_e))
        if len(componentes) != 2:
            continue  # solo para chequear

        comp1, comp2 = componentes

        # Para cada combinación de nodos entre las dos componentes, conectamos con una nueva arista
        for a, b in itertools.product(comp1, comp2):
            if {a, b} == {u, v}:
                continue  # No añadir de nuevo la misma arista original

            T_nuevo = T_sin_e.copy()
            T_nuevo.add_edge(a, b)
            nuevos_arboles.add(T_nuevo)

    return nuevos_arboles

#################################################################################################

def es_mst_unico(D):
    """
    Determina si el Árbol Recubridor Mínimo (MST) de una matriz de disimilitud D es único.
    
    Parámetros:
      D: Matriz de adyacencia/pesos (numpy array o lista de listas).
         D[i][j] es el peso de la arista entre i y j.
         
    Retorna:
      True si el MST es único.
      False si existen múltiples MSTs.
    """
    D = np.array(D)
    n = len(D)
    G = nx.Graph()
    
    # 1. Construir el grafo completo a partir de la matriz D
    for i in range(n):
        for j in range(i + 1, n):
            # Asumimos grafo conexo y no dirigido
            G.add_edge(i, j, weight=D[i][j])
            
    # 2. Calcular UN Árbol Recubridor Mínimo (MST) de referencia
    # Kruskal o Prim funcionan bien aquí.
    T = nx.minimum_spanning_tree(G)
    
    # 3. Verificar la condición de unicidad para cada arista NO perteneciente al árbol
    for u, v, data in G.edges(data=True):
        # Si la arista (u,v) YA está en el MST, la ignoramos
        if T.has_edge(u, v):
            continue
            
        peso_externo = data['weight']
        
        # Encontramos el camino único en el árbol T entre u y v
        # nx.shortest_path en un árbol devuelve el único camino posible
        try:
            camino = nx.shortest_path(T, source=u, target=v)
        except nx.NetworkXNoPath:
            # Esto solo pasaría si el grafo original no fuera conexo
            raise ValueError("El grafo no es conexo, no tiene un MST global.")
        
        # Buscamos el peso máximo dentro de ese camino en T
        max_peso_ciclo = -float('inf')
        
        for k in range(len(camino) - 1):
            n1, n2 = camino[k], camino[k+1]
            # Obtenemos el peso de la arista en el árbol
            peso_arista_t = T[n1][n2]['weight']
            if peso_arista_t > max_peso_ciclo:
                max_peso_ciclo = peso_arista_t
        
        # 4. Comparación crítica
        # Si la arista externa pesa LO MISMO que la más pesada del ciclo,
        # podríamos intercambiarlas y tener otro MST con el mismo peso total.
        # Usamos np.isclose para evitar errores de punto flotante.
        if np.isclose(peso_externo, max_peso_ciclo):
            return False  # ¡Encontramos un empate! NO es único.

    # Si revisamos todas las aristas externas y todas eran estrictamente mayores
    return True


######################################################################################

def w(T, D):
    """
    Calcula el peso total de un árbol T, tomando los pesos desde la matriz D.

    Parámetros:
    - T: grafo de networkx (debe ser un árbol)
    - D: matriz de disimilitud (numpy array o lista de listas), D[i][j] es el peso entre i y j

    Retorna:
    - suma total de los pesos de las aristas del árbol, usando D
    """
    peso_total = 0
    for u, v in T.edges():
        peso_total += D[u][v]
    return peso_total

#################################################################################################

def MST(D):
    '''
    Dada una disimilitud D, MST(D) devuelve el árbol recubridor de peso mínimo del espacio de disimilitud dado
    '''
    G = Kn(len(D),D)
    return nx.minimum_spanning_tree(G, weight='weight')

def es_mst_unico(D):
    """
    Determina si el Árbol Recubridor Mínimo (MST) de una matriz de disimilitud D es único.
    
    Parámetros:
      D: Matriz de adyacencia/pesos (numpy array o lista de listas).
         D[i][j] es el peso de la arista entre i y j.
         
    Retorna:
      True si el MST es único.
      False si existen múltiples MSTs.
    """
    D = np.array(D)
    n = len(D)
    G = nx.Graph()
    
    # 1. Construir el grafo completo a partir de la matriz D
    for i in range(n):
        for j in range(i + 1, n):
            # Asumimos grafo conexo y no dirigido
            G.add_edge(i, j, weight=D[i][j])
            
    # 2. Calcular UN Árbol Recubridor Mínimo (MST) de referencia
    # Kruskal o Prim funcionan bien aquí.
    T = nx.minimum_spanning_tree(G)
    
    # 3. Verificar la condición de unicidad para cada arista NO perteneciente al árbol
    for u, v, data in G.edges(data=True):
        # Si la arista (u,v) YA está en el MST, la ignoramos
        if T.has_edge(u, v):
            continue
            
        peso_externo = data['weight']
        
        # Encontramos el camino único en el árbol T entre u y v
        # nx.shortest_path en un árbol devuelve el único camino posible
        try:
            camino = nx.shortest_path(T, source=u, target=v)
        except nx.NetworkXNoPath:
            # Esto solo pasaría si el grafo original no fuera conexo
            raise ValueError("El grafo no es conexo, no tiene un MST global.")
        
        # Buscamos el peso máximo dentro de ese camino en T
        max_peso_ciclo = -float('inf')
        
        for k in range(len(camino) - 1):
            n1, n2 = camino[k], camino[k+1]
            # Obtenemos el peso de la arista en el árbol
            peso_arista_t = T[n1][n2]['weight']
            if peso_arista_t > max_peso_ciclo:
                max_peso_ciclo = peso_arista_t
        
        # 4. Comparación crítica
        # Si la arista externa pesa LO MISMO que la más pesada del ciclo,
        # podríamos intercambiarlas y tener otro MST con el mismo peso total.
        # Usamos np.isclose para evitar errores de punto flotante.
        if np.isclose(peso_externo, max_peso_ciclo):
            return False  # ¡Encontramos un empate! NO es único.

    # Si revisamos todas las aristas externas y todas eran estrictamente mayores
    return True



class UnionFind:
    """Estructura de datos para gestión eficiente de conjuntos disjuntos."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
            return True
        return False

# def buscar_mst_distintos(D, tiempo_limite_segundos):
#     """
#     Busca la mayor cantidad de MSTs distintos posibles en un tiempo T.
    
#     Args:
#         D (np.array o list): Matriz de disimilitud nxn.
#         tiempo_limite_segundos (float): Tiempo máximo de ejecución en segundos.
        
#     Returns:
#         int: Cantidad de árboles únicos encontrados.
#         set: Conjunto conteniendo los árboles (como frozensets de aristas).
#     """
#     start_time = time.time()
#     n = len(D)
    
#     # 1. Extraer todas las aristas de la matriz (u, v, peso)
#     aristas = []
#     for i in range(n):
#         for j in range(i + 1, n):
#             aristas.append((i, j, D[i][j]))
    
#     # 2. Ordenar aristas por peso (fundamental para Kruskal)
#     aristas.sort(key=lambda x: x[2])
    
#     # 3. Agrupar aristas por peso
#     # Esto es CRUCIAL: Solo podemos mezclar el orden de aristas que pesan lo mismo.
#     # itertools.groupby requiere que la lista ya esté ordenada por la clave.
#     grupos_por_peso = []
#     for peso, grupo in itertools.groupby(aristas, key=lambda x: x[2]):
#         grupos_por_peso.append(list(grupo))
        
#     mst_encontrados = set()
#     iteraciones = 0
    
#     # 4. Bucle principal: Generar MSTs hasta que se acabe el tiempo
#     while time.time() - start_time < tiempo_limite_segundos:
#         uf = UnionFind(n)
#         aristas_mst_actual = []
#         n_aristas_agregadas = 0
        
#         # Ejecutar Kruskal
#         for grupo in grupos_por_peso:
#             # Si el grupo tiene más de 1 arista, mezclamos para explorar variantes
#             # (Hacemos una copia para no alterar el orden original para la siguiente vuelta)
#             grupo_actual = grupo[:]
#             if len(grupo_actual) > 1:
#                 random.shuffle(grupo_actual)
            
#             for u, v, w in grupo_actual:
#                 if uf.union(u, v):
#                     # Guardamos la arista ordenada para que (1,2) sea igual a (2,1)
#                     arista_normalizada = tuple(sorted((u, v)))
#                     aristas_mst_actual.append(arista_normalizada)
#                     n_aristas_agregadas += 1
            
#             # Optimización: Si ya tenemos n-1 aristas, el árbol está completo
#             if n_aristas_agregadas == n - 1:
#                 break
        
#         # Convertimos la lista de aristas a un frozenset para poder guardarlo en un set
#         # (los sets eliminan duplicados automáticamente)
#         arbol_hash = frozenset(aristas_mst_actual)
#         mst_encontrados.add(arbol_hash)
#         iteraciones += 1

#     # Convertimos cada conjunto de aristas único en un objeto nx.Graph
#     lista_grafos_nx = []
    
#     for conjunto_aristas in mst_encontrados:
#         T = nx.Graph()
#         T.add_nodes_from(range(n)) # Aseguramos que tenga los n nodos (incluso si n=1)
        
#         for u, v in conjunto_aristas:
#             peso = D[u][v] # Recuperamos el peso de la matriz original
#             T.add_edge(u, v, weight=peso)
            
#         lista_grafos_nx.append(T)

#     # print(f"Tiempo finalizado. Iteraciones totales: {iteraciones}")
#     return len(lista_grafos_nx), lista_grafos_nx

# (Asegúrate de tener importada tu clase UnionFind aquí)

def buscar_mst_distintos(D, tiempo_limite_segundos):
    """
    Busca la mayor cantidad de MSTs distintos posibles en un tiempo estricto T.
    """
    start_time = time.time()
    n = len(D)
    
    aristas = []
    for i in range(n):
        for j in range(i + 1, n):
            aristas.append((i, j, D[i][j]))
            
    aristas.sort(key=lambda x: x[2])
    
    grupos_por_peso = []
    for peso, grupo in itertools.groupby(aristas, key=lambda x: x[2]):
        grupos_por_peso.append(list(grupo))
        
    mst_encontrados_hashes = set() # Solo para llevar control de cuáles ya vimos
    lista_grafos_nx = []           # Lista final de salida
    
    # 4. Bucle principal: Generar y CONSTRUIR MSTs hasta que se acabe el tiempo
    while time.time() - start_time < tiempo_limite_segundos:
        uf = UnionFind(n) # Asumo que tienes tu clase UnionFind definida
        aristas_mst_actual = []
        n_aristas_agregadas = 0
        
        for grupo in grupos_por_peso:
            grupo_actual = grupo[:]
            if len(grupo_actual) > 1:
                random.shuffle(grupo_actual)
            
            for u, v, w in grupo_actual:
                if uf.union(u, v):
                    arista_normalizada = tuple(sorted((u, v)))
                    aristas_mst_actual.append(arista_normalizada)
                    n_aristas_agregadas += 1
            
            if n_aristas_agregadas == n - 1:
                break
                
        arbol_hash = frozenset(aristas_mst_actual)
        
        # 5. Si encontramos un árbol nuevo, construimos el nx.Graph DENTRO del cronómetro
        if arbol_hash not in mst_encontrados_hashes:
            mst_encontrados_hashes.add(arbol_hash)
            
            T = nx.Graph()
            T.add_nodes_from(range(n)) 
            
            for u, v in arbol_hash:
                peso = D[u][v] 
                T.add_edge(u, v, weight=peso)
                
            lista_grafos_nx.append(T)

    return len(lista_grafos_nx), lista_grafos_nx

#################################################################################################

def exp1(n,nexp):
    '''
    Dado n número de elementos en el espacio, nexp cantidad de experimentos, se generan nexp espacios de disimilitud distintos de tamaño n. Se calculan peso los árboles de peso minimo, así como su PR en todo su entorno de hojas. Se retornan tres elementos: lista de vecinos
    '''
    D = disimilitud(n)
    gt = set()
    mst = MST(D)
    delta = reconocer_caminos_robinson(mst,D)[0]
    
    # Considerar D anterior !!
    muestra = comparacion_w_PR(D,nexp)

    for x in muestra:
        if x[0][1]>delta:
            gt.add((w(x[1],D),x[0][1],tuple(sorted(x[1].edges()))))

    muestras = set()
    for T in gt:
        G = nx.Graph()
        G.add_edges_from(T[2])
        aux2 = generar_arboles_reconectando_hojas(G)
        for t in aux2:
            aux3 = reconocer_caminos_robinson(t,D)
            if aux3[0]>delta:
                muestras.add((w(t,D),aux3[0],tuple(sorted(t.edges()))))
    

    return gt,muestras, D

def exp2(n,nexp):
    '''
    Dado n, se genera una disimilitud aleatoria (X,D) con |X|=n. Sampleamos nexp árboles, a los que calculamos su peso y métrica PR. Si un árbol aleatorio presenta mejor métrica que el MST, entonces además calculamos los árboles vecinos (por aristas) que se encuentran a distancia 1 del 'buen árbol'
    
    :param n: cantidad de elementos de la disimilitud
    :param nexp: cantidad de árboles sampleados
    '''

    p = 0 # contador de progreso
    D = disimilitud(n)
    gt = set()
    mst = MST(D)
    delta = reconocer_caminos_robinson(mst,D)[0]
    
    print(f'exp2:{p}%')

    muestra = comparacion_w_PR(D,nexp)
    p+=30
    print(f'exp2:{p}%')
    for x in muestra:
        if x[0][1]>delta:
            # gt es por 'good trees'
            gt.add((w(x[1],D),x[0][1],tuple(sorted(x[1].edges()))))
            
    p+=25
    print(f'exp2:{p}%')

    muestras = set()
    i=0
    for T in gt:
        i+=1
        G = nx.Graph()
        G.add_edges_from(T[2])

        if 100*(i+1)/len(gt)>p:
            print(f'exp2: {p}%')
            p+=5

        aux2 = generar_arboles_reconectando_aristas(G)
        for t in aux2:
            aux3 = reconocer_caminos_robinson(t,D)
            if aux3[0]>delta:
                muestras.add((w(t,D),aux3[0],tuple(sorted(t.edges()))))
    

    return gt,muestras, D


def exp3(n, n_disimilitudes):
    '''
    Dado n
    '''
    output_final = list()
    for i in range(n_disimilitudes):
        experimento = list()
        D = disimilitud(n,n*(n-1)/2)
        S = set()
        max_T = n**(n-2)
        while len(S)< 0.9*max_T:
            print(f'{len(S)*100/max_T}% completado')
            T = nx.random_labeled_tree(n)
            L1 = generar_arboles_reconectando_aristas(T)
            L2 = generar_arboles_reconectando_hojas(T)

            for T1 in L1:
                aux = tuple(sorted(T1.edges()))
                if (aux in S):
                    continue
                else:
                    S.add(aux)
                    experimento.append((reconocer_caminos_robinson(T1,D)[0],w(T1,D), T1))
            
            for T2 in L2:
                aux = tuple(sorted(T2.edges()))
                if (aux in S):
                    continue
                else:
                    S.add(aux)
                    experimento.append((reconocer_caminos_robinson(T2,D)[0],w(T2,D), T2))

        output_final.append([D,experimento])
    
    return(output_final)

# Modificación 1/Sept: código provisto por IA

from itertools import product
import heapq
from typing import List, Tuple, Iterable

def _edges_from_pruefer(seq: List[int], n: int) -> List[Tuple[int, int]]:
    """
    Construye la lista de aristas (u, v) de un árbol etiquetado en {0,...,n-1}
    a partir de una secuencia de Prüfer 'seq' de largo n-2.
    """
    degree = [1] * n
    for v in seq:
        degree[v] += 1

    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)

    edges = []
    for v in seq:
        u = heapq.heappop(leaves)
        edges.append((u, v))
        degree[u] -= 1
        degree[v] -= 1
        if degree[v] == 1:
            heapq.heappush(leaves, v)

    u = heapq.heappop(leaves)
    w = heapq.heappop(leaves)
    edges.append((u, w))
    return edges

def _all_labeled_trees_edges(n: int) -> Iterable[List[Tuple[int, int]]]:
    """
    Genera TODAS las listas de aristas (u, v) de los árboles etiquetados en {0,...,n-1}
    usando la biyección de Prüfer. Cantidad total: n^(n-2).
    """
    if n < 2:
        yield []
        return
    if n == 2:
        yield [(0, 1)]
        return

    for seq in product(range(n), repeat=n-2):
        yield _edges_from_pruefer(list(seq), n)

def exp3_new(n: int, n_disimilitudes: int):
    """
    Genera n_disimilitudes matrices D = disimilitud(n), enumera TODOS los árboles recubridores
    de K_n ponderados por D[i][j], aplica reconocer_caminos_robinson(T_j, D) a cada árbol, y
    construye las salidas requeridas.

    Salida:
      - output_PR: lista (por cada D) de listas de tuplas (n, (2*aux_PR)/(n*(n-1))).
      - output_RE: lista (por cada D) de listas de tuplas (n, aux_RE).
      - Disimilitudes: lista de las matrices D generadas por disimilitud(n).
    """
    output_PR: List[List[Tuple[int, float]]] = []
    output_RE: List[List[Tuple[int, float]]] = []
    Disimilitudes: List[List[List[float]]] = []

    for _ in range(n_disimilitudes):
        D = disimilitud(n) 
        
        Disimilitudes.append(D)

        pr_list: List[Tuple[int, float]] = []
        re_list: List[Tuple[int, float]] = []

        for edges in _all_labeled_trees_edges(n):
            # Construir el grafo con NetworkX
            T_j = nx.Graph()
            T_j.add_nodes_from(range(n))
            for u, v in edges:
                T_j.add_edge(u, v, weight=D[u][v])

            res = reconocer_caminos_robinson(T_j, D)  
            aux_PR = res[0]
            aux_RE = res[3]

            pr_list.append((n, (2 * aux_PR) / (n * (n - 1))))
            re_list.append((n, aux_RE))

        output_PR.append(pr_list)
        output_RE.append(re_list)

    return output_PR, output_RE, Disimilitudes

######################################## FIN IA #######################

def exp3_new_euc(n: int, n_disimilitudes: int):
    """
    Genera n_disimilitudes matrices D = disimilitud(n), enumera TODOS los árboles recubridores
    de K_n ponderados por D[i][j], aplica reconocer_caminos_robinson(T_j, D) a cada árbol, y
    construye las salidas requeridas.

    Salida:
      - output_PR: lista (por cada D) de listas de tuplas (n, (2*aux_PR)/(n*(n-1))).
      - output_RE: lista (por cada D) de listas de tuplas (n, aux_RE).
      - Disimilitudes: lista de las matrices D generadas por disimilitud(n).
    """
    output_PR: List[List[Tuple[int, float]]] = []
    output_RE: List[List[Tuple[int, float]]] = []
    Disimilitudes: List[List[List[float]]] = []

    for _ in range(n_disimilitudes):
        D = dis_euc(n)  # <- función provista por ti
        Disimilitudes.append(D)

        pr_list: List[Tuple[int, float]] = []
        re_list: List[Tuple[int, float]] = []

        for edges in _all_labeled_trees_edges(n):
            # Construir el grafo con NetworkX
            T_j = nx.Graph()
            T_j.add_nodes_from(range(n))
            for u, v in edges:
                T_j.add_edge(u, v, weight=D[u][v])

            res = reconocer_caminos_robinson(T_j, D)  # <- función provista por ti
            aux_PR = res[0]
            aux_RE = res[3]

            pr_list.append((n, (2 * aux_PR) / (n * (n - 1))))
            re_list.append((n, aux_RE))

        output_PR.append(pr_list)
        output_RE.append(re_list)

    return output_PR, output_RE, Disimilitudes

def exp3_new_poincare(n: int, n_disimilitudes: int):
    """
    Genera n_disimilitudes matrices D = disimilitud(n), enumera TODOS los árboles recubridores
    de K_n ponderados por D[i][j], aplica reconocer_caminos_robinson(T_j, D) a cada árbol, y
    construye las salidas requeridas.

    Salida:
      - output_PR: lista (por cada D) de listas de tuplas (n, (2*aux_PR)/(n*(n-1))).
      - output_RE: lista (por cada D) de listas de tuplas (n, aux_RE).
      - Disimilitudes: lista de las matrices D generadas por disimilitud(n).
    """
    output_PR: List[List[Tuple[int, float]]] = []
    output_RE: List[List[Tuple[int, float]]] = []
    Disimilitudes: List[List[List[float]]] = []

    for _ in range(n_disimilitudes):
        D = dis_poincare(n)  # <- función provista por ti
        Disimilitudes.append(D)

        pr_list: List[Tuple[int, float]] = []
        re_list: List[Tuple[int, float]] = []

        for edges in _all_labeled_trees_edges(n):
            # Construir el grafo con NetworkX
            T_j = nx.Graph()
            T_j.add_nodes_from(range(n))
            for u, v in edges:
                T_j.add_edge(u, v, weight=D[u][v])

            res = reconocer_caminos_robinson(T_j, D)  # <- función provista por ti
            aux_PR = res[0]
            aux_RE = res[3]

            pr_list.append((n, (2 * aux_PR) / (n * (n - 1))))
            re_list.append((n, aux_RE))

        output_PR.append(pr_list)
        output_RE.append(re_list)

    return output_PR, output_RE, Disimilitudes


def exp4():
    N = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,33,36,39,42,45,48,51,54,57,60,65,70,75,80,85,90,95,100]
    
    output = list()
    output2 = list()
    
    for n in N:
        print(f'Calculando para n = {n}')
        if n<=20:
            for i in range(20):
                D = disimilitud(n)
                T = MST(D)
                aux = reconocer_caminos_robinson(T,D)
                PR_T, RE_T = aux[0], aux[3]
                output.append((n,(2*PR_T)/(n*(n-1))))
                output2.append((n,RE_T))
        else:
            for i in range(20):
                D = disimilitud(n)
                T = MST(D)
                aux = reconocer_caminos_robinson(T,D)
                PR_T, RE_T = aux[0], aux[3]
                output.append((n,(2*PR_T)/(n*(n-1))))
                output2.append((n,RE_T))
        # else:
        #     D = disimilitud(n)
        #     T = MST(D)
        #     aux = reconocer_caminos_robinson(T,D)
        #     PR_T, RE_T = aux[0], aux[3]
        #     output.append((n,(2*PR_T)/(n*(n-1))))
        #     output2.append((n,RE_T))

    return output, output2


def exp5():
    output_PR = list()
    output_RE = list()
    N=[100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,320,340,360]
    
    for n in N:
        for i in range(20):
            D = disimilitud(n)
            T = MST(D)
            aux = reconocer_caminos_robinson(T,D)
            PR_T, RE_T = aux[0], aux[3]
            output_PR.append((n,(2*PR_T)/(n*(n-1))))
            output_RE.append((n,RE_T))

    return output_PR, output_RE

def exp6():
    output_PR = list()
    output_RE = list()
    D_n = list()
    M=[100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,320,340,360]
    
    N = [int(x/10) for x in M]

    for n in N:
        for i in range(20):
            D = disimilitud(n)
            T = MST(D)
            aux = reconocer_caminos_robinson(T,D)
            PR_T, RE_T = aux[0], aux[3]
            output_PR.append((n,(2*PR_T)/(n*(n-1))))
            output_RE.append((n,RE_T))
            D_n.append(D)

    return output_PR, output_RE, D_n

def exp7():
    output_PR = list()
    output_RE = list()
    D_n = list()
    N=range(5,50,5)+range(50,100,10)+[100,150,200,250,300,350]
    
    for i in range(20):
        print(f'Calculando experimento i = {i}')
        for n in N:
            print(f'{i}-esimo experimento para n={n}')
            D = disimilitud(n)
            T = nx.random_labeled_tree(n)
            aux = reconocer_caminos_robinson(T,D)
            PR_T, RE_T = aux[0], aux[3]
            output_PR.append((n,(2*PR_T)/(n*(n-1))))
            output_RE.append((n,RE_T))
            # for j in range(10):
            #     print(f'Calculando {j}-esimo ejemplo aleatorio para la {i}-esima dism para n = {n}')
            #     T_j = nx.random_labeled_tree(n)
            #     aux_j = reconocer_caminos_robinson(T_j,D)
            #     PR_T_j, RE_T_j = aux_j[0], aux_j[3]
            #     output_PR_T_random.append((n,(2*PR_T_j)/(n*(n-1))))
            #     output_RE_T_random.append((n,RE_T_j))

    return output_PR, output_RE


def exp7_mod():
    output_PR = list()
    output_RE = list()
    N=range(5,50)+range(50,100,5)+[100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,320,340,360]
    
    for n in N:
        print(f'Calculando para n = {n}')
        for i in range(20):
            D = disimilitud(n)
            T = nx.random_labeled_tree(n)
            aux = reconocer_caminos_robinson(T,D)
            PR_T, RE_T = aux[0], aux[3]
            output_PR.append((n,(2*PR_T)/(n*(n-1))))
            output_RE.append((n,RE_T))
            # for j in range(10):
            #     print(f'Calculando {j}-esimo ejemplo aleatorio para la {i}-esima dism para n = {n}')
            #     T_j = nx.random_labeled_tree(n)
            #     aux_j = reconocer_caminos_robinson(T_j,D)
            #     PR_T_j, RE_T_j = aux_j[0], aux_j[3]
            #     output_PR_T_random.append((n,(2*PR_T_j)/(n*(n-1))))
            #     output_RE_T_random.append((n,RE_T_j))

    return output_PR, output_RE



def exp7_chiquito():
    output_PR = list()
    output_PR_T_random = list()
    output_RE = list()
    output_RE_T_random = list()
    D_n = list()
    N=range(5,50)
    
    for n in N:
        print(f'Calculando para n = {n}')
        for i in range(20):
        
            D = disimilitud(n)
            T = MST(D)
            aux = reconocer_caminos_robinson(T,D)
            PR_T, RE_T = aux[0], aux[3]
            output_PR.append((n,(2*PR_T)/(n*(n-1))))
            output_RE.append((n,RE_T))
            for j in range(10):
                print(f'Calculando {j}-esimo ejemplo aleatorio para la {i}-esima dism para n = {n}')
                T_j = nx.random_labeled_tree(n)
                aux_j = reconocer_caminos_robinson(T_j,D)
                PR_T_j, RE_T_j = aux_j[0], aux_j[3]
                output_PR_T_random.append((n,(2*PR_T_j)/(n*(n-1))))
                output_RE_T_random.append((n,RE_T_j))

    return output_PR, output_PR_T_random, output_RE, output_RE_T_random 

def exp8_chiquitito():
    output_PR = list()
    output_PR_T_random = list()
    output_RE = list()
    output_RE_T_random = list()
    D_n = list()
    N=range(5,50)
    
    for n in N:
        print(f'Calculando para n = {n}')
        for i in range(20):
        
            D = dis_euc(n)[0]
            T = MST(D)
            aux = reconocer_caminos_robinson(T,D)
            PR_T, RE_T = aux[0], aux[3]
            output_PR.append((n,(2*PR_T)/(n*(n-1))))
            output_RE.append((n,RE_T))
            for j in range(10):
                print(f'Calculando {j}-esimo ejemplo aleatorio para la {i}-esima dism para n = {n}')
                T_j = nx.random_labeled_tree(n)
                aux_j = reconocer_caminos_robinson(T_j,D)
                PR_T_j, RE_T_j = aux_j[0], aux_j[3]
                output_PR_T_random.append((n,(2*PR_T_j)/(n*(n-1))))
                output_RE_T_random.append((n,RE_T_j))

    return output_PR, output_PR_T_random, output_RE, output_RE_T_random     

def exp8_mid():
    N = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,33,36,39,42,45,48,51,54,57,60,65,70,75,80,85,90,95,100]
    
    output = list()
    output2 = list()
    
    for n in N:
        print(f'Calculando para n = {n}')
        if n<=20:
            for i in range(20):
                D = dis_euc(n)[0]
                T = MST(D)
                aux = reconocer_caminos_robinson(T,D)
                PR_T, RE_T = aux[0], aux[3]
                output.append((n,(2*PR_T)/(n*(n-1))))
                output2.append((n,RE_T))
        else:
            for i in range(20):
                D = dis_euc(n)[0]
                T = MST(D)
                aux = reconocer_caminos_robinson(T,D)
                PR_T, RE_T = aux[0], aux[3]
                output.append((n,(2*PR_T)/(n*(n-1))))
                output2.append((n,RE_T))
        # else:
        #     D = disimilitud(n)
        #     T = MST(D)
        #     aux = reconocer_caminos_robinson(T,D)
        #     PR_T, RE_T = aux[0], aux[3]
        #     output.append((n,(2*PR_T)/(n*(n-1))))
        #     output2.append((n,RE_T))

    return output, output2


def exp9_mid():
    N = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,33,36,39,42,45,48,51,54,57,60,65,70,75,80,85,90,95,100]
    
    output = list()
    output2 = list()
    
    for n in N:
        print(f'Calculando para n = {n}')
        if n<=20:
            for i in range(20):
                D = dis_poincare(n)[0]
                T = MST(D)
                aux = reconocer_caminos_robinson(T,D)
                PR_T, RE_T = aux[0], aux[3]
                output.append((n,(2*PR_T)/(n*(n-1))))
                output2.append((n,RE_T))
        else:
            for i in range(20):
                D = dis_poincare(n)[0]
                T = MST(D)
                aux = reconocer_caminos_robinson(T,D)
                PR_T, RE_T = aux[0], aux[3]
                output.append((n,(2*PR_T)/(n*(n-1))))
                output2.append((n,RE_T))
        # else:
        #     D = disimilitud(n)
        #     T = MST(D)
        #     aux = reconocer_caminos_robinson(T,D)
        #     PR_T, RE_T = aux[0], aux[3]
        #     output.append((n,(2*PR_T)/(n*(n-1))))
        #     output2.append((n,RE_T))

    return output, output2

##########################################################

def exp_random():
    output_PR = list()
    output_RE = list()
    output_PR_j = list()
    output_RE_j = list()
    N = [5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250,300,350]
    
    for i in range(5):
        print(f'Calculando experimento i = {i}')
        for n in N:
            print(f'{i}-esimo experimento para n={n}')
            D = disimilitud(n)
            T = MST(D)
            aux = reconocer_caminos_robinson(T,D)
            PR_T, RE_T = aux[0], aux[3]
            output_PR.append((n,(2*PR_T)/(n*(n-1))))
            output_RE.append((n,RE_T))
            for j in range(4):
                T_j = nx.random_labeled_tree(n)
                aux_j = reconocer_caminos_robinson(T_j,D)
                PR_T_j, RE_T_j = aux_j[0], aux_j[3]
                output_PR_j.append((n,(2*PR_T_j)/(n*(n-1))))
                output_RE_j.append((n,RE_T_j))


    return output_PR, output_RE, output_PR_j, output_RE_j

    return output_PR, output_RE

def exp_rob():
    output_PR = list()
    output_RE = list()
    output_PR_j = list()
    output_RE_j = list()
    N = [5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250,300,350]
    
    for i in range(5):
        print(f'Calculando experimento i = {i}')
        for n in N:
            print(f'{i}-esimo experimento para n={n}')
            D = disimilitud_Rob(n)
            T = MST(D)
            aux = reconocer_caminos_robinson(T,D)
            PR_T, RE_T = aux[0], aux[3]
            output_PR.append((n,(2*PR_T)/(n*(n-1))))
            output_RE.append((n,RE_T))
            for j in range(4):
                T_j = nx.random_labeled_tree(n)
                aux_j = reconocer_caminos_robinson(T_j,D)
                PR_T_j, RE_T_j = aux_j[0], aux_j[3]
                output_PR_j.append((n,(2*PR_T_j)/(n*(n-1))))
                output_RE_j.append((n,RE_T_j))


    return output_PR, output_RE, output_PR_j, output_RE_j

    return output_PR, output_RE

def exp_euc():
    output_PR = list()
    output_RE = list()
    output_PR_j = list()
    output_RE_j = list()
    N = [5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]
        #  ,150,200,250,300,350]
    
    for i in range(5):
        print(f'Calculando experimento i = {i}')
        for n in N:
            print(f'{i}-esimo experimento para n={n}')
            D = dis_euc(n)[0]
            T = MST(D)
            aux = reconocer_caminos_robinson(T,D)
            PR_T, RE_T = aux[0], aux[3]
            output_PR.append((n,(2*PR_T)/(n*(n-1))))
            output_RE.append((n,RE_T))
            for j in range(4):
                T_j = nx.random_labeled_tree(n)
                aux_j = reconocer_caminos_robinson(T_j,D)
                PR_T_j, RE_T_j = aux_j[0], aux_j[3]
                output_PR_j.append((n,(2*PR_T_j)/(n*(n-1))))
                output_RE_j.append((n,RE_T_j))


    return output_PR, output_RE, output_PR_j, output_RE_j

def exp_poincare():
    '''
    Se generan disimilitudes Poincaré y se calculan las métricas de sus MST. Las salidas son dos métricas con
    '''
    output_PR = list()
    output_RE = list()
    output_PR_j = list()
    output_RE_j = list()
    N = [5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250,300,350]
    
    for i in range(5):
        print(f'Calculando experimento i = {i}')
        for n in N:
            print(f'{i}-esimo experimento para n={n}')
            D = dis_poincare(n)[0]
            T = MST(D)
            aux = reconocer_caminos_robinson(T,D)
            PR_T, RE_T = aux[0], aux[3]
            output_PR.append((n,(2*PR_T)/(n*(n-1))))
            output_RE.append((n,RE_T))
            for j in range(4):
                T_j = nx.random_labeled_tree(n)
                aux_j = reconocer_caminos_robinson(T_j,D)
                PR_T_j, RE_T_j = aux_j[0], aux_j[3]
                output_PR_j.append((n,(2*PR_T_j)/(n*(n-1))))
                output_RE_j.append((n,RE_T_j))


    return output_PR, output_RE, output_PR_j, output_RE_j

# def max_PR_tree(D):
    # n = len(D)
    # all = n**(n-2)
# 
    # obs_sample = int(0.9*all)





def exp_MST_vs_T(n):
    '''
    Dado n, se genera un espacio de disimilitud junto a todos sus MST's. Hay dos salidas:
        D: disimilitud con la que se trabajó
        output que es una lista de listas, donde cada elemento está compuesto por una tupla (PR_T,w_T) y su árbol correspondiente.
    '''
    D = disimilitud(n)
    G = Kn(n,D)

    mst_list = all_minimum_spanning_trees(G)
    output = list()

    for T in mst_list:
        output.append([(reconocer_caminos_robinson(T,D)[0], w(T, D)), T])
        vecinos = generar_arboles_reconectando_hojas(T)
        for T_aux in vecinos:
            output.append([(reconocer_caminos_robinson(T_aux,D)[0], w(T, D)), T])
    
    
    return D,output

#################################################################################################

def comparacion_w_PR(D, n_experimentos):
    '''
    Dado un espacio de disimilitud (X,D), se calculan tantos árboles recubridores aleatorios como n_experimentos se entregue. La salida es una lista de n_experimentos listas, formadas por una tupla (w_T,PR_T) y el árbol T correspondiente.
    '''
    n = len(D)

    data = []
    k=0

    for i in range(n_experimentos):
        T_j = nx.random_labeled_tree(n)
        aux = reconocer_caminos_robinson(T_j,D)
        data.append([(w(T_j, D),aux[0]),T_j])
        # Lo siguiente es solo para ver estado de proceso para experimentos grandes
        # if (100*(i+1)/n_experimentos)>=10*k:
        #     print(f'comparacion_w_PR: {10*k}%')
        #     k+=10

    return data


def exp_noise(n=20, n_iter=50):
    # Rango de ruido: nos interesa ver qué pasa antes y después de 1.0
    # Vamos de 0 a 2.5 para ver la caída claramente
    niveles_ruido = np.linspace(0, 5, 40)
    
    promedios = []
    desviaciones = []
    
    for epsilon in niveles_ruido:
        scores = []
        for _ in range(n_iter):
            # 1. Generar Árbol Base (Pesos = 1.0 fijo)
            T_real = nx.random_tree(n)
            for u, v in T_real.edges():
                T_real[u][v]['weight'] = 1.0
                
            # 2. Matriz de Distancia Original (Enteros)
            # nodelist asegura el orden consistente de índices
            nodos = sorted(list(T_real.nodes()))
            D_base = nx.floyd_warshall_numpy(T_real, nodelist=nodos)
            
            # 3. Matriz de Ruido
            # Ruido simétrico Uniforme[0, epsilon]
            # Nota: Podrías usar normal, pero Uniforme[0,1] ilustra mejor tu punto exacto
            noise = np.random.rand(n, n) * epsilon
            noise = (noise + noise.T) / 2  # Hacerlo simétrico
            np.fill_diagonal(noise, 0)
            
            D_ruido = D_base + noise
            
            # 4. Calcular MST del ruidoso
            # Creamos grafo completo con pesos ruidosos
            K_ruido = nx.complete_graph(n)
            for i in range(n):
                for j in range(i + 1, n):
                    K_ruido[i][j]['weight'] = D_ruido[i, j]
            
            MST_ruido = nx.minimum_spanning_tree(K_ruido)
            
            # 5. Métrica: ¿Qué porcentaje de las aristas originales recuperamos?
            # Esta es la medida más directa de "Optimalidad estructural"
            aristas_real = set(frozenset(e) for e in T_real.edges())
            aristas_mst = set(frozenset(e) for e in MST_ruido.edges())
            
            coincidencia = len(aristas_real.intersection(aristas_mst)) / len(aristas_real)
            scores.append(coincidencia)
            
        promedios.append(np.mean(scores))
        desviaciones.append(np.std(scores))
    
    return niveles_ruido, np.array(promedios), np.array(desviaciones)


def exp_neighbor_leaf_iter_tracking(t, T_max_prs_lot, lista_matrices):
    """
    Realiza búsqueda local y guarda la secuencia de valores aceptados.
    
    Returns:
        tuple: (arboles_optimizados, historiales_valores)
        Donde historiales_valores es una lista de listas: [[v0, v1, v2...], ...]
    """
    optimized_trees = []
    val_histories = [] # Aquí guardaremos solo los valores en orden
    
    print(f"--- Iniciando Exploración por Iteraciones (t={t}s por árbol) ---")

    for i, arbol_inicial in enumerate(T_max_prs_lot):
        if arbol_inicial is None:
            optimized_trees.append(None)
            val_histories.append([])
            continue
            
        arbol = arbol_inicial.copy() 
        D = lista_matrices[i]
        n = len(D)
        
        # Valor inicial
        raw_pr = reconocer_caminos_robinson(arbol, D)[0]
        pr_ref = 2 * raw_pr / (n * (n - 1))
        
        # Historial: Comenzamos con el valor inicial (iteración 0)
        current_history = [pr_ref]
        
        start_time = time.time()
        mejora_encontrada = True 
        
        print(f"\nProcesando ID {i+1}...")

        while (time.time() - start_time < t) and mejora_encontrada:
            mejora_encontrada = False 
            hojas = [nodo for nodo, grado in arbol.degree() if grado == 1]
            random.shuffle(hojas)
            
            for hoja in hojas:
                vecino_actual = list(arbol.neighbors(hoja))[0]
                arbol.remove_edge(hoja, vecino_actual)
                
                posibles = [nodo for nodo in arbol.nodes() if nodo != hoja and nodo != vecino_actual]
                random.shuffle(posibles)
                
                for nuevo_vecino in posibles:
                    arbol.add_edge(hoja, nuevo_vecino)
                    
                    nuevo_raw = reconocer_caminos_robinson(arbol, D)[0]
                    nuevo_pr = 2 * nuevo_raw / (n * (n - 1))
                    
                    if nuevo_pr >= pr_ref:
                        # Si encontramos una mejora (o movimiento lateral), la registramos
                        if nuevo_pr > pr_ref:
                            current_history.append(nuevo_pr)
                            print(f"   [Paso {len(current_history)-1}] Mejora: {pr_ref:.4f} -> {nuevo_pr:.4f}")
                        
                        pr_ref = nuevo_pr
                        mejora_encontrada = True
                        break 
                    else:
                        arbol.remove_edge(hoja, nuevo_vecino)
                
                if mejora_encontrada:
                    break
                else:
                    arbol.add_edge(hoja, vecino_actual)
        
        optimized_trees.append(arbol)
        val_histories.append(current_history)

    return optimized_trees, val_histories

def PR(T,D):
    return reconocer_caminos_robinson(T,D)[0]

def RE(T,D):
    return reconocer_caminos_robinson(T,D)[3]


#################################################################################################

# def metricas(d, T):
#     """
#     Dada una matriz de disimilitud d:[[n]]^2 -> [[n]] y T un árbol recubridor de [[n]], la función 'metricas' devolverá los valores de PR y RE en forma de lista, junto al número de hojas, así como todos los extremos de los caminos que son Robinson dentro de T. 
#     """
#     # Contador para caminos que cumplen las condiciones
#     PR = 0
#     RE = 0
#     # Almacenamps los extremos de los caminos válidos
#     extremos_caminos_validos = []
#     # Iteramos sobre todos los pares de vértices
#     for x in T.nodes:
#         for y in T.nodes:
#             if x != y:
#                 # Obtenemos todos los caminos simples entre x e y en el árbol
#                 for camino in nx.all_simple_paths(T, source=x, target=y):
#                     cumple_condicion = True
#                     # Verificamos condición para cada punto intermedio z en el camino para la métrica RE
#                     for k in range(1, len(camino) - 1):
#                         z = camino[k]
#                         if not (d[x, y] > d[x, z] and d[x, y] > d[z, y]):
#                             cumple_condicion = False
#                             break
#                     # Si el camino cumple la condición, aumentamos en uno el valor de las métricas. Como se revisa todo el árbol, al final solo nos quedamos con la mitad del valor encontrado, pues cada camino y arista es contada dos veces.
#                     if cumple_condicion:
#                         PR += 1
#                         RE += len(camino)
#                         extremos_caminos_validos.append((x, y))

#     num_hojas = contar_hojas(T)

#     # output: PR, RE, cantidad de hojas del árbol, y una lista con  los extremos de los caminos Robinson del árbol
#     return PR/2, RE/2, num_hojas, extremos_caminos_validos

#################################################################################################

def matriz_incidencia_arbol(n):
    # Generar un árbol aleatorio con n nodos
    arbol = nx.random_labeled_tree(n)
    
    # Obtener la matriz de incidencia del árbol
    nx.draw(arbol, with_labels=True, node_color='lightblue', font_weight='bold')    
    matriz_incidencia = nx.incidence_matrix(arbol, oriented=False).toarray()
    

    return matriz_incidencia

#################################################################################################

def cluster_hyp(n,d):
    """
    Función que crea la matriz de incidencia del hipergrafo de clúster del espacio de disimilitud ([[n]],d). Como output adicional, se tienen los cliques de todos los grafos límite, los que representan clústers del hipergrafo.
    """
    cliques = list()
    for alpha in range(n):
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                if d[i,j]<=alpha:
                    G.add_edges_from([(i,j)])
 
        if len(G)!=0:cliques.append(list(nx.find_cliques(G)))

    hedges = list()

    for clusters in cliques:
        for hedge in clusters:
            hedges.append(np.sort(hedge))
    
    return reduc_col(hyp(n,hedges)), cliques

#################################################################################################

def ball_hyp(n,d):
    """
    Función que crea la matriz de incidencia del hipergrafo de bolas del espacio de disimilitud ([[n]],d), así como las bolas del espacio.
    """
    balls = list()
    for alpha in range(n):
        ball = list()
        for i in range(n):
            aux = [i]
            for j in range(n):
                if d[i,j]<=alpha and i!=j:
                    aux.append(j)
            if len(aux)>1 : ball.append(aux)
        balls.append(ball)
    
    hedges = list()

    for b in balls:
        for hedge in b:
            hedges.append(np.sort(hedge))
    
    return reduc_col(hyp(n,hedges)), balls

#################################################################################################

def ball2_hyp(n,d):
    """
    Función que crea la matriz de incidencia del hipergrafo de 2-bolas del espacio de disimilitud ([[n]],d) y las 2-bolas de éste.
    """
    b2s = list()
    for i in range(n):
        for j in range(i+1,n):
            if i!=j:
                b2= [i,j]
                for k in range(n):
                    if (i!=k) and (j!=k):
                        M = max(d[i,k],d[j,k])
                        if (M<=d[i,j]):
                            b2.append(k)
                    if len(b2)>1: b2s.append(b2)
    
    return reduc_col(hyp(n,b2s)), b2s
 
#################################################################################################

def reduc_col(M):
    """
    Dada una matriz M (usando numpy), elimina las columnas duplicadas,
    conservando solo la primera ocurrencia de cada columna.
    
    Parámetros:
      - M: np.ndarray de dimensión (n, m) donde n es el número de filas y m el número de columnas.
    
    Retorna:
      - Una matriz de dimensión (n, k) con las columnas únicas (en el orden de aparición original),
        donde k es el número de columnas únicas.

    En contexto de hipergrafos, reduc_col significa eliminar hiperarisas repetidas
    """
    # Lista para guardar los índices de las columnas únicas
    idx_unicas = []
    # Conjunto para almacenar las columnas ya vistas (convertidas a tupla)
    vistas = set()
    
    # Recorremos las columnas de la matriz
    for i in range(M.shape[1]):
        # Convertimos la columna a una tupla (para poder compararla y usarla en el conjunto)
        col = tuple(M[:, i])
        # Si la columna no se ha visto antes, se añade a la lista y al conjunto
        if col not in vistas:
            vistas.add(col)
            idx_unicas.append(i)
            
    # Retornamos la matriz con solo las columnas únicas, en el orden original
    return M[:, idx_unicas]

#################################################################################################TRABAJO CON SERGIO#################################################################################################################################################################################################3

def generador_de_instancias(N,S):
    '''
    N es una lista de enteros que dirá la cantidad de elementos que tendrá cada espacio
    Ex: N=[8,12,15,20]
    S es una 3-tupla que dirá la cantidad de disimilitudes euclidianas, de poincaré, y aleatorias generar, respectivamente.
    '''
    dis_list = []

    for n in N:
        for _ in range(S[0]):
            D = dis_euc(n)[0]
            dis_list.append((D,'Euclidiano'))


        for _ in range(S[1]):
            D = dis_poincare(n)[0]
            dis_list.append((D,'de Poincaré'))
            

        for _ in range(S[2]):
            D = disimilitud(n)
            dis_list.append((D,'Aleatorio'))


    return dis_list


def save_benchmark(datos_acumulados):
    """
    Crea una carpeta con timestamp y guarda las matrices junto con su etiqueta.
    
    Args:
        datos_acumulados (list): Lista de tuplas (matriz, string).
                                 Estructura esperada: [(np.array, "Texto descriptivo"), ...]
    
    Returns:
        str: El nombre de la carpeta creada.
    """
    # 1. Obtener fecha y hora actual para el nombre
    now = datetime.now()
    timestamp = now.strftime("%d%m%y_%H%M") 
    nombre_carpeta = f"benchmark_{timestamp}"
    
    # 2. Crear la carpeta si no existe
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)
        print(f"Carpeta creada: {nombre_carpeta}")
    
    # 3. Guardar cada tupla
    for i, dato in enumerate(datos_acumulados):
        ID = i + 1
        nombre_archivo = os.path.join(nombre_carpeta, f"{ID}.txt")

        # Validación básica para asegurar que es la tupla esperada
        if isinstance(dato, (tuple, list)) and len(dato) == 2:
            matriz = dato[0]
            texto_header = dato[1]
        else:
            print(f"Advertencia: El dato en índice {ID} no tiene el formato (matriz, string). Se omite.")
            continue

        try:
            # Abrimos el archivo en modo escritura ('w')
            with open(nombre_archivo, 'w', encoding='utf-8') as f:
                # 1. Escribimos el string (agregamos salto de línea \n)
                f.write(f"{texto_header}\n")
                
                # 2. Escribimos la matriz en el mismo archivo abierto
                # fmt='%.6f' para decimales, o fmt='%d' si son enteros puros
                np.savetxt(f, matriz, fmt='%.6f')
                
        except Exception as e:
            print(f"Error al guardar el archivo {ID}.txt: {e}")

    print(f"Se han guardado {len(datos_acumulados)} archivos en '{nombre_carpeta}'.")
    return nombre_carpeta

def load_benchmark(nombre_carpeta, ID):
    """
    Lee una matriz específica desde la carpeta del benchmark dada su ID.
    
    Args:
        nombre_carpeta (str): El nombre de la carpeta (ej: 'benchmark_250126_1400')
        ID (int): El número identificador de la matriz (1, 2, ...).
        
    Returns:
        np.array: La matriz recuperada como array de numpy.
    """
    nombre_archivo = os.path.join(nombre_carpeta, f"{ID}.txt")
    
    try:
        matriz = np.loadtxt(nombre_archivo)
        return matriz
    except FileNotFoundError:
        print(f"Error: El archivo {nombre_archivo} no existe.")
        return None
    except Exception as e:
        print(f"Error al leer la matriz: {e}")
        return None
    

def procesar_benchmark_a_df(nombre_carpeta, t=5):
    """
    Lee archivos .txt de una carpeta (formato: header + matriz),
    calcula métricas usando la librería 'qr' y retorna una lista de matrices
    y un DataFrame con los resultados.

    Args:
        nombre_carpeta (str): Ruta a la carpeta con los archivos .txt
        t (int): Tiempo en segundos para la búsqueda de MSTs distintos (default 5).

    Returns:
        tuple: (lista_matrices, df_resultados)
    """
    lista_matrices = []
    data = []

    # 1. Obtener y ordenar archivos numéricamente (1.txt, 2.txt...)
    try:
        archivos = [f for f in os.listdir(nombre_carpeta) if f.endswith('.txt')]
        # Ordenamos por el número en el nombre del archivo
        archivos.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except FileNotFoundError:
        print(f"Error: La carpeta '{nombre_carpeta}' no existe.")
        return [], pd.DataFrame()

    print(f"Procesando {len(archivos)} archivos en '{nombre_carpeta}'...")

    for archivo in archivos:
        ruta_completa = os.path.join(nombre_carpeta, archivo)
        
        try:
            # 2. Leer Header (Tipo) y Matriz
            with open(ruta_completa, 'r', encoding='utf-8') as f:
                tipo = f.readline().strip()  # Primera línea es el tipo
                # Numpy lee desde la posición actual del cursor (después del header)
                D = np.loadtxt(f)

            lista_matrices.append(D)
            n = D.shape[0]

            # 3. Cálculos con la librería 'qr'
            # Cálculo del MST base y sus métricas
            MST_j = MST(D)
            w_MST = w(MST_j, D)
            PR_D = reconocer_caminos_robinson(MST_j, D)[0]
            PRS_D = 2 * PR_D / (n * (n - 1)) if n > 1 else 0

            # Verificación de Unicidad
            unico = es_mst_unico(D)
            
            mst_count_str = 'Único'
            std_prs = 0.0

            if not unico:
                # Si no es único, buscamos distintos en tiempo 't'
                # C se espera que sea (cantidad, lista_de_arboles)
                C = buscar_mst_distintos(D, t)
                mst_count_str = C[0]
                
                # Calculamos el STD del PR normalizado para los árboles encontrados
                # Nota: Recalculamos PR para cada árbol 'x' en C[1]
                lista_prs_variantes = []
                for x in C[1]:
                    # Asumiendo que x es un árbol en formato compatible con reconocer_caminos_robinson
                    pr_val = reconocer_caminos_robinson(x, D)[0]
                    prs_val = 2 * pr_val / (n * (n - 1))
                    lista_prs_variantes.append(prs_val)
                
                std_prs = np.std(lista_prs_variantes)

            # 4. Agregar al diccionario de datos
            data.append({
                'n': int(n),
                'Tipo': tipo,
                f'MST en {t}s': mst_count_str,
                'peso MST': w_MST,
                '$PR_D(MST)$': PR_D,
                'max_valor': PR_D, # Según tu requerimiento
                '$PR^S_D(MST)$': PRS_D,
                '$STD PR^S_D(MST)$': std_prs
            })

        except Exception as e:
            print(f"Error procesando archivo {archivo}: {e}")

    # 5. Crear DataFrame
    df_resultados = pd.DataFrame(data)
    
    return lista_matrices, df_resultados


# def fase_constructiva_1(matrices, tiempo):
#     """
#     Ejecuta la Fase 1 (Constructiva) buscando árboles recubridores de peso mínimo 
#     (MST) distintos durante un tiempo determinado.
    
#     Args:
#         matrices (list): Lista de matrices de disimilitud (numpy arrays).
#         tiempo (int/float): Tiempo en segundos permitido para buscar MSTs en cada matriz.
        
#     Returns:
#         tuple: (historial_fase, T_max_prs)
#             - historial_fase: Lista de listas, donde cada sublista contiene la evolución 
#                               del mejor PR encontrado hasta el momento para una matriz.
#             - T_max_prs: Lista con el objeto 'árbol' (MST) que obtuvo el mejor PR para cada matriz.
#     """
#     historial_fase = []
#     T_max_prs = []
    
#     print(f"--- Iniciando Fase 1 Constructiva (Tiempo de búsqueda por matriz: {tiempo}s) ---")

#     for i, D in enumerate(matrices):
#         n = len(D)
        
#         # Ejecución de búsqueda de MSTs distintos
#         # Asumimos que qr.buscar_mst_distintos devuelve una tupla donde el índice [1] 
#         # contiene el conjunto/lista de árboles encontrados.
#         L = buscar_mst_distintos(D, tiempo)
#         trees_found = list(L[1])
        
#         print(f"Procesando Matriz {i+1}/{len(matrices)} (n={n}) -> {len(trees_found)} árboles encontrados.")
        
#         # Inicializamos con el valor "cero" para mostrar el estado inicial antes del primer árbol
#         prs = [0]
#         prs_max = [0]
#         M = 0
        
#         # Evaluación de los árboles encontrados
#         for arbol in trees_found:
#             # Reconocer caminos Robinson y calcular PR
#             pr_val = reconocer_caminos_robinson(arbol, D)[0]
#             current_pr_norm = 2 * pr_val / (n * (n - 1))
            
#             prs.append(current_pr_norm)
            
#             # Lógica para mantener la curva "escalonada" (el mejor hasta el momento)
#             if current_pr_norm >= M:
#                 M = current_pr_norm
                
#             prs_max.append(M)

#         # Guardamos la historia completa de esta matriz
#         historial_fase.append(prs_max)
        
#         # Lógica para extraer y guardar el mejor árbol definitivo
#         if len(prs) > 1: # Chequeamos > 1 porque el índice 0 es el '0' artificial que pusimos
#             max_val = max(prs)
#             # Restamos 1 porque 'prs' tiene un elemento extra al inicio ([0])
#             # respecto a la lista 'trees_found'
#             max_idx = prs.index(max_val) - 1 
#             best_tree = trees_found[max_idx]
            
#             T_max_prs.append(best_tree)
#             print(f"   -> Mejor PR: {max_val:.4f} (Árbol #{max_idx + 1} de {len(trees_found)})")
#         else:
#             print("   -> No se encontraron árboles en el tiempo asignado.")
#             T_max_prs.append(None)
            
#     print("--- Fase 1 Finalizada ---\n")
#     return historial_fase, T_max_prs

def fase_constructiva_1(matrices, tiempo):
    """
    Ejecuta la Fase 1 (Constructiva) buscando árboles recubridores de peso mínimo 
    (MST) distintos durante un tiempo determinado.
    
    Args:
        matrices (list): Lista de matrices de disimilitud (numpy arrays).
        tiempo (int/float): Tiempo en segundos permitido para buscar MSTs en cada matriz.
        
    Returns:
        tuple: (historial_fase, T_all)
            - historial_fase: Lista de listas, donde cada sublista contiene la evolución 
                              del mejor PR encontrado hasta el momento para una matriz.
            - T_all: Lista de listas, donde cada sublista contiene TODOS los árboles
                     encontrados para esa matriz, ordenados de menor a mayor PR.
    """
    historial_fase = []
    T_all = [] # <--- NUEVO: Lista para todos los árboles
    
    print(f"--- Iniciando Fase 1 Constructiva (Tiempo de búsqueda por matriz: {tiempo}s) ---")

    for i, D in enumerate(matrices):
        n = len(D)
        
        # Ejecución de búsqueda de MSTs distintos
        L = buscar_mst_distintos(D, tiempo)
        trees_found = list(L[1])
        
        print(f"Procesando Matriz {i+1}/{len(matrices)} (n={n}) -> {len(trees_found)} árboles encontrados.")
        
        prs_max = [0]
        M = 0
        
        # NUEVO: Lista temporal para guardar tuplas de (árbol, métrica_pr)
        arboles_con_metricas = []
        
        # Evaluación de los árboles encontrados
        for arbol in trees_found:
            # Reconocer caminos Robinson y calcular PR
            pr_val = reconocer_caminos_robinson(arbol, D)[0]
            current_pr_norm = 2 * pr_val / (n * (n - 1))
            
            # Guardamos el árbol y su métrica juntos
            arboles_con_metricas.append((arbol, current_pr_norm))
            
            # Lógica para mantener la curva "escalonada" (el mejor hasta el momento)
            if current_pr_norm >= M:
                M = current_pr_norm
                
            prs_max.append(M)

        # Guardamos la historia completa de esta matriz
        historial_fase.append(prs_max)
        
        # --- NUEVA LÓGICA: Ordenar y extraer todos los árboles ---
        if arboles_con_metricas:
            # Ordenamos la lista de tuplas basándonos en el valor PR (índice 1), de menor a mayor
            arboles_con_metricas.sort(key=lambda x: x[1])
            
            # Extraemos solo los objetos 'árbol' ya ordenados
            arboles_ordenados = [item[0] for item in arboles_con_metricas]
            
            T_all.append(arboles_ordenados)
            
            # Para el print, mostramos el rango de PR encontrado (peor y mejor)
            peor_pr = arboles_con_metricas[0][1]
            mejor_pr = arboles_con_metricas[-1][1]
            print(f"   -> Árboles guardados y ordenados. PR Mínimo: {peor_pr:.4f} | PR Máximo: {mejor_pr:.4f}")
        else:
            # Caso borde por si no encuentra nada en los 'tiempo' segundos
            print("   -> No se encontraron árboles en el tiempo asignado.")
            T_all.append([]) 
            
    print("--- Fase 1 Finalizada ---\n")
    return historial_fase, T_all


def ConstructionPhase(D, t_mst, n_pert, epsilon, distribucion='normal'):
    """
    Combina la búsqueda de MSTs base y la aplicación de ruido.
    Devuelve un pool de árboles perturbados ordenados de mejor a peor PR.
    """
    n = len(D)
    
    # 1. Búsqueda de MSTs base
    L = buscar_mst_distintos(D, t_mst)
    trees_found = list(L[1])
    
    # Fallback por si la función no encuentra nada
    if not trees_found:
        trees_found = [MST(D)]
        
    arboles_perturbados_con_metricas = []
    
    # 2. Aplicar n_pert iteraciones de ruido a CADA árbol base encontrado
    for arbol_base in trees_found:
        val_base_original = reconocer_caminos_robinson(arbol_base, D)[0]
        mejor_pr_ruido = 2 * val_base_original / (n * (n - 1))
        mejor_arbol_perturbado = arbol_base.copy()
        
        for j in range(n_pert):
            if distribucion == 'normal':
                R = np.random.normal(0, 1, size=(n, n))
            else:
                R = np.random.uniform(-1, 1, size=(n, n))
                
            R = (R + R.T) / 2.0
            np.fill_diagonal(R, 0)
            
            D_noise = np.abs(D + epsilon * R)
            arbol_candidato = MST(D_noise) 
            
            val_candidato = reconocer_caminos_robinson(arbol_candidato, D)[0]
            pr_candidato = 2 * val_candidato / (n * (n - 1))
            
            if pr_candidato > mejor_pr_ruido:
                mejor_pr_ruido = pr_candidato
                mejor_arbol_perturbado = arbol_candidato.copy()
                
        # Guardamos el mejor representante perturbado de este árbol base
        arboles_perturbados_con_metricas.append((mejor_arbol_perturbado, mejor_pr_ruido))
        
    # 3. Ordenar de mayor a menor PR (para que T_pool[0] sea el mejor)
    arboles_perturbados_con_metricas.sort(key=lambda x: x[1], reverse=True)
    
    T_pool = [item[0] for item in arboles_perturbados_con_metricas]
    metricas_pool = [item[1] for item in arboles_perturbados_con_metricas]
    
    return T_pool, metricas_pool



def optimizacion_ruido_unificada(matrices_filtradas, historial_fase1, arboles_fase1, total_iteraciones, lista_ids, categoria="general", e=0.1, distribucion='normal'):
    """
    Función maestra para ejecutar la Fase 2 (Ruido Dinámico) aplicable a 
    las categorías 'lot', 'few' y 'unique'. Actualiza la lista de árboles ganadores 
    y genera registros de profiling.
    
    Args:
        matrices_filtradas (list): Las matrices D originales correspondientes al grupo.
        historial_fase1 (list): Lista de listas con los historiales de PR de la Fase 1.
        arboles_fase1 (list): Lista de los mejores árboles (MST) encontrados en la Fase 1.
        total_iteraciones (int): Límite superior de iteraciones conjuntas (Fase 1 + Fase 2).
        lista_ids (list): IDs originales de las matrices.
        categoria (str): 'LOT', 'FEW' o 'UNIQUE' (usado para nombrar archivos y gráficas).
        e (float): Intensidad de la perturbación (ruido).
        distribucion (str): 'normal' o 'uniforme'.
        
    Returns:
        tuple: (historiales_fase2, arboles_actualizados, df_hitos, df_mejoras)
    """
    
    historiales_fase2 = []
    
    # Creamos una copia de la lista de árboles para no sobreescribir la variable original de forma insegura
    arboles_actualizados = arboles_fase1.copy() 
    
    # Registros para Profiling
    registros_hitos = []
    registros_mejoras = []
    
    print(f"\n--- Iniciando Fase 2 (Ruido Dinámico) | Categoría: {categoria.upper()} | Meta: {total_iteraciones} iters ---")

    for i, D in enumerate(matrices_filtradas):
        n = len(D)
        id_real = lista_ids[i]
        
        # 1. Rescatar valores base
        max_pr_so_far = historial_fase1[i][-1]
        iteraciones_fase1 = len(historial_fase1[i]) - 1
        iteraciones_faltantes = total_iteraciones - iteraciones_fase1
        
        historial_actual = [max_pr_so_far]
        start_time = time.time()
        
        print(f"Matriz ID {id_real} (n={n}): Fase 1 hizo {iteraciones_fase1} iters. Faltan {iteraciones_faltantes} iters (Base PR: {max_pr_so_far:.4f})")
        
        if iteraciones_faltantes > 0:
            for j in range(iteraciones_faltantes):
                iteracion_global = iteraciones_fase1 + j + 1
                
                # --- Lógica de Perturbación ---
                if distribucion == 'normal':
                    R = np.random.normal(0, 1, size=(n, n))
                elif distribucion == 'uniforme':
                    R = np.random.uniform(-1, 1, size=(n, n))
                
                R = (R + R.T) / 2.0
                np.fill_diagonal(R, 0)
                
                D_noise = np.abs(D + e * R)
                MST_D_noise = MST(D_noise)
                
                val = reconocer_caminos_robinson(MST_D_noise, D)[0]
                pr_val = 2 * val / (n * (n - 1))
                
                tiempo_actual = time.time() - start_time
                
                # --- Evaluación de Mejora ---
                if pr_val > max_pr_so_far:
                    # Guardamos datos para profiling
                    registros_mejoras.append({
                        'Categoría': categoria.upper(), 'ID Matriz': id_real, 'Dimensión (n)': n,
                        'Iteracion Global': iteracion_global, 'Tiempo (segundos)': round(tiempo_actual, 4),
                        'PR Anterior': max_pr_so_far, 'PR Nuevo': pr_val, 'Magnitud Mejora': pr_val - max_pr_so_far
                    })
                    
                    # Actualizamos récord y el árbol ganador en nuestra lista de retorno
                    max_pr_so_far = pr_val
                    arboles_actualizados[i] = MST_D_noise.copy() 
                    
                historial_actual.append(max_pr_so_far)
                
                # --- Profiling: Hitos cada 100 iteraciones ---
                if iteracion_global % 100 == 0:
                    registros_hitos.append({
                        'Categoría': categoria.upper(), 'ID Matriz': id_real, 'Dimensión (n)': n,
                        'Iteracion Global': iteracion_global, 'Tiempo Acumulado (s)': round(tiempo_actual, 4),
                        'Mejor PR Actual': max_pr_so_far
                    })
        
        # Si la Fase 1 agotó el límite de iteraciones, simplemente rellenamos
        elif iteraciones_faltantes <= 0:
             print(f"  -> La matriz ID {id_real} ya superó o igualó el límite en la Fase 1. No se aplica ruido.")
             
        historiales_fase2.append(historial_actual)

    # =========================================================================
    # Creación de DataFrames y Exportación a Excel
    # =========================================================================
    df_hitos = pd.DataFrame(registros_hitos)
    df_mejoras = pd.DataFrame(registros_mejoras)
    
    timestamp = datetime.now().strftime('%d%m%y_%H%M')
    nombre_archivo = f'Profiling_Fase2_Ruido_{categoria.upper()}_{timestamp}.xlsx'
    
    with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
        if not df_hitos.empty:
            df_hitos.sort_values(by=['Dimensión (n)', 'ID Matriz', 'Iteracion Global']).to_excel(writer, sheet_name='Hitos_cada_100', index=False)
        if not df_mejoras.empty:
            df_mejoras.sort_values(by=['Dimensión (n)', 'ID Matriz', 'Iteracion Global']).to_excel(writer, sheet_name='Mejoras', index=False)
            
    print(f"\n-> Fase 2 Finalizada. Profiling exportado a: {nombre_archivo}")

    return historiales_fase2, arboles_actualizados, df_hitos, df_mejoras


# Generar lista inicial de movimientos posibles (hoja -> nuevo_vecino)
def obtener_movimientos(arbol_actual):
    movs = []
    hojas = [nodo for nodo, grado in arbol_actual.degree() if grado == 1]
    for hoja in hojas:
        vecino_actual = list(arbol_actual.neighbors(hoja))[0]
        posibles = [nodo for nodo in arbol_actual.nodes() if nodo != hoja and nodo != vecino_actual]
        for p in posibles:
            movs.append((hoja, vecino_actual, p))
    random.shuffle(movs)
    return movs


# def busqueda_local_FI_tiempo(matrices, arboles_base, historiales_fase2, lista_ids, t_max, x_limite, j_limite, update_tree=True, categoria="FI"):
#     historiales_fase3 = []
#     arboles_fase3 = [] # NUEVO: Lista para guardar los árboles finales
#     registros_mejoras = []
    
#     print(f"\n--- Fase 3: Local Search FIRST IMPROVEMENT [{categoria}] ---")
#     print(f"Parámetros: t_max={t_max}s | Tolerancia x={x_limite} | Estancamiento j={j_limite} | Actualizar Árbol={update_tree}")

#     for i, D in enumerate(matrices):
#         n = len(D)
#         id_real = lista_ids[i]
        
#         max_pr_so_far = historiales_fase2[i][-1]
#         historial_actual = [max_pr_so_far]
        
#         arbol = arboles_base[i].copy() if arboles_base[i] is not None else None
        
#         if arbol is None:
#             historiales_fase3.append(historial_actual)
#             arboles_fase3.append(None) # Añadimos None si no hay árbol
#             continue
            
#         print(f"Procesando ID {id_real}... PR Base: {max_pr_so_far:.4f}")
        
#         start_time = time.time()
#         sin_mejora = 0  # Contador 'x' de iteraciones sin mejora
#         estancamiento_j = 0
#         iteracion_local = 0
#         hubo_mejora_estricta = False # NUEVO: Bandera para saber si mejoró alguna vez
        
        

#         movimientos = obtener_movimientos(arbol)
#         idx_mov = 0
        
#         # Bucle principal gobernado por TIEMPO y LÍMITES X, J
#         while (time.time() - start_time) < t_max and sin_mejora < x_limite and estancamiento_j < j_limite:
            
#             # Si se nos acaban los movimientos de este árbol, los regeneramos
#             if idx_mov >= len(movimientos):
#                 movimientos = obtener_movimientos(arbol)
#                 idx_mov = 0
                
#             hoja, v_viejo, v_nuevo = movimientos[idx_mov]
#             idx_mov += 1
#             iteracion_local += 1
#             estancamiento_j += 1
            
#             # Aplicar movimiento
#             arbol.remove_edge(hoja, v_viejo)
#             arbol.add_edge(hoja, v_nuevo)
            
#             # Evaluar
#             val = reconocer_caminos_robinson(arbol, D)[0]
#             nuevo_pr = 2 * val / (n * (n - 1))
#             tiempo_transcurrido = time.time() - start_time
            
#             # Criterio de aceptación (>= para permitir movimientos laterales en mesetas)
#             if nuevo_pr >= max_pr_so_far:
#                 if nuevo_pr > max_pr_so_far:
#                     registros_mejoras.append({
#                         'Categoría': categoria, 'ID Matriz': id_real, 'Dimensión (n)': n,
#                         'Iteración Local': iteracion_local, 'Tiempo (s)': round(tiempo_transcurrido, 4),
#                         'PR Anterior': max_pr_so_far, 'PR Nuevo': nuevo_pr, 'Mejora': nuevo_pr - max_pr_so_far
#                     })
#                     max_pr_so_far = nuevo_pr
#                     estancamiento_j = 0
#                     hubo_mejora_estricta = True # NUEVO: Marcamos que la matriz sí logró mejorar
                
#                 sin_mejora = 0 # Reseteamos el contador 'x' porque hallamos algo bueno (>=)
#                 historial_actual.append(max_pr_so_far)
                
#                 # Control del árbol inmutable
#                 if not update_tree:
#                     # Deshacemos el cambio para que el árbol base NO cambie
#                     arbol.remove_edge(hoja, v_nuevo)
#                     arbol.add_edge(hoja, v_viejo)
#                 else:
#                     # Si actualizamos, debemos recalcular los vecindarios para la nueva topología
#                     movimientos = obtener_movimientos(arbol)
#                     idx_mov = 0
#             else:
#                 # Si no mejoró, deshacemos obligatoriamente y sumamos al contador
#                 arbol.remove_edge(hoja, v_nuevo)
#                 arbol.add_edge(hoja, v_viejo)
#                 sin_mejora += 1
#                 historial_actual.append(max_pr_so_far)

#         # NUEVO: Si terminó el while y nunca hubo mejora estricta, registramos el tiempo total
#         tiempo_total_bucle = time.time() - start_time
#         if not hubo_mejora_estricta:
#             registros_mejoras.append({
#                 'Categoría': categoria, 'ID Matriz': id_real, 'Dimensión (n)': n,
#                 'Iteración Local': iteracion_local, 'Tiempo (s)': round(tiempo_total_bucle, 4),
#                 'PR Anterior': max_pr_so_far, 'PR Nuevo': max_pr_so_far, 'Mejora': 0.0
#             })

#         # NUEVO: Guardar el árbol final resultante para esta matriz
#         arboles_fase3.append(arbol.copy())

#         # Determinar motivo de parada para el print (Bloque limpio)
#         if tiempo_total_bucle >= t_max:
#             motivo = "Tiempo excedido"
#         elif sin_mejora >= x_limite:
#             motivo = f"Límite {x_limite} evals fallidas"
#         else:
#             motivo = f"Estancamiento: {j_limite} iters sin mejora estricta"
            
#         print(f"  -> Detenido por: {motivo}. Mejor PR: {max_pr_so_far:.4f}")
#         historiales_fase3.append(historial_actual)

#     # Exportación
#     df_mejoras = pd.DataFrame(registros_mejoras)
#     timestamp = datetime.now().strftime('%d%m%y_%H%M')
#     if not df_mejoras.empty:
#         df_mejoras.to_excel(f'Mejoras_Fase3_FI_{timestamp}.xlsx', index=False)

#     # RETORNO ACTUALIZADO
#     return historiales_fase3, arboles_fase3, df_mejoras



# def busqueda_local_FI_estricta(matrices, arboles_base, historiales_fase2, lista_ids, t_max, update_tree=True, categoria="FI-Estricto"):
#     historiales_fase3 = []
#     arboles_fase3 = [] 
#     registros_mejoras = []
    
#     print(f"\n--- Fase 3: Local Search FIRST IMPROVEMENT [{categoria}] ---")
#     print(f"Parámetros: t_max={t_max}s | Solo Mejoras Estrictas | Actualizar Árbol={update_tree}")

#     for i, D in enumerate(matrices):
#         n = len(D)
#         id_real = lista_ids[i]
        
#         max_pr_so_far = historiales_fase2[i][-1]
#         historial_actual = [max_pr_so_far]
        
#         arbol = arboles_base[i].copy() if arboles_base[i] is not None else None
        
#         if arbol is None:
#             historiales_fase3.append(historial_actual)
#             arboles_fase3.append(None)
#             continue
            
#         print(f"Procesando ID {id_real}... PR Base: {max_pr_so_far:.4f}")
        
#         start_time = time.time()
#         iteracion_local = 0
#         hubo_mejora_estricta_global = False 
#         motivo = ""
        
#         movimientos = obtener_movimientos(arbol)
        
#         # Bucle principal gobernado por TIEMPO
#         while (time.time() - start_time) < t_max:
#             hubo_mejora_en_ronda = False
            
#             for mov in movimientos:
#                 # Chequeo de tiempo en medio de la evaluación del vecindario
#                 tiempo_transcurrido = time.time() - start_time
#                 if tiempo_transcurrido >= t_max:
#                     break
                    
#                 hoja, v_viejo, v_nuevo = mov
#                 iteracion_local += 1
                
#                 # Aplicar movimiento
#                 arbol.remove_edge(hoja, v_viejo)
#                 arbol.add_edge(hoja, v_nuevo)
                
#                 # Evaluar
#                 val = reconocer_caminos_robinson(arbol, D)[0]
#                 nuevo_pr = 2 * val / (n * (n - 1))
                
#                 # CRITERIO ESTRICTO: Solo aceptamos si es MAYOR al récord
#                 if nuevo_pr > max_pr_so_far:
#                     registros_mejoras.append({
#                         'Categoría': categoria, 'ID Matriz': id_real, 'Dimensión (n)': n,
#                         'Iteración Local': iteracion_local, 'Tiempo (s)': round(tiempo_transcurrido, 4),
#                         'PR Anterior': max_pr_so_far, 'PR Nuevo': nuevo_pr, 'Mejora': nuevo_pr - max_pr_so_far
#                     })
                    
#                     max_pr_so_far = nuevo_pr
#                     hubo_mejora_estricta_global = True
#                     hubo_mejora_en_ronda = True
#                     historial_actual.append(max_pr_so_far)
                    
#                     if update_tree:
#                         # Al encontrar una mejora, rompemos el for para regenerar el vecindario
#                         break 
#                     else:
#                         # Si es inmutable, deshacemos pero guardamos el récord
#                         arbol.remove_edge(hoja, v_nuevo)
#                         arbol.add_edge(hoja, v_viejo)
#                 else:
#                     # Si no es mayor (es menor o igual), se rechaza
#                     arbol.remove_edge(hoja, v_nuevo)
#                     arbol.add_edge(hoja, v_viejo)
#                     historial_actual.append(max_pr_so_far)

#             # --- Evaluaciones de parada ---
#             tiempo_total_bucle = time.time() - start_time
            
#             if tiempo_total_bucle >= t_max:
#                 motivo = "Tiempo excedido"
#                 break
#             elif not hubo_mejora_en_ronda:
#                 # Se revisó TODO el vecindario y no hubo ninguna mejora estricta
#                 motivo = "Vecindario agotado (Óptimo local)"
#                 break
#             else:
#                 # Si hubo mejora, actualizamos los movimientos disponibles para la nueva topología
#                 movimientos = obtener_movimientos(arbol)

#         # Si terminó y no hubo ninguna mejora en absoluto para esta matriz
#         if not hubo_mejora_estricta_global:
#             registros_mejoras.append({
#                 'Categoría': categoria, 'ID Matriz': id_real, 'Dimensión (n)': n,
#                 'Iteración Local': iteracion_local, 'Tiempo (s)': round(time.time() - start_time, 4),
#                 'PR Anterior': max_pr_so_far, 'PR Nuevo': max_pr_so_far, 'Mejora': 0.0
#             })

#         arboles_fase3.append(arbol.copy())
#         historiales_fase3.append(historial_actual)
            
#         print(f"  -> Detenido por: {motivo}. Mejor PR: {max_pr_so_far:.4f}")

#     # Exportación
#     df_mejoras = pd.DataFrame(registros_mejoras)
#     timestamp = datetime.now().strftime('%d%m%y_%H%M')
#     if not df_mejoras.empty:
#         df_mejoras.to_excel(f'Mejoras_Fase3_FI_{timestamp}.xlsx', index=False)

#     return historiales_fase3, arboles_fase3, df_mejoras




# def busqueda_local_FI_tiempo_modif(matrices, arboles_base, historiales_fase2, lista_ids, t_max, k_segundos, update_tree=True, categoria="FI-Modif"):
#     historiales_fase3 = []
#     arboles_fase3 = []
#     registros = [] 
    
#     def contar_hojas(arbol_actual):
#         return sum(1 for _, grado in arbol_actual.degree() if grado == 1) if arbol_actual else 0
    
#     print(f"\n--- Fase 3: Local Search FI [Solo Tiempo] ({categoria}) ---")
#     print(f"Parámetros: t_max={t_max}s | Reportes cada k={k_segundos}s | Actualizar Árbol={update_tree}")

#     for i, D in enumerate(matrices):
#         n = len(D)
#         id_real = lista_ids[i]
        
#         max_pr_so_far = historiales_fase2[i][-1]
#         historial_actual = [max_pr_so_far]
        
#         arbol = arboles_base[i].copy() if arboles_base[i] is not None else None
        
#         if arbol is None:
#             historiales_fase3.append(historial_actual)
#             arboles_fase3.append(None)
#             continue
            
#         print(f"Procesando ID {id_real} (n={n})... PR Base: {max_pr_so_far:.4f} | Hojas Iniciales: {contar_hojas(arbol)}")
        
#         # 1. Registro inicial
#         registros.append({
#             'ID Matriz': id_real,
#             'n': n, # NUEVO
#             'Evento': 'Inicio',
#             'tiempo': 0.0,
#             'iteracion': 0,
#             'metrica': max_pr_so_far,
#             'hojas': contar_hojas(arbol) 
#         })
        
#         start_time = time.time()
#         next_report_time = start_time + k_segundos
#         iteracion_local = 0
#         motivo_detencion = ""
        
#         # Memoria Tabú para FI
#         arboles_revisados = set()
#         firma_base = frozenset([tuple(sorted(e)) for e in arbol.edges()])
#         arboles_revisados.add(firma_base)
        
#         def obtener_movimientos(arbol_actual):
#             movs = []
#             hojas_nodos = [nodo for nodo, grado in arbol_actual.degree() if grado == 1]
#             for h in hojas_nodos:
#                 vecino_actual = list(arbol_actual.neighbors(h))[0]
#                 posibles = [nodo for nodo in arbol_actual.nodes() if nodo != h and nodo != vecino_actual]
#                 for p in posibles:
#                     movs.append((h, vecino_actual, p))
#             random.shuffle(movs)
#             return movs

#         movimientos = obtener_movimientos(arbol)
        
#         while (time.time() - start_time) < t_max:
#             hubo_cambio = False
            
#             for mov in movimientos:
#                 current_time = time.time()
#                 tiempo_transcurrido = current_time - start_time
                
#                 if tiempo_transcurrido >= t_max:
#                     break
                    
#                 if current_time >= next_report_time:
#                     registros.append({
#                         'ID Matriz': id_real,
#                         'n': n, # NUEVO
#                         'Evento': f'Reporte {k_segundos} s',
#                         'tiempo': round(tiempo_transcurrido, 4),
#                         'iteracion': iteracion_local,
#                         'metrica': max_pr_so_far,
#                         'hojas': contar_hojas(arbol)
#                     })
#                     next_report_time += k_segundos 
                
#                 hoja, v_viejo, v_nuevo = mov
#                 iteracion_local += 1
                
#                 arbol.remove_edge(hoja, v_viejo)
#                 arbol.add_edge(hoja, v_nuevo)
                
#                 val = reconocer_caminos_robinson(arbol, D)[0]
#                 nuevo_pr = 2 * val / (n * (n - 1))
                
#                 if nuevo_pr >= max_pr_so_far:
#                     firma_nuevo_arbol = frozenset([tuple(sorted(e)) for e in arbol.edges()])
                    
#                     if firma_nuevo_arbol in arboles_revisados:
#                         arbol.remove_edge(hoja, v_nuevo)
#                         arbol.add_edge(hoja, v_viejo)
#                         motivo_detencion = "Atrapado en ciclo Tabú (Óptimo Local)"
#                         hubo_cambio = False
#                         break 
#                     else:
#                         arboles_revisados.add(firma_nuevo_arbol)
                        
#                         if nuevo_pr > max_pr_so_far:
#                             max_pr_so_far = nuevo_pr
#                             registros.append({
#                                 'ID Matriz': id_real,
#                                 'n': n, # NUEVO
#                                 'Evento': 'Mejora Estricta',
#                                 'tiempo': round(time.time() - start_time, 4),
#                                 'iteracion': iteracion_local,
#                                 'metrica': max_pr_so_far,
#                                 'hojas': contar_hojas(arbol)
#                             })
                            
#                         historial_actual.append(max_pr_so_far)
                        
#                         if not update_tree:
#                             arbol.remove_edge(hoja, v_nuevo)
#                             arbol.add_edge(hoja, v_viejo)
#                         else:
#                             hubo_cambio = True
#                             break 
#                 else:
#                     arbol.remove_edge(hoja, v_nuevo)
#                     arbol.add_edge(hoja, v_viejo)
#                     historial_actual.append(max_pr_so_far)

#             tiempo_total = time.time() - start_time
            
#             if tiempo_total >= t_max:
#                 motivo_detencion = "Tiempo Máximo Excedido"
#                 break
#             elif motivo_detencion:
#                 break
#             elif not hubo_cambio:
#                 motivo_detencion = "Vecindario Agotado (Óptimo Local)"
#                 break
#             else:
#                 movimientos = obtener_movimientos(arbol)

#         if not motivo_detencion:
#             motivo_detencion = "Tiempo Máximo Excedido"

#         # Registro Final
#         registros.append({
#             'ID Matriz': id_real,
#             'n': n, # NUEVO
#             'Evento': f'Fin: {motivo_detencion}',
#             'tiempo': round(time.time() - start_time, 4),
#             'iteracion': iteracion_local,
#             'metrica': max_pr_so_far,
#             'hojas': contar_hojas(arbol)
#         })

#         print(f"  -> Detenido por: {motivo_detencion}. Mejor PR: {max_pr_so_far:.4f} | Hojas Finales: {contar_hojas(arbol)}")
#         arboles_fase3.append(arbol.copy())
#         historiales_fase3.append(historial_actual)

#     df_registros = pd.DataFrame(registros)
#     # Orden de columnas actualizado
#     df_registros = df_registros[['ID Matriz', 'n', 'Evento', 'tiempo', 'iteracion', 'metrica', 'hojas']]
    
#     timestamp = datetime.now().strftime('%d%m%y_%H%M')
#     nombre_archivo = f'Registros_Fase3_FI_Tabu_{timestamp}.xlsx'
#     if not df_registros.empty:
#         df_registros.to_excel(nombre_archivo, index=False)
#         print(f"\n-> Resultados exportados a: {nombre_archivo}")

#     return historiales_fase3, arboles_fase3, df_registros



# def busqueda_local_BI_tiempo_modif(matrices, arboles_base, historiales_fase2, lista_ids, t_max, k_segundos, update_tree=True, categoria="BI-Modif"):
#     historiales_fase3 = []
#     arboles_fase3 = []
#     registros = [] 
    
#     def contar_hojas(arbol_actual):
#         return sum(1 for _, grado in arbol_actual.degree() if grado == 1) if arbol_actual else 0
    
#     print(f"\n--- Fase 3: Local Search BI [Solo Tiempo] ({categoria}) ---")
#     print(f"Parámetros: t_max={t_max}s | Reportes cada k={k_segundos}s | Actualizar Árbol={update_tree}")

#     for i, D in enumerate(matrices):
#         n = len(D)
#         id_real = lista_ids[i]
        
#         max_pr_so_far = historiales_fase2[i][-1]
#         historial_actual = [max_pr_so_far]
        
#         arbol = arboles_base[i].copy() if arboles_base[i] is not None else None
        
#         if arbol is None:
#             historiales_fase3.append(historial_actual)
#             arboles_fase3.append(None)
#             continue
            
#         print(f"Procesando ID {id_real} (n={n})... PR Base: {max_pr_so_far:.4f} | Hojas Iniciales: {contar_hojas(arbol)}")
        
#         # 1. Registro inicial
#         registros.append({
#             'ID Matriz': id_real,
#             'n': n,               # NUEVO
#             'ventana': 0,         # NUEVO: Comienza en la ventana 0
#             'Evento': 'Inicio',
#             'tiempo': 0.0,
#             'iteracion': 0,
#             'metrica': max_pr_so_far,
#             'hojas': contar_hojas(arbol)
#         })
        
#         start_time = time.time()
#         next_report_time = start_time + k_segundos
#         iteracion_local_total = 0
#         ventana_actual = 0        # NUEVO: Contador de ventanas para el registro
#         motivo_detencion = ""
        
#         arboles_revisados = set()
#         firma_base = frozenset([tuple(sorted(e)) for e in arbol.edges()])
#         arboles_revisados.add(firma_base)
        
#         while (time.time() - start_time) < t_max:
#             ventana_actual += 1   # NUEVO: Iniciamos una nueva ventana
#             mejor_pr_ronda = -1
#             mejor_movimiento_ronda = None
            
#             movimientos_posibles = []
#             hojas = [nodo for nodo, grado in arbol.degree() if grado == 1]
#             for hoja in hojas:
#                 v_viejo = list(arbol.neighbors(hoja))[0]
#                 posibles = [nodo for nodo in arbol.nodes() if nodo != hoja and nodo != v_viejo]
#                 for p in posibles:
#                     movimientos_posibles.append((hoja, v_viejo, p))
                    
#             random.shuffle(movimientos_posibles)
            
#             for mov in movimientos_posibles:
#                 current_time = time.time()
#                 tiempo_transcurrido = current_time - start_time
                
#                 if tiempo_transcurrido >= t_max:
#                     break
                    
#                 if current_time >= next_report_time:
#                     registros.append({
#                         'ID Matriz': id_real,
#                         'n': n,
#                         'ventana': ventana_actual, # NUEVO
#                         'Evento': f'Reporte {k_segundos} s',
#                         'tiempo': round(tiempo_transcurrido, 4),
#                         'iteracion': iteracion_local_total,
#                         'metrica': max_pr_so_far,
#                         'hojas': contar_hojas(arbol)
#                     })
#                     next_report_time += k_segundos
                
#                 hoja, v_viejo, v_nuevo = mov
#                 iteracion_local_total += 1
                
#                 arbol.remove_edge(hoja, v_viejo)
#                 arbol.add_edge(hoja, v_nuevo)
                
#                 val = reconocer_caminos_robinson(arbol, D)[0]
#                 nuevo_pr = 2 * val / (n * (n - 1))
                
#                 if nuevo_pr > mejor_pr_ronda:
#                     mejor_pr_ronda = nuevo_pr
#                     mejor_movimiento_ronda = mov
                    
#                 arbol.remove_edge(hoja, v_nuevo)
#                 arbol.add_edge(hoja, v_viejo)
                
#                 historial_actual.append(max_pr_so_far)
            
#             tiempo_total = time.time() - start_time
            
#             if mejor_movimiento_ronda is not None:
#                 hoja_ganadora, v_viejo, v_nuevo = mejor_movimiento_ronda
                
#                 arbol.remove_edge(hoja_ganadora, v_viejo)
#                 arbol.add_edge(hoja_ganadora, v_nuevo)
                
#                 firma_nuevo_arbol = frozenset([tuple(sorted(e)) for e in arbol.edges()])
                
#                 if firma_nuevo_arbol in arboles_revisados:
#                     motivo_detencion = "Atrapado en ciclo Tabú (Óptimo Local)"
#                     arbol.remove_edge(hoja_ganadora, v_nuevo)
#                     arbol.add_edge(hoja_ganadora, v_viejo)
#                     break
#                 else:
#                     arboles_revisados.add(firma_nuevo_arbol)
                
#                 if mejor_pr_ronda > max_pr_so_far:
#                     max_pr_so_far = mejor_pr_ronda
                    
#                     registros.append({
#                         'ID Matriz': id_real,
#                         'n': n,
#                         'ventana': ventana_actual, # NUEVO
#                         'Evento': 'Mejora Estricta',
#                         'tiempo': round(time.time() - start_time, 4),
#                         'iteracion': iteracion_local_total,
#                         'metrica': max_pr_so_far,
#                         'hojas': contar_hojas(arbol)
#                     })
#                     historial_actual[-1] = max_pr_so_far
                
#                 if not update_tree:
#                     arbol.remove_edge(hoja_ganadora, v_nuevo)
#                     arbol.add_edge(hoja_ganadora, v_viejo)
#                     motivo_detencion = "Vecindario Agotado (Árbol inmutable)"
#                     break

#             if tiempo_total >= t_max:
#                 motivo_detencion = "Tiempo Máximo Excedido"
#                 break
                
#         if not motivo_detencion:
#             motivo_detencion = "Tiempo Máximo Excedido"

#         # Registro Final
#         registros.append({
#             'ID Matriz': id_real,
#             'n': n,
#             'ventana': ventana_actual, # NUEVO: Ventana en la que terminó
#             'Evento': f'Fin: {motivo_detencion}',
#             'tiempo': round(time.time() - start_time, 4),
#             'iteracion': iteracion_local_total,
#             'metrica': max_pr_so_far,
#             'hojas': contar_hojas(arbol)
#         })

#         print(f"  -> Detenido por: {motivo_detencion}. Ventanas: {ventana_actual} | Mejor PR: {max_pr_so_far:.4f} | Hojas Finales: {contar_hojas(arbol)}")
#         arboles_fase3.append(arbol.copy())
#         historiales_fase3.append(historial_actual)

#     df_registros = pd.DataFrame(registros)
#     # Orden de columnas actualizado
#     df_registros = df_registros[['ID Matriz', 'n', 'ventana', 'Evento', 'tiempo', 'iteracion', 'metrica', 'hojas']]
    
#     timestamp = datetime.now().strftime('%d%m%y_%H%M')
#     nombre_archivo = f'Registros_Fase3_BI_Tiempo_{timestamp}.xlsx'
#     if not df_registros.empty:
#         df_registros.to_excel(nombre_archivo, index=False)
#         print(f"\n-> Resultados exportados a: {nombre_archivo}")

#     return historiales_fase3, arboles_fase3, df_registros




# def ejecutar_FI_estricto(D, arbol_base, id_real, mst_index, pr_base, t_max):
#     """Ejecuta First Improvement sobre un único árbol y retorna sus registros."""
#     n = len(D)
#     arbol = arbol_base.copy()
#     registros = []
#     max_pr_so_far = pr_base
    
#     # 1. Registro INICIO
#     registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': 0.0, 
#                       'iteracion': 0, 'mejora': 0.0, 
#                       'pr_best': max_pr_so_far, 'evento': 'Inicio'})
                      
#     start_time = time.time()
#     iteracion = 0
#     movimientos = obtener_movimientos(arbol)
    
#     while (time.time() - start_time) < t_max:
#         hubo_mejora_en_ronda = False
        
#         for mov in movimientos:
#             t_actual = time.time() - start_time
#             if t_actual >= t_max: break
                
#             hoja, v_viejo, v_nuevo = mov
#             iteracion += 1
            
#             # Movimiento temporal
#             arbol.remove_edge(hoja, v_viejo)
#             arbol.add_edge(hoja, v_nuevo)
            
#             val = reconocer_caminos_robinson(arbol, D)[0] # Asegúrate de que 'qr' esté importado
#             nuevo_pr = 2 * val / (n * (n - 1))
            
#             # Mejora Estricta (FIRST IMPROVEMENT)
#             if nuevo_pr > max_pr_so_far:
#                 mejora = nuevo_pr - max_pr_so_far
#                 max_pr_so_far = nuevo_pr
                
#                 # 2. Registro MEJORA
#                 registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_actual, 4), 
#                                   'iteracion': iteracion, 'mejora': mejora, 
#                                   'pr_best': max_pr_so_far, 'evento': 'Mejora'})
#                 hubo_mejora_en_ronda = True
#                 break # Rompemos para regenerar vecindario desde la nueva topología
#             else:
#                 # Deshacer
#                 arbol.remove_edge(hoja, v_nuevo)
#                 arbol.add_edge(hoja, v_viejo)
                
#         # Controles de parada
#         if (time.time() - start_time) >= t_max: break
#         elif not hubo_mejora_en_ronda: break # Óptimo local alcanzado
#         else: movimientos = obtener_movimientos(arbol)
            
#     # 3. Registro FIN
#     t_final = time.time() - start_time
#     registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_final, 4), 
#                       'iteracion': iteracion, 'mejora': 0.0, 
#                       'pr_best': max_pr_so_far, 'evento': 'Fin'})
                      
#     return arbol, registros


# def ejecutar_BI_estricto(D, arbol_base, id_real, mst_index, pr_base, t_max):
#     """Ejecuta Best Improvement sobre un único árbol y retorna sus registros."""
#     n = len(D)
#     arbol = arbol_base.copy()
#     registros = []
#     max_pr_so_far = pr_base
    
#     registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': 0.0, 
#                       'iteracion': 0, 'mejora': 0.0, 
#                       'pr_best': max_pr_so_far, 'evento': 'Inicio'})
                      
#     start_time = time.time()
#     iteracion = 0
    
#     while (time.time() - start_time) < t_max:
#         mejor_pr_ronda = -1
#         mejor_mov_ronda = None
        
#         movimientos = obtener_movimientos(arbol)
        
#         # Evaluar TODO el vecindario
#         for mov in movimientos:
#             t_actual = time.time() - start_time
#             if t_actual >= t_max: break
                
#             hoja, v_viejo, v_nuevo = mov
#             iteracion += 1
            
#             arbol.remove_edge(hoja, v_viejo)
#             arbol.add_edge(hoja, v_nuevo)
            
#             val = reconocer_caminos_robinson(arbol, D)[0]
#             nuevo_pr = 2 * val / (n * (n - 1))
            
#             if nuevo_pr > mejor_pr_ronda:
#                 mejor_pr_ronda = nuevo_pr
#                 mejor_mov_ronda = mov
                
#             arbol.remove_edge(hoja, v_nuevo)
#             arbol.add_edge(hoja, v_viejo)
            
#         t_actual = time.time() - start_time
        
#         # Mejora Estricta (BEST IMPROVEMENT)
#         if mejor_mov_ronda is not None and mejor_pr_ronda > max_pr_so_far:
#             hoja_g, v_v, v_n = mejor_mov_ronda
#             arbol.remove_edge(hoja_g, v_v)
#             arbol.add_edge(hoja_g, v_n)
            
#             mejora = mejor_pr_ronda - max_pr_so_far
#             max_pr_so_far = mejor_pr_ronda
            
#             registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_actual, 4), 
#                               'iteracion': iteracion, 'mejora': mejora, 
#                               'pr_best': max_pr_so_far, 'evento': 'Mejora'})
#         else:
#             break # Óptimo local alcanzado
            
#         if t_actual >= t_max: break

#     t_final = time.time() - start_time
#     registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_final, 4), 
#                       'iteracion': iteracion, 'mejora': 0.0, 
#                       'pr_best': max_pr_so_far, 'evento': 'Fin'})
                      
#     return arbol, registros

# 4 de mayo 2026.


# def ejecutar_FI_estricto(D, arbol_base, id_real, mst_index, pr_base, t_max, start_id_time):
#     """Ejecuta First Improvement sobre un único árbol y retorna sus registros con tiempo global del ID."""
#     n = len(D)
#     arbol = arbol_base.copy()
#     registros = []
#     max_pr_so_far = pr_base
    
#     t_global_inicio = time.time() - start_id_time
    
#     # Registro Inicio
#     registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_global_inicio, 4), 
#                       'iteracion': 0, 'pr_actual': pr_base, 'mejora': 0.0, 
#                       'pr_best': max_pr_so_far, 'evento': 'Inicio'})
                      
#     start_local_time = time.time()
#     iteracion = 0
#     movimientos = obtener_movimientos(arbol)
    
#     while (time.time() - start_local_time) < t_max:
#         hubo_mejora_en_ronda = False
        
#         for mov in movimientos:
#             t_actual_local = time.time() - start_local_time
#             if t_actual_local >= t_max: break
                
#             hoja, v_viejo, v_nuevo = mov
#             iteracion += 1
            
#             arbol.remove_edge(hoja, v_viejo)
#             arbol.add_edge(hoja, v_nuevo)
            
#             val = reconocer_caminos_robinson(arbol, D)[0] 
#             nuevo_pr = 2 * val / (n * (n - 1))
            
#             if nuevo_pr > max_pr_so_far:
#                 mejora = nuevo_pr - max_pr_so_far
#                 max_pr_so_far = nuevo_pr
#                 t_global_mejora = time.time() - start_id_time
                
#                 # Registro Mejora
#                 registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_global_mejora, 4), 
#                                   'iteracion': iteracion, 'pr_actual': nuevo_pr, 'mejora': mejora, 
#                                   'pr_best': max_pr_so_far, 'evento': 'Mejora'})
#                 hubo_mejora_en_ronda = True
#                 break 
#             else:
#                 arbol.remove_edge(hoja, v_nuevo)
#                 arbol.add_edge(hoja, v_viejo)
                
#         if (time.time() - start_local_time) >= t_max: break
#         elif not hubo_mejora_en_ronda: break
#         else: movimientos = obtener_movimientos(arbol)
            
#     t_global_fin = time.time() - start_id_time
    
#     # Registro Fin
#     registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_global_fin, 4), 
#                       'iteracion': iteracion, 'pr_actual': max_pr_so_far, 'mejora': 0.0, 
#                       'pr_best': max_pr_so_far, 'evento': 'Fin'})
                      
#     return arbol, registros


# def ejecutar_BI_estricto(D, arbol_base, id_real, mst_index, pr_base, t_max, start_id_time):
#     """Ejecuta Best Improvement sobre un único árbol y retorna sus registros con tiempo global del ID."""
#     n = len(D)
#     arbol = arbol_base.copy()
#     registros = []
#     max_pr_so_far = pr_base
    
#     t_global_inicio = time.time() - start_id_time
    
#     # Registro Inicio
#     registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_global_inicio, 4), 
#                       'iteracion': 0, 'pr_actual': pr_base, 'mejora': 0.0, 
#                       'pr_best': max_pr_so_far, 'evento': 'Inicio'})
                      
#     start_local_time = time.time() 
#     iteracion = 0
    
#     while (time.time() - start_local_time) < t_max:
#         mejor_pr_ronda = -1
#         mejor_mov_ronda = None
        
#         movimientos = obtener_movimientos(arbol)
        
#         # Evaluar TODO el vecindario
#         for mov in movimientos:
#             t_actual_local = time.time() - start_local_time
#             if t_actual_local >= t_max: break
                
#             hoja, v_viejo, v_nuevo = mov
#             iteracion += 1
            
#             arbol.remove_edge(hoja, v_viejo)
#             arbol.add_edge(hoja, v_nuevo)
            
#             val = reconocer_caminos_robinson(arbol, D)[0]
#             nuevo_pr = 2 * val / (n * (n - 1))
            
#             if nuevo_pr > mejor_pr_ronda:
#                 mejor_pr_ronda = nuevo_pr
#                 mejor_mov_ronda = mov
                
#             arbol.remove_edge(hoja, v_nuevo)
#             arbol.add_edge(hoja, v_viejo)
            
#         t_actual_local = time.time() - start_local_time
#         t_global_mejora = time.time() - start_id_time 
        
#         # Mejora Estricta (BEST IMPROVEMENT)
#         if mejor_mov_ronda is not None and mejor_pr_ronda > max_pr_so_far:
#             hoja_g, v_v, v_n = mejor_mov_ronda
#             arbol.remove_edge(hoja_g, v_v)
#             arbol.add_edge(hoja_g, v_n)
            
#             mejora = mejor_pr_ronda - max_pr_so_far
#             max_pr_so_far = mejor_pr_ronda
            
#             # Registro Mejora
#             registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_global_mejora, 4), 
#                               'iteracion': iteracion, 'pr_actual': mejor_pr_ronda, 'mejora': mejora, 
#                               'pr_best': max_pr_so_far, 'evento': 'Mejora'})
#         else:
#             break # Óptimo local alcanzado
            
#         if t_actual_local >= t_max: break

#     t_global_fin = time.time() - start_id_time
    
#     # Registro Fin
#     registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_global_fin, 4), 
#                       'iteracion': iteracion, 'pr_actual': max_pr_so_far, 'mejora': 0.0, 
#                       'pr_best': max_pr_so_far, 'evento': 'Fin'})
                      
#     return arbol, registros


def ConstructionPhase2(D, t_mst, n_pert, epsilon, distribucion='normal'):
    """
    Combina la búsqueda de MSTs base y la aplicación de ruido, 
    y ahora retorna un diccionario con las métricas de esta fase.
    """
    n = len(D)
    
    # ==========================================
    # SUB-FASE 1: BÚSQUEDA MST BASE
    # ==========================================
    start_mst = time.time()
    
    L = buscar_mst_distintos(D, t_mst)
    trees_found = list(L[1])
    
    if not trees_found:
        trees_found = [MST(D)]
        
    cantidad_mst_encontrados = len(trees_found)
    
    # Calcular el mejor PR ANTES del ruido para el reporte
    mejor_pr_mst_base = 0.0
    for arbol in trees_found:
        val = reconocer_caminos_robinson(arbol, D)[0]
        pr_actual = 2 * val / (n * (n - 1))
        if pr_actual > mejor_pr_mst_base:
            mejor_pr_mst_base = pr_actual
            
    tiempo_empleado_mst = time.time() - start_mst
    
    # ==========================================
    # SUB-FASE 2: APLICACIÓN DE RUIDO
    # ==========================================
    start_ruido = time.time()
    
    arboles_perturbados_con_metricas = []
    iteraciones_ruido_totales = 0
    
    for arbol_base in trees_found:
        val_base_original = reconocer_caminos_robinson(arbol_base, D)[0]
        mejor_pr_ruido = 2 * val_base_original / (n * (n - 1))
        mejor_arbol_perturbado = arbol_base.copy()
        
        for j in range(n_pert):
            iteraciones_ruido_totales += 1
            
            if distribucion == 'normal':
                R = np.random.normal(0, 1, size=(n, n))
            else:
                R = np.random.uniform(-1, 1, size=(n, n))
                
            R = (R + R.T) / 2.0
            np.fill_diagonal(R, 0)
            
            D_noise = np.abs(D + epsilon * R)
            arbol_candidato = MST(D_noise) 
            
            val_candidato = reconocer_caminos_robinson(arbol_candidato, D)[0]
            pr_candidato = 2 * val_candidato / (n * (n - 1))
            
            if pr_candidato > mejor_pr_ruido:
                mejor_pr_ruido = pr_candidato
                mejor_arbol_perturbado = arbol_candidato.copy()
                
        arboles_perturbados_con_metricas.append((mejor_arbol_perturbado, mejor_pr_ruido))
        
    tiempo_empleado_ruido = time.time() - start_ruido
    
    # ==========================================
    # CIERRE Y EMPAQUETADO
    # ==========================================
    arboles_perturbados_con_metricas.sort(key=lambda x: x[1], reverse=True)
    
    T_pool = [item[0] for item in arboles_perturbados_con_metricas]
    metricas_pool = [item[1] for item in arboles_perturbados_con_metricas]
    
    mejor_metrica_ruido = metricas_pool[0]
    
    # Construimos el diccionario con el reporte exacto que pediste
    reporte_fase_1_y_2 = {
        'Cant_MST_Encontrados': cantidad_mst_encontrados,
        'Tiempo_MST_s': round(tiempo_empleado_mst, 4),
        'Mejor_Metrica_MST': round(mejor_pr_mst_base, 4),
        'Cant_Iteraciones_Ruido': iteraciones_ruido_totales,
        'Tiempo_Ruido_s': round(tiempo_empleado_ruido, 4),
        'Mejor_Metrica_Ruido': round(mejor_metrica_ruido, 4)
    }
    
    return T_pool, metricas_pool, reporte_fase_1_y_2

def experimento_preparacion_independiente(matrices, lista_ids, t_mst, n_pert, epsilon=0.25, distribucion='normal'):
    """
    Ejecuta EXCLUSIVAMENTE la Fase Constructiva (MST) y la inyección de Ruido.
    Ideal para calibrar parámetros sin correr el costoso pipeline de búsqueda local.
    """
    reporte_preparacion = []
    
    print(f"=== INICIANDO EXPERIMENTO INDEPENDIENTE: MST + RUIDO ===")
    print(f"Tiempo MST: {t_mst}s | Iteraciones Ruido: {n_pert} | eps: {epsilon}\n")

    for i, D in enumerate(matrices):
        id_real = lista_ids[i]
        n = len(D)
        
        print(f"[{id_real}] Procesando (n={n})...")
        
        # Utilizamos directamente la función ConstructionPhase que ya creamos
        # (Asegúrate de haber ejecutado la celda donde definimos ConstructionPhase antes)
        T_pool, metricas_pool, reporte_f1_f2 = ConstructionPhase2(D, t_mst, n_pert, epsilon, distribucion)
        
        # Empaquetamos los datos con la info de la matriz
        datos_matriz = {
            'ID': id_real,
            'n': n
        }
        # Fusionamos el ID y la n con los datos del reporte interno
        datos_matriz.update(reporte_f1_f2) 
        
        reporte_preparacion.append(datos_matriz)
        
        # Feedback rápido en consola para saber cómo le fue
        msts = reporte_f1_f2['Cant_MST_Encontrados']
        pr_base = reporte_f1_f2['Mejor_Metrica_MST']
        pr_ruido = reporte_f1_f2['Mejor_Metrica_Ruido']
        print(f"  -> {msts} MSTs hallados | PR Base: {pr_base:.4f} | PR post-Ruido: {pr_ruido:.4f}")

    # ==========================================
    # EXPORTACIÓN DEL REPORTE
    # ==========================================
    carpeta_salida = 'resultados meta'
    os.makedirs(carpeta_salida, exist_ok=True)
    timestamp = datetime.now().strftime('%d%m%y_%H%M')
    
    df_reporte = pd.DataFrame(reporte_preparacion)
    nombre_archivo = os.path.join(carpeta_salida, f'Experimento_Fase1_y_Ruido_{timestamp}.xlsx')
    
    df_reporte.to_excel(nombre_archivo, index=False)
    
    print(f"\n✅ Experimento completado con éxito.")
    print(f"📊 Excel generado en: {nombre_archivo}")
    
    return df_reporte


def ejecutar_FI_estricto(D, arbol_base, id_real, mst_index, pr_base, t_max, tiempo_previo_acumulado):
    n = len(D)
    arbol = arbol_base.copy()
    registros = []
    max_pr_so_far = pr_base
    
    # Registro Inicio
    registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(tiempo_previo_acumulado, 4), 
                      'iteracion': 0, 'pr_actual': pr_base, 'mejora': 0.0, 
                      'pr_best': max_pr_so_far, 'evento': 'Inicio'})
                      
    start_local_time = time.time()
    iteracion = 0
    movimientos = obtener_movimientos(arbol)
    
    while (time.time() - start_local_time) < t_max:
        hubo_mejora_en_ronda = False
        
        for mov in movimientos:
            t_actual_local = time.time() - start_local_time
            if t_actual_local >= t_max: break
                
            hoja, v_viejo, v_nuevo = mov
            iteracion += 1
            
            arbol.remove_edge(hoja, v_viejo)
            arbol.add_edge(hoja, v_nuevo)
            
            val = reconocer_caminos_robinson(arbol, D)[0] 
            nuevo_pr = 2 * val / (n * (n - 1))
            
            if nuevo_pr > max_pr_so_far:
                mejora = nuevo_pr - max_pr_so_far
                max_pr_so_far = nuevo_pr
                t_global_mejora = tiempo_previo_acumulado + t_actual_local
                
                # Registro Mejora
                registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_global_mejora, 4), 
                                  'iteracion': iteracion, 'pr_actual': nuevo_pr, 'mejora': mejora, 
                                  'pr_best': max_pr_so_far, 'evento': 'Mejora'})
                hubo_mejora_en_ronda = True
                break 
            else:
                arbol.remove_edge(hoja, v_nuevo)
                arbol.add_edge(hoja, v_viejo)
                
        if (time.time() - start_local_time) >= t_max: break
        elif not hubo_mejora_en_ronda: break
        else: movimientos = obtener_movimientos(arbol)
            
    t_final_local = time.time() - start_local_time
    t_global_fin = tiempo_previo_acumulado + t_final_local
    
    # Registro Fin
    registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_global_fin, 4), 
                      'iteracion': iteracion, 'pr_actual': max_pr_so_far, 'mejora': 0.0, 
                      'pr_best': max_pr_so_far, 'evento': 'Fin'})
                      
    return arbol, registros, t_final_local


def ejecutar_BI_estricto(D, arbol_base, id_real, mst_index, pr_base, t_max, tiempo_previo_acumulado):
    n = len(D)
    arbol = arbol_base.copy()
    registros = []
    max_pr_so_far = pr_base
    
    # Registro Inicio
    registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(tiempo_previo_acumulado, 4), 
                      'iteracion': 0, 'pr_actual': pr_base, 'mejora': 0.0, 
                      'pr_best': max_pr_so_far, 'evento': 'Inicio'})
                      
    start_local_time = time.time() 
    iteracion = 0
    
    while (time.time() - start_local_time) < t_max:
        mejor_pr_ronda = -1
        mejor_mov_ronda = None
        
        movimientos = obtener_movimientos(arbol)
        
        for mov in movimientos:
            t_actual_local = time.time() - start_local_time
            if t_actual_local >= t_max: break
                
            hoja, v_viejo, v_nuevo = mov
            iteracion += 1
            
            arbol.remove_edge(hoja, v_viejo)
            arbol.add_edge(hoja, v_nuevo)
            
            val = reconocer_caminos_robinson(arbol, D)[0]
            nuevo_pr = 2 * val / (n * (n - 1))
            
            if nuevo_pr > mejor_pr_ronda:
                mejor_pr_ronda = nuevo_pr
                mejor_mov_ronda = mov
                
            arbol.remove_edge(hoja, v_nuevo)
            arbol.add_edge(hoja, v_viejo)
            
        t_actual_local = time.time() - start_local_time
        
        if mejor_mov_ronda is not None and mejor_pr_ronda > max_pr_so_far:
            hoja_g, v_v, v_n = mejor_mov_ronda
            arbol.remove_edge(hoja_g, v_v)
            arbol.add_edge(hoja_g, v_n)
            
            mejora = mejor_pr_ronda - max_pr_so_far
            max_pr_so_far = mejor_pr_ronda
            t_global_mejora = tiempo_previo_acumulado + t_actual_local
            
            # Registro Mejora
            registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_global_mejora, 4), 
                              'iteracion': iteracion, 'pr_actual': mejor_pr_ronda, 'mejora': mejora, 
                              'pr_best': max_pr_so_far, 'evento': 'Mejora'})
        else:
            break 
            
        if t_actual_local >= t_max: break

    t_final_local = time.time() - start_local_time
    t_global_fin = tiempo_previo_acumulado + t_final_local
    
    # Registro Fin
    registros.append({'ID': id_real, 'n': n, 'mst': mst_index, 'tiempo': round(t_global_fin, 4), 
                      'iteracion': iteracion, 'pr_actual': max_pr_so_far, 'mejora': 0.0, 
                      'pr_best': max_pr_so_far, 'evento': 'Fin'})
                      
    return arbol, registros, t_final_local




def ConstructionPhase(D, t_mst, n_pert, epsilon, distribucion='normal'):
    """
    Combina la búsqueda de MSTs base y la aplicación de ruido.
    Devuelve un pool de árboles perturbados ordenados de mejor a peor PR.
    """
    n = len(D)
    
    # 1. Búsqueda de MSTs base
    L = buscar_mst_distintos(D, t_mst)
    trees_found = list(L[1])
    
    # Fallback por si la función no encuentra nada
    if not trees_found:
        trees_found = [MST(D)]
        
    arboles_perturbados_con_metricas = []
    
    # 2. Aplicar n_pert iteraciones de ruido a CADA árbol base encontrado
    for arbol_base in trees_found:
        val_base_original = reconocer_caminos_robinson(arbol_base, D)[0]
        mejor_pr_ruido = 2 * val_base_original / (n * (n - 1))
        mejor_arbol_perturbado = arbol_base.copy()
        
        for j in range(n_pert):
            if distribucion == 'normal':
                R = np.random.normal(0, 1, size=(n, n))
            else:
                R = np.random.uniform(-1, 1, size=(n, n))
                
            R = (R + R.T) / 2.0
            np.fill_diagonal(R, 0)
            
            D_noise = np.abs(D + epsilon * R)
            arbol_candidato = MST(D_noise) 
            
            val_candidato = reconocer_caminos_robinson(arbol_candidato, D)[0]
            pr_candidato = 2 * val_candidato / (n * (n - 1))
            
            if pr_candidato > mejor_pr_ruido:
                mejor_pr_ruido = pr_candidato
                mejor_arbol_perturbado = arbol_candidato.copy()
                
        # Guardamos el mejor representante perturbado de este árbol base
        arboles_perturbados_con_metricas.append((mejor_arbol_perturbado, mejor_pr_ruido))
        
    # 3. Ordenar de mayor a menor PR (para que T_pool[0] sea el mejor)
    arboles_perturbados_con_metricas.sort(key=lambda x: x[1], reverse=True)
    
    T_pool = [item[0] for item in arboles_perturbados_con_metricas]
    metricas_pool = [item[1] for item in arboles_perturbados_con_metricas]
    
    return T_pool, metricas_pool





def pipeline_maestro_final(matrices, lista_ids, t_max, t_mst, n_pert, epsilon=0.25, distribucion='normal'):
    
    resumen_ejecutivo = []
    registros_crudos_FI = []
    registros_crudos_BI = []
    
    print(f"=== INICIANDO PIPELINE MAESTRO (Versión MS Estricto) ===")
    print(f"Límite t_max: {t_max}s | t_mst: {t_mst}s | n_pert: {n_pert} | eps: {epsilon}\n")

    for i, D in enumerate(matrices):
        id_real = lista_ids[i]
        n = len(D)
        
        print(f"[{id_real}] Iniciando procesamiento (n={n})")
        
        for estrategia_bl in ["First_Improvement", "Best_Improvement"]:
            print(f"  -> Ejecutando estrategia: {estrategia_bl}")
            
            tiempo_inicio_ms = time.time()
            tiempo_inicio_constructivo = time.time()
            
            # --- 1. FASE CONSTRUCTIVA ---
            T_pool, metricas_pool = ConstructionPhase(D, t_mst, n_pert, epsilon, distribucion)
            T_init_mejor = T_pool[0]
            Metrica_constructiva = metricas_pool[0]
            Tiempo_constructivo = time.time() - tiempo_inicio_constructivo
            
            # --- 2. FASE BÚSQUEDA LOCAL (1ra Iteración) ---
            t_restante_bl1 = t_max - (time.time() - tiempo_inicio_ms)
            if t_restante_bl1 <= 0: t_restante_bl1 = 0.1 # Seguro anti-colapso
            
            tiempo_previo = time.time() - tiempo_inicio_ms
            
            if estrategia_bl == "First_Improvement":
                T_bl_1, registros_1, _ = ejecutar_FI_estricto(D, T_init_mejor, id_real, 1, Metrica_constructiva, t_restante_bl1, tiempo_previo)
                registros_crudos_FI.extend(registros_1)
            else:
                T_bl_1, registros_1, _ = ejecutar_BI_estricto(D, T_init_mejor, id_real, 1, Metrica_constructiva, t_restante_bl1, tiempo_previo)
                registros_crudos_BI.extend(registros_1)
                
            val_bl_1 = reconocer_caminos_robinson(T_bl_1, D)[0]
            Metrica_BL = 2 * val_bl_1 / (n * (n - 1))
            Tiempo_BL = time.time() - tiempo_inicio_constructivo
            
            # --- 3. FASE MULTISTART (Ejecución completa) ---
            T_best_ms = T_bl_1
            Metrica_MS = Metrica_BL
            
            idx_pool = 1 # Empezamos desde el segundo árbol
            
            while (time.time() - tiempo_inicio_ms < t_max) and (idx_pool < len(T_pool)):
                T_init = T_pool[idx_pool]
                Metrica_base_local = metricas_pool[idx_pool]
                t_restante_ms = t_max - (time.time() - tiempo_inicio_ms)
                tiempo_previo_ms = time.time() - tiempo_inicio_ms
                
                if estrategia_bl == "First_Improvement":
                    T_local, reg_local, _ = ejecutar_FI_estricto(D, T_init, id_real, idx_pool+1, Metrica_base_local, t_restante_ms, tiempo_previo_ms)
                    registros_crudos_FI.extend(reg_local)
                else:
                    T_local, reg_local, _ = ejecutar_BI_estricto(D, T_init, id_real, idx_pool+1, Metrica_base_local, t_restante_ms, tiempo_previo_ms)
                    registros_crudos_BI.extend(reg_local)
                
                val_local = reconocer_caminos_robinson(T_local, D)[0]
                pr_local = 2 * val_local / (n * (n - 1))
                
                if pr_local > Metrica_MS:
                    T_best_ms = T_local
                    Metrica_MS = pr_local
                    
                idx_pool += 1
                
            Tiempo_MS = time.time() - tiempo_inicio_ms
            
            # --- 4. REPORTE DE DATOS ---
            resumen_ejecutivo.append({
                'ID': id_real,
                'N': n,
                'Estrategia': estrategia_bl,
                'Metrica_constructivo': round(Metrica_constructiva, 4),
                'Tiempo_constructivo': round(Tiempo_constructivo, 4),
                'Metrica_BL': round(Metrica_BL, 4),
                'Tiempo_BL': round(Tiempo_BL, 4),
                'Metrica_MS': round(Metrica_MS, 4),
                'Tiempo_MS': round(Tiempo_MS, 4)
            })

    # ==========================================
    # EXPORTACIÓN DE ARCHIVOS
    # ==========================================
    carpeta_salida = 'resultados meta'
    os.makedirs(carpeta_salida, exist_ok=True)
    timestamp = datetime.now().strftime('%d%m%y_%H%M')
    
    # 1. Exportar Resumen Ejecutivo (El que pide el profesor)
    df_resumen = pd.DataFrame(resumen_ejecutivo)
    nombre_resumen = os.path.join(carpeta_salida, f'Resumen_MS_{timestamp}.xlsx')
    df_resumen.to_excel(nombre_resumen, index=False)
    print(f"\n✅ Resumen Ejecutivo generado: {nombre_resumen}")
    
    # 2. Exportar Datos Crudos (Para ti)
    cols_crudas = ['ID', 'n', 'mst', 'tiempo', 'iteracion', 'pr_actual', 'mejora', 'pr_best', 'evento']
    df_FI = pd.DataFrame(registros_crudos_FI)
    df_BI = pd.DataFrame(registros_crudos_BI)
    
    if not df_FI.empty:
        df_FI = df_FI[cols_crudas]
        nombre_fi = os.path.join(carpeta_salida, f'Crudo_FI_MS_{timestamp}.xlsx')
        df_FI.to_excel(nombre_fi, index=False)
        print(f"-> Excel crudo FI generado: {nombre_fi}")
        
    if not df_BI.empty:
        df_BI = df_BI[cols_crudas]
        nombre_bi = os.path.join(carpeta_salida, f'Crudo_BI_MS_{timestamp}.xlsx')
        df_BI.to_excel(nombre_bi, index=False)
        print(f"-> Excel crudo BI generado: {nombre_bi}")

    print("=== PIPELINE MAESTRO COMPLETADO ===")
    return df_resumen, df_FI, df_BI



def generar_resumen_exacto_por_id(ruta_archivo_entrada, ruta_archivo_salida=None):
    """
    Genera un resumen analítico por ID extraído de los logs de Búsqueda Local,
    con todas las columnas solicitadas para el reporte final.
    """
    print(f"Leyendo datos desde: {ruta_archivo_entrada}...")
    df = pd.read_excel(ruta_archivo_entrada)
    
    resumen_data = []
    
    grupos_id = df.groupby('ID')
    
    for id_matriz, grupo in grupos_id:
        
        # --- 3. Métrica de la mejor solución tras Búsqueda Local ---
        pr_final_bl = grupo['pr_best'].max()
        
        filas_max_pr = grupo[grupo['pr_best'] == pr_final_bl]
        mejor_fila = filas_max_pr.iloc[0]
        
        # --- 4. Iteración del multistart en la que se encontró (mst) ---
        iteracion_multistart = mejor_fila['mst']
        
        # --- 2. Métrica final etapa constructiva (búsqueda MST + fase ruido) ---
        fila_inicio = grupo[(grupo['mst'] == iteracion_multistart) & (grupo['evento'] == 'Inicio')]
        if not fila_inicio.empty:
            pr_constructiva_ruido = fila_inicio.iloc[0]['pr_actual']
        else:
            pr_constructiva_ruido = grupo[grupo['mst'] == iteracion_multistart].iloc[0]['pr_actual']
            
        # --- 5. Tiempo acumulado hasta la mejor solución ---
        tiempo_acumulado_mejor = mejor_fila['tiempo']
        
        # --- 6. Número de iteraciones LOCALES de la BL en ese multistart ---
        iteraciones_bl_local = mejor_fila['iteracion']
        
        # --- 7. Tiempo total de ejecución por ID ---
        tiempo_total_id = grupo['tiempo'].max()
        
        # --- [NUEVO] 8. Total de iteraciones en todo el ID ---
        # Sumamos el máximo de iteraciones alcanzado por cada MST revisado
        total_iteraciones_id = grupo.groupby('mst')['iteracion'].max().sum()
        
        # --- [NUEVO] 9. Iteraciones acumuladas hasta la mejor solución ---
        # Sumamos todas las iteraciones de los MST anteriores al ganador
        mst_previos = grupo[grupo['mst'] < iteracion_multistart]
        iteraciones_previas = mst_previos.groupby('mst')['iteracion'].max().sum() if not mst_previos.empty else 0
        
        # Y le sumamos las iteraciones que tomó el MST ganador hasta el hallazgo
        iteraciones_globales_hasta_mejor = iteraciones_previas + iteraciones_bl_local
        
        # Consolidamos las columnas exactas
        resumen_data.append({
            'ID': id_matriz,
            'Metrica_Constructiva_Ruido': pr_constructiva_ruido,
            'Metrica_Mejor_Solucion_BL': pr_final_bl,
            'Iteracion_Multistart': iteracion_multistart,
            'Tiempo_Acumulado_Mejor_Solucion': round(tiempo_acumulado_mejor, 4),
            'Iteraciones_BL_en_Multistart': iteraciones_bl_local,
            'Iteraciones_Globales_Hasta_Mejor': iteraciones_globales_hasta_mejor,
            'Tiempo_Total_ID': round(tiempo_total_id, 4),
            'Total_Iteraciones_ID': total_iteraciones_id
        })
        
    df_resumen = pd.DataFrame(resumen_data)
    
    # Exportación
    if ruta_archivo_salida is None:
        # Extraemos solo el nombre del archivo base sin las carpetas previas
        nombre_base = os.path.basename(ruta_archivo_entrada).replace('.xlsx', '')
        
        # --- NUEVO: Creación/verificación de carpeta ---
        carpeta_salida = 'resultados meta'
        os.makedirs(carpeta_salida, exist_ok=True)
        
        # Generamos la ruta final apuntando a la nueva carpeta
        ruta_archivo_salida = os.path.join(carpeta_salida, f"{nombre_base}_Resumen_Final.xlsx")
        
    df_resumen.to_excel(ruta_archivo_salida, index=False)
    print(f"✅ Resumen generado con éxito en: {ruta_archivo_salida}\n")
    
    return df_resumen



def busqueda_local_BI_tiempo(
        matrices, 
        arboles_base, 
        historiales_fase2, 
        lista_ids, 
        t_ventana, 
        y_limite, 
        j_limite, 
        update_tree=True, 
        categoria="BI"):
    """
    Fase 3: Búsqueda Local Best Improvement con Memoria (Tabu).
    - t_ventana: Tiempo (segundos) dedicado a evaluar el vecindario en cada ronda.
    - y_limite: Rondas máximas sin superar el récord global.
    - j_limite: Rondas máximas consecutivas cayendo en un árbol ya revisado.
    """
    historiales_fase3 = []
    arboles_fase3 = [] # NUEVO: Lista para guardar los árboles finales
    registros_mejoras = []
    
    print(f"\n--- Fase 3: Local Search BI + Memoria [{categoria}] ---")
    print(f"Params: Ventana t={t_ventana}s | Límite Global y={y_limite} | Límite Tabú j={j_limite} | Update={update_tree}")

    for i, D in enumerate(matrices):
        n = len(D)
        id_real = lista_ids[i]
        
        # 1. Rescatar valores base
        max_pr_so_far = historiales_fase2[i][-1]
        historial_actual = [max_pr_so_far]
        
        arbol = arboles_base[i].copy() if arboles_base[i] is not None else None
        if arbol is None:
            historiales_fase3.append(historial_actual)
            arboles_fase3.append(None) # Añadimos None si no hay árbol
            continue
            
        print(f"Procesando ID {id_real}... PR Base: {max_pr_so_far:.4f}")
        start_time_global = time.time()
        
        # 2. Inicializar Contadores, Memoria Tabú y Métricas de Despliegue
        rondas_sin_mejora_global = 0  
        rondas_en_arboles_revisados = 0 
        iteracion_local_total = 0
        hubo_mejora_estricta = False # NUEVO: Bandera para registrar si mejoró
        
        # Nuevos contadores para el despliegue final
        total_ventanas_ejecutadas = 0
        total_vecinos_generados = 0
        total_vecinos_evaluados = 0
        
        # Memoria: Guardamos la "firma" del árbol base
        arboles_revisados = set()
        firma_base = frozenset([tuple(sorted(e)) for e in arbol.edges()])
        arboles_revisados.add(firma_base)
        
        # 3. Bucle Principal de Rondas (Ventanas)
        while rondas_sin_mejora_global < y_limite and rondas_en_arboles_revisados < j_limite:
            total_ventanas_ejecutadas += 1
            window_start = time.time()
            mejor_pr_ronda = -1
            mejor_movimiento_ronda = None
            
            # --- OBTENER VECINDARIO ---
            movimientos_posibles = []
            hojas = [nodo for nodo, grado in arbol.degree() if grado == 1]
            for hoja in hojas:
                v_viejo = list(arbol.neighbors(hoja))[0]
                posibles_conexiones = [nodo for nodo in arbol.nodes() if nodo != hoja and nodo != v_viejo]
                for p in posibles_conexiones:
                    movimientos_posibles.append((hoja, v_viejo, p))
            
            random.shuffle(movimientos_posibles) 
            total_vecinos_generados += len(movimientos_posibles) # Sumamos al total posible
            
            # --- EVALUAR VECINDARIO (Limitado por t_ventana) ---
            for mov in movimientos_posibles:
                if time.time() - window_start >= t_ventana:
                    break # Se acabó el tiempo de la ronda
                
                iteracion_local_total += 1
                total_vecinos_evaluados += 1 # Sumamos a los efectivamente revisados
                hoja, v_viejo, v_nuevo = mov
                
                # Aplicar movimiento temporalmente
                arbol.remove_edge(hoja, v_viejo)
                arbol.add_edge(hoja, v_nuevo)
                
                # Calcular métrica
                val = reconocer_caminos_robinson(arbol, D)[0]
                nuevo_pr = 2 * val / (n * (n - 1))
                
                # Rescatar el mejor de esta ronda
                if nuevo_pr > mejor_pr_ronda:
                    mejor_pr_ronda = nuevo_pr
                    mejor_movimiento_ronda = mov
                
                # Deshacer para seguir evaluando
                arbol.remove_edge(hoja, v_nuevo)
                arbol.add_edge(hoja, v_viejo)
                
                historial_actual.append(max_pr_so_far) 
            
            # --- FIN DE LA RONDA: ANÁLISIS DEL GANADOR ---
            if mejor_movimiento_ronda is not None:
                hoja_ganadora, v_viejo, v_nuevo = mejor_movimiento_ronda
                
                # Aplicar el mejor movimiento al árbol base para extraer su "firma"
                arbol.remove_edge(hoja_ganadora, v_viejo)
                arbol.add_edge(hoja_ganadora, v_nuevo)
                
                firma_nuevo_arbol = frozenset([tuple(sorted(e)) for e in arbol.edges()])
                
                # Criterio J: ¿Ya habíamos estado en esta topología?
                if firma_nuevo_arbol in arboles_revisados:
                    rondas_en_arboles_revisados += 1
                else:
                    rondas_en_arboles_revisados = 0 
                    arboles_revisados.add(firma_nuevo_arbol)
                
                # Criterio Y: ¿Supera nuestro récord histórico global?
                if mejor_pr_ronda > max_pr_so_far:
                    tiempo_hallazgo = time.time() - start_time_global
                    registros_mejoras.append({
                        'Categoría': categoria, 'ID Matriz': id_real, 'Dimensión (n)': n,
                        'Iteración Local': iteracion_local_total, 'Tiempo (s)': round(tiempo_hallazgo, 4),
                        'PR Anterior': max_pr_so_far, 'PR Nuevo': mejor_pr_ronda, 'Mejora': mejor_pr_ronda - max_pr_so_far
                    })
                    
                    max_pr_so_far = mejor_pr_ronda
                    rondas_sin_mejora_global = 0 
                    hubo_mejora_estricta = True # NUEVO: Marcamos que logró mejorar
                    historial_actual[-1] = max_pr_so_far
                else:
                    rondas_sin_mejora_global += 1
                
                # Manejo del "Árbol Inmutable"
                if not update_tree:
                    arbol.remove_edge(hoja_ganadora, v_nuevo)
                    arbol.add_edge(hoja_ganadora, v_viejo)
            else:
                # Si no pudo evaluar ni un solo vecino
                rondas_sin_mejora_global += 1
                rondas_en_arboles_revisados += 1

        # NUEVO: Registro forzado si no hubo mejora
        tiempo_total_bucle = time.time() - start_time_global
        if not hubo_mejora_estricta:
            registros_mejoras.append({
                'Categoría': categoria, 'ID Matriz': id_real, 'Dimensión (n)': n,
                'Iteración Local': iteracion_local_total, 'Tiempo (s)': round(tiempo_total_bucle, 4),
                'PR Anterior': max_pr_so_far, 'PR Nuevo': max_pr_so_far, 'Mejora': 0.0
            })
            
        # NUEVO: Guardamos el árbol final resultante
        arboles_fase3.append(arbol.copy())

        # --- TEXTO DE DESPLIEGUE FINAL POR ID ---
        porcentaje_cobertura = (total_vecinos_evaluados / total_vecinos_generados * 100) if total_vecinos_generados > 0 else 0
        motivo = f"Límite Y ({y_limite} rondas sin mejora)" if rondas_sin_mejora_global >= y_limite else f"Límite J ({j_limite} rondas Tabú)"
        
        print(f"  -> Fin ID {id_real}: Detenido por {motivo}.")
        print(f"     Ventanas ejecutadas: {total_ventanas_ejecutadas}")
        print(f"     Cobertura del vecindario: {porcentaje_cobertura:.2f}% ({total_vecinos_evaluados}/{total_vecinos_generados} evaluados). PR Final: {max_pr_so_far:.4f}")
            
        historiales_fase3.append(historial_actual)

    # Exportación
    df_mejoras = pd.DataFrame(registros_mejoras)
    timestamp = datetime.now().strftime('%d%m%y_%H%M')
    if not df_mejoras.empty:
        df_mejoras.to_excel(f'Mejoras_Fase3_BI_Tabu_{timestamp}.xlsx', index=False)

    # NUEVO RETORNO
    return historiales_fase3, arboles_fase3, df_mejoras


def graficar_evolucion_completa(hist_fase1, hist_fase2, hist_fase3, lista_ids, categoria="General"):
    """
    Grafica la evolución de la métrica PR a lo largo de las 3 fases para cada matriz.
    """
    print(f"--- Generando gráfico de evolución (3 Fases) para {categoria} ---")
    
    # Configuración de estilo
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(21, 20))
    
    # Usaremos una paleta de colores para diferenciar cada matriz
    colores = sns.color_palette("tab10", n_colors=len(lista_ids))
    
    # Variables para saber dónde pintar los fondos (usaremos la primera matriz como referencia de los quiebres)
    # Asumimos que la Fase 1 y 2 tuvieron la misma cantidad de iteraciones para todas las matrices
    quiebre_fase1 = len(hist_fase1[0]) - 1
    quiebre_fase2 = quiebre_fase1 + len(hist_fase2[0]) - 1
    max_iteraciones_totales = 0

    for i, id_matriz in enumerate(lista_ids):
        # 1. Coser las 3 listas evitando la duplicación del punto de conexión
        h1 = hist_fase1[i]
        h2 = hist_fase2[i][1:] if len(hist_fase2[i]) > 1 else []
        h3 = hist_fase3[i][1:] if len(hist_fase3[i]) > 1 else []
        
        y_total = h1 + h2 + h3
        x_total = range(len(y_total))
        
        # Actualizar el máximo para el sombreado del fondo
        if len(y_total) > max_iteraciones_totales:
            max_iteraciones_totales = len(y_total)

        # 2. Graficar la línea de esta matriz
        plt.plot(x_total, y_total, label=f'Matriz ID {id_matriz}', color=colores[i % len(colores)], linewidth=2, alpha=0.8)

    # 3. Añadir fondos de colores para distinguir las fases
    plt.axvspan(0, quiebre_fase1, color='lightblue', alpha=0.3, label='Fase 1: Constructiva (Kruskal)')
    plt.axvspan(quiebre_fase1, quiebre_fase2, color='lightgreen', alpha=0.3, label='Fase 2: Perturbación (Ruido)')
    plt.axvspan(quiebre_fase2, max_iteraciones_totales, color='lightcoral', alpha=0.3, label='Fase 3: Búsqueda Local')

    # 4. Líneas verticales divisorias
    plt.axvline(x=quiebre_fase1, color='gray', linestyle='--', linewidth=1.5)
    plt.axvline(x=quiebre_fase2, color='gray', linestyle='--', linewidth=1.5)

    # 5. Títulos y etiquetas
    plt.title(f"Evolución del Espacio T-Robinson: Constructiva $\\rightarrow$ Ruido $\\rightarrow$ Local Search\nCategoría: {categoria.upper()}", 
              fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Cantidad de Evaluaciones (Iteraciones)", fontsize=12)
    plt.ylabel("Mejor Valor PR Encontrado ($PR_D$)", fontsize=12)
    plt.ylim([-0.0001,1.00001])

    # 6. Ajustar leyenda (separando las matrices de las fases)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Leyenda")
    
    plt.tight_layout()
    
    # 7. Guardar imagen
    timestamp = datetime.now().strftime('%d%m%y_%H%M')
    nombre_archivo = f'Evolucion_Completa_{categoria.upper()}_{timestamp}.png'
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"¡Gráfico guardado como: {nombre_archivo}!")

# =====================================================================
# EJEMPLO DE USO 
# =====================================================================
# graficar_evolucion_completa(
#     hist_fase1 = historial_fase,
#     hist_fase2 = historial_fase2,
#     hist_fase3 = hist_fase3,  # El que te devolvió la función de tiempo
#     lista_ids = range(len(matrices)),
#     categoria = "ALL (First Improvement)"
# )

#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

# def min_w_tree(n, d):
#     """
#     Calcula el Árbol de Peso Mínimo usando el Algoritmo de Prim.
    
#     Parámetros:
#     n (int): Número de vértices.
#     weight_matrix (numpy.ndarray): Matriz (n-1) x (n-1) con los pesos de las aristas.
    
#     Retorna:
#     tuple: (list, networkx.Graph) 
#         - Lista de tuplas (u, v, w) donde u y v son los vértices de una arista y w su peso.
#         - Un objeto networkx.Graph que representa el árbol recubridor mínimo.
#     """
#     # Convertimos la matriz a un formato simétrico n x n
    
#     # Lista para almacenar las aristas del MST
#     mst = []
#     mst_graph = nx.Graph()

    
#     # Cola de prioridad para seleccionar la siguiente arista de menor peso
#     pq = []
    
#     # Inicializamos el algoritmo con el nodo 0
#     visitado = set()
#     visitado.add(0)
    
#     # Insertamos las aristas del nodo inicial en la cola de prioridad
#     for j in range(1, n):
#         heapq.heappush(pq, (d[0, j], 0, j))  # (peso, nodo_origen, nodo_destino)
    
#     while len(visitado) < n:
#         peso, u, v = heapq.heappop(pq)
        
#         if v in visitado:
#             continue
        
#         # Agregar la arista al MST
#         mst.append((u, v, peso))
#         mst_graph.add_edge(u,v,weight=peso)
#         visitado.add(v)
        
#         # Añadir nuevas aristas a la cola de prioridad
#         for w in range(n):
#             if w not in visitado:
#                 heapq.heappush(pq, (d[v, w], v, w))
    
#     return mst, mst_graph

# def exp_PR(n,m,n_disimilitudes, n_experimentos):
#     '''
#     Inputs:
#         n: número de elementos
#         m: rango de las disimilitudes
#         n_disimilitudes: número de disimilitudes que se crearán en cada iteración del experimento
#         n_experimentos: cantidad de árboles aleatorios generados por disimilitud

#     Outputs:

#         casos_interesantes = [[d_j, T_aux, T_min_j, PR_Tmins_j], ...]

#     donde d_j es la j-ésima disimilitud generada, sobre la cual se encontró el conjunto T_aux de árboles recubridores aleatorios. El árbol de peso mínimo es T_min_j y su valor PR_d es PR_Tmins_j.
#     '''


#     casos_interesantes = []
#     PR_Tmins = []
#     # Conjunto de disimilitudes
#     D = [] # todas las disimilitudes
#     D_j = [] # disimilitudes interesantes

#     for j in tqdm(range(n_disimilitudes)):

#         # Calculamos PR de árbol recubridor de peso mínimo
#         d_j = disimilitud(n,m)
#         D.append(d_j)
#         T_min_j = min_w_tree(n,d_j)[1]
#         resultados = metricas(d_j, T_min_j)
#         PR_Tmins.append((T_min_j,resultados[0], resultados[3]))

#         # Calculamos métrica sobre un conjunto de árboles recubridores aleatorios
#         T_aux = []

#         # Iniciamos experimento con disimilitud d_j
#         for i in range(n_experimentos):
#             T_i = nx.random_labeled_tree(n)
#             aux = metricas(d_j,T_i)
#             PR_i = aux[0]
#             uv_i = aux[3]

#             if PR_i > PR_Tmins[j][1]:
#                 D_j.append(d_j)
#                 T_aux.append([T_i, uv_i, PR_i])

#         if len(T_aux)>0:
#             casos_interesantes.append([d_j,T_aux, PR_Tmins[j]])

#         time.sleep(0.05)

#         return casos_interesantes
    
#################################################################################################

# def exp_PR_Rob(n,m,n_disimilitudes, n_experimentos):
#     '''
#     Inputs:
#         n: número de elementos
#         m: rango de las disimilitudes
#         n_disimilitudes: número de disimilitudes Robinson que se crearán en cada iteración del experimento
#         n_experimentos: cantidad de árboles aleatorios generados por disimilitud

#     Outputs:

#         casos_interesantes = [[d_j, T_aux, T_min_j, PR_Tmins_j], ...]

#     donde d_j es la j-ésima disimilitud generada, sobre la cual se encontró el conjunto T_aux de árboles recubridores aleatorios. El árbol de peso mínimo es T_min_j y su valor PR_d es PR_Tmins_j.
#     '''
#     casos_interesantes = []
#     PR_Tmins = []

#     for j in tqdm(range(n_disimilitudes)):

#         # Calculamos PR de árbol recubridor de peso mínimo
#         d_j = disimilitud_Rob(n,m)
#         T_min_j = min_w_tree(n,d_j)[1]
#         resultados = metricas(d_j, T_min_j)
#         PR_Tmins.append((T_min_j, resultados[0], resultados[3]))

#         # Calculamos métrica sobre un conjunto de árboles recubridores aleatorios
#         T_aux = []

#         # Iniciamos experimento con disimilitud d_j
#         for i in range(n_experimentos):
#             T_i = nx.random_labeled_tree(n)
#             aux = metricas(d_j,T_i)
#             PR_i = aux[0]
#             uv_i = aux[3]

#             if PR_i > PR_Tmins[j][1]:
#                 T_aux.append([T_i, uv_i, PR_i])

#         if len(T_aux)>0:
#             casos_interesantes.append([d_j, T_aux, PR_Tmins[j]])

#         time.sleep(0.05)

#         return casos_interesantes

#################################################################################################
##########  ROBINSON DISSIMILARITIES RECOGNITION ##########################
#################################################################################################

def partition(d,x,y):
    Ll = set()
    Lm = set()
    Lr = set()
    Ml = set()
    Mm = set()
    Mr = set()
    Rl = set()
    Rm = set()
    Rr = set()
    X = set()
    Y = set()
    Aol = set()
    Aor = set()
    Aoo = set()
    Aa = set()

    dxy = d[x][y]

    n = np.size(d[0])
    S = set(range(n))
    S = S-{x,y}
    for z in S:
        dzx = dxz = d[x][z]
        dyz = dzy =d[y][z]
        if dzy > max(dzx,dxy):
            if dxz > dxy: Ll.add(z)
            elif dxz == dxy: Lm.add(z)
            elif dxy > dxz: Lr.add()
        elif dxy > max(dxz,dzy):
            if dzy > dzx: Ml.add(z)
            elif dzy == dxz: Mm.add(z)
            elif dxz > dzy: Mr.add(z)
        elif dzx > max(dzy,dxy):
            if dxy > dzy: Rl.add(z)
            elif dxy == dzy: Rm.add(z)
            elif dzy > dxy: Rr.add(z)
        elif dzy == dxy and dxy > dzx: X.add(z)
        elif dzx == dxy and dxy > dyz: Y.add(z)
        elif dyz == dxz and dzx > dxy: Ao.add(z)
        else: Aa.add(z)
    
    if len(Ao)>0:
        dyzl = 0
        for i in Ll:
            if d[i][y] > dyzl:
                dyzl = d[i][y]
                zl = i
        dxzr = 0
        for j in Rr:
            if d[x][j] > dxzr:
                dxzr = d[x][j]
                zr = j
        for k in Ao:
            if d[y][k] < dyzl and d[zl][k] <= d[zl][x]: Aol.add(z)
            elif d[x][k] < dxzr and d[zr][z] <= d[zr][y]: Aor.add(z)
            else: Aoo.add(z)

    L = [Ll, Lm, Lr]
    M = [Ml, Mm, Mr]
    R = [Rl, Rm, Rr]
    Ao = [(Aol,zl), (Aor,zr), Aoo]
    return L, M, R, X, Y, Ao, Aa 

##################################################################################################

def internal_contradiction(d,x,y):
    L, M, R, X, Y, Ao, Aa = partition(d,x,y)
    cond1 = len(L[2])>0 and len(M)>0 and len(R[0])>0 and len(Aa)>0
    cond2 = len(Ao[0][0].intersection(Ao[1][0]))>0
    if cond1 or cond2: return True
    A0 = Ao[0][0].union(Ao[1][0], Ao[2])
    for z in A0:
        cond3 = d[y][z] < d[y][Ao[0][1]] and d[Ao[0][1]][x] < d[Ao[0][1]][z] and d[Ao[0][1]][z] < d[Ao[0][1]][y]
        cond4 = d[x][z] < d[y][Ao[1][1]] and d[Ao[1][1]][y] < d[Ao[1][1]][z] and d[Ao[1][1]][z] < d[Ao[1][1]][x]
        if cond3 or cond4: return True
    return False

#################################################################################################

def UniversalPQTree(d):
    S = range(len(d[0]))
    T = list(itertools.permutations(S))
    return T

#################################################################################################

# def PQTreeUpdate(d,T,L) :
#     T_i = UniversalPQTree(d)

#################################################################################################



#################################################################################################