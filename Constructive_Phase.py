import networkx as nx # type: ignore
import itertools
import numpy as np# type: ignore
import random
import heapq
import time
import os
from datetime import datetime
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore


import os
import pandas as pd # type: ignore

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

def es_camino_robinson(camino, D):
    '''
    Dado un espacio de disimilitud (X,D), y un camino x1x2...x(n-1)xn dentro de un árbol tal que todos sus subcaminos son Robinson. Esta función retorna 'True' si el camino dado es Robinson bajo D. 'False' en otro caso.
    '''
    x1, x2, x3, x4 = camino[0], camino[1], camino[-2], camino[-1]
    if D[x1,x4] < max(D[x1,x3],D[x2,x4]):
        return False
    return True


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

def MST(D):
    '''
    Dada una disimilitud D, MST(D) devuelve el árbol recubridor de peso mínimo del espacio de disimilitud dado
    '''
    G = Kn(len(D),D)
    return nx.minimum_spanning_tree(G, weight='weight')


def buscar_mst_distintos(D, tiempo_limite_segundos):
    """
    Busca MSTs exactos distintos, cortando ABRUPTAMENTE cuando se agota el tiempo.
    El árbol que quede a medio construir se descarta.
    """
    start_time = time.time()
    n = len(D)
    
    aristas = [(i, j, D[i][j]) for i in range(n) for j in range(i + 1, n)]
    aristas.sort(key=lambda x: x[2])
    
    grupos_por_peso = [list(grupo) for peso, grupo in itertools.groupby(aristas, key=lambda x: x[2])]
        
    mst_encontrados_hashes = set() 
    lista_grafos_nx = []           
    
    # El reloj gobierna el bucle principal
    while time.time() - start_time < tiempo_limite_segundos:
        uf = UnionFind(n) 
        aristas_mst_actual = []
        n_aristas_agregadas = 0
        
        for grupo in grupos_por_peso:
            grupo_actual = grupo[:]
            if len(grupo_actual) > 1:
                random.shuffle(grupo_actual)
            
            for u, v, w in grupo_actual:
                if uf.union(u, v):
                    aristas_mst_actual.append(tuple(sorted((u, v))))
                    n_aristas_agregadas += 1
            
            if n_aristas_agregadas == n - 1:
                break
                
        # Si el tiempo se acabó a mitad del for anterior, este árbol no sirve.
        # Cortamos antes de guardarlo.
        if time.time() - start_time >= tiempo_limite_segundos:
            break
            
        arbol_hash = frozenset(aristas_mst_actual)
        
        if arbol_hash not in mst_encontrados_hashes:
            mst_encontrados_hashes.add(arbol_hash)
            T = nx.Graph()
            T.add_nodes_from(range(n)) 
            for u, v in arbol_hash:
                T.add_edge(u, v, weight=D[u][v])
            lista_grafos_nx.append(T)

    return len(lista_grafos_nx), lista_grafos_nx


# def ConstructionPhase(D, t_mst, n_pert, epsilon, distribucion='normal'):
#     """
#     Fase Constructiva estricta. 
#     Tope de tiempo global: 20 * t_mst (Evaluación + Ruido).
#     Los árboles no evaluados a tiempo no entran al pool.
#     """
#     start_global = time.time()
#     limite_fase_constructiva = 20 * t_mst # <--- Aquí controlas la guillotina global
#     n = len(D)
    
#     pool_hashes = set()
#     arboles_con_metricas = []

#     # ==========================================
#     # ETAPA 1: MSTs Exactos (Tope: t_mst)
#     # ==========================================
#     start_mst = time.time()
#     _, trees_found = buscar_mst_distintos(D, t_mst)
    
#     if not trees_found:
#         trees_found = [MST(D)]
#     tiempo_empleado_mst = time.time() - start_mst
        
#     # Evaluar los MSTs exactos 
#     mejor_pr_mst = 0.0
#     for T in trees_found:
#         if time.time() - start_global >= limite_fase_constructiva:
#             break # Guillotina: se acabó el tiempo global
            
#         aristas_hash = frozenset([tuple(sorted((u, v))) for u, v in T.edges()])
#         if aristas_hash not in pool_hashes:
#             pool_hashes.add(aristas_hash)
            
#             val = reconocer_caminos_robinson(T, D)[0]
#             pr_score = 2 * val / (n * (n - 1))
#             arboles_con_metricas.append((T, pr_score))
            
#             if pr_score > mejor_pr_mst:
#                 mejor_pr_mst = pr_score

#     # ==========================================
#     # ETAPA 2: Ruido y Evaluación Simultánea
#     # ==========================================
#     start_ruido = time.time()
#     iteraciones_ruido_completadas = 0
#     mejor_pr_ruido = mejor_pr_mst
    
#     for _ in range(n_pert):
#         if time.time() - start_global >= limite_fase_constructiva:
#             break # Guillotina
            
#         iteraciones_ruido_completadas += 1
        
#         # 1. Aplicar ruido
#         if distribucion == 'normal':
#             R = np.random.normal(0, 1, size=(n, n))
#         else:
#             R = np.random.uniform(-1, 1, size=(n, n))
            
#         R = (R + R.T) / 2.0
#         np.fill_diagonal(R, 0)
        
#         D_noise = np.abs(D + epsilon * R)
#         T_stocastico = MST(D_noise)
        
#         # 2. Guillotina previa a la evaluación (que es lo más costoso)
#         if time.time() - start_global >= limite_fase_constructiva:
#             break 
            
#         aristas_hash = frozenset([tuple(sorted((u, v))) for u, v in T_stocastico.edges()])
#         if aristas_hash not in pool_hashes:
#             pool_hashes.add(aristas_hash)
            
#             # 3. Evaluar
#             val = reconocer_caminos_robinson(T_stocastico, D)[0]
#             pr_score = 2 * val / (n * (n - 1))
#             arboles_con_metricas.append((T_stocastico, pr_score))
            
#             if pr_score > mejor_pr_ruido:
#                 mejor_pr_ruido = pr_score

#     tiempo_empleado_ruido_eval = time.time() - start_ruido

#     # ==========================================
#     # ETAPA 3: Ordenar y Reportar
#     # ==========================================
#     arboles_con_metricas.sort(key=lambda x: x[1], reverse=True)
    
#     # Seguro por si la guillotina cortó todo antes de guardar el primer árbol
#     if not arboles_con_metricas:
#         T_fallback = MST(D)
#         val = reconocer_caminos_robinson(T_fallback, D)[0]
#         arboles_con_metricas.append((T_fallback, 2 * val / (n * (n - 1))))
        
#     T_pool = [item[0] for item in arboles_con_metricas]
#     metricas_pool = [item[1] for item in arboles_con_metricas]
    
#     # Mantenemos el mini-reporte para tu Excel
#     reporte_fase_1_y_2 = {
#         'Cant_MST_Encontrados': len(trees_found),
#         'Tiempo_MST_s': round(tiempo_empleado_mst, 4),
#         'Mejor_Metrica_MST': round(mejor_pr_mst, 4),
#         'Cant_Iteraciones_Ruido': iteraciones_ruido_completadas, # Solo las que logró terminar
#         'Tiempo_Ruido_y_Eval_s': round(tiempo_empleado_ruido_eval, 4),
#         'Mejor_Metrica_Ruido': round(mejor_pr_ruido, 4)
#     }
    
#     return T_pool, metricas_pool, reporte_fase_1_y_2


def ConstructionPhase(D, t_mst, n_pert, epsilon, distribucion='normal'):
    """
    Fase Constructiva On-The-Fly.
    1. Genera un MST exacto -> Verifica si es único -> Lo evalúa -> Repite hasta t_mst.
    2. Genera ruido -> Extrae MST -> Verifica si es único -> Lo evalúa -> Repite n_pert veces.
    * Límite global de seguridad (Guillotina): 20 * t_mst
    """
    start_global = time.time()
    limite_fase_constructiva = 20 * t_mst 
    n = len(D)
    
    pool_hashes = set()
    arboles_con_metricas = []

    # ==========================================
    # ETAPA 1: BÚSQUEDA Y EVALUACIÓN DE MSTs EXACTOS
    # ==========================================
    start_mst = time.time()
    msts_exactos_evaluados = 0
    mejor_pr_mst = 0.0
    
    # Preparación de aristas para Kruskal
    aristas = [(i, j, D[i][j]) for i in range(n) for j in range(i + 1, n)]
    aristas.sort(key=lambda x: x[2])
    grupos_por_peso = [list(grupo) for peso, grupo in itertools.groupby(aristas, key=lambda x: x[2])]
    
    # El reloj t_mst gobierna todo el ciclo (Construcción + Evaluación)
    while (time.time() - start_mst < t_mst) and (time.time() - start_global < limite_fase_constructiva):
        uf = UnionFind(n) 
        aristas_mst_actual = []
        n_aristas_agregadas = 0
        
        # 1. Construir un árbol con Kruskal aleatorizado
        for grupo in grupos_por_peso:
            grupo_actual = grupo[:]
            if len(grupo_actual) > 1:
                random.shuffle(grupo_actual)
            
            for u, v, w in grupo_actual:
                if uf.union(u, v):
                    aristas_mst_actual.append(tuple(sorted((u, v))))
                    n_aristas_agregadas += 1
            
            if n_aristas_agregadas == n - 1:
                break
                
        # Corte abrupto: Si el tiempo se acabó armando el árbol, lo descartamos
        if (time.time() - start_mst >= t_mst) or (time.time() - start_global >= limite_fase_constructiva):
            break
            
        arbol_hash = frozenset(aristas_mst_actual)
        
        # 2. FILTRO DE UNICIDAD: Solo evaluamos si el árbol es 100% nuevo
        if arbol_hash not in pool_hashes:
            pool_hashes.add(arbol_hash)
            
            # Instanciar el grafo
            T = nx.Graph()
            T.add_nodes_from(range(n)) 
            for u, v in arbol_hash:
                T.add_edge(u, v, weight=D[u][v])
                
            # 3. Evaluar Métrica PR
            val = reconocer_caminos_robinson(T, D)[0]
            pr_score = 2 * val / (n * (n - 1))
            
            arboles_con_metricas.append((T, pr_score))
            msts_exactos_evaluados += 1
            
            if pr_score > mejor_pr_mst:
                mejor_pr_mst = pr_score

    tiempo_empleado_mst = time.time() - start_mst
    
    # Fallback de seguridad: Si t_mst era tan corto que no alcanzó a evaluar ni 1 árbol,
    # forzamos la creación del MST estándar para no romper el algoritmo.
    if msts_exactos_evaluados == 0:
        T_fallback = MST(D)
        arbol_hash = frozenset([tuple(sorted((u, v))) for u, v in T_fallback.edges()])
        pool_hashes.add(arbol_hash)
        val = reconocer_caminos_robinson(T_fallback, D)[0]
        mejor_pr_mst = 2 * val / (n * (n - 1))
        arboles_con_metricas.append((T_fallback, mejor_pr_mst))
        msts_exactos_evaluados = 1

    # ==========================================
    # ETAPA 2: APLICACIÓN DE RUIDO Y EVALUACIÓN
    # ==========================================
    start_ruido = time.time()
    iteraciones_ruido_completadas = 0
    mejor_pr_ruido = mejor_pr_mst
    
    for _ in range(n_pert):
        # Guillotina global (20 * t_mst)
        if time.time() - start_global >= limite_fase_constructiva:
            break 
            
        iteraciones_ruido_completadas += 1
        
        # 1. Aplicar ruido a la matriz original
        if distribucion == 'normal':
            R = np.random.normal(0, 1, size=(n, n))
        else:
            R = np.random.uniform(-1, 1, size=(n, n))
            
        R = (R + R.T) / 2.0
        np.fill_diagonal(R, 0)
        
        D_noise = np.abs(D + epsilon * R)
        
        # 2. Extraer MST de la matriz ruidosa
        T_stocastico = MST(D_noise)
        
        # Guillotina previa a la evaluación (lo más costoso)
        if time.time() - start_global >= limite_fase_constructiva:
            break 
            
        # 3. FILTRO DE UNICIDAD: Verificamos si el ruido generó un árbol que ya teníamos
        arbol_hash = frozenset([tuple(sorted((u, v))) for u, v in T_stocastico.edges()])
        
        if arbol_hash not in pool_hashes:
            pool_hashes.add(arbol_hash)
            
            # 4. Evaluar el nuevo árbol
            val = reconocer_caminos_robinson(T_stocastico, D)[0]
            pr_score = 2 * val / (n * (n - 1))
            arboles_con_metricas.append((T_stocastico, pr_score))
            
            if pr_score > mejor_pr_ruido:
                mejor_pr_ruido = pr_score

    tiempo_empleado_ruido_eval = time.time() - start_ruido

    # ==========================================
    # ETAPA 3: ORDENAMIENTO Y CIERRE
    # ==========================================
    # Ordenamos todo el pool combinado (Exactos + Ruido) de mayor a menor PR
    arboles_con_metricas.sort(key=lambda x: x[1], reverse=True)
    
    T_pool = [item[0] for item in arboles_con_metricas]
    metricas_pool = [item[1] for item in arboles_con_metricas]
    
    # Mini-reporte actualizado para tu Excel
    reporte_fase_1_y_2 = {
        'Cant_MST_Encontrados': msts_exactos_evaluados,
        'Tiempo_MST_s': round(tiempo_empleado_mst, 4),
        'Mejor_Metrica_MST': round(mejor_pr_mst, 4),
        'Cant_Iteraciones_Ruido': iteraciones_ruido_completadas, 
        'Tiempo_Ruido_y_Eval_s': round(tiempo_empleado_ruido_eval, 4),
        'Mejor_Metrica_Ruido': round(mejor_pr_ruido, 4)
    }
    
    return T_pool, metricas_pool, reporte_fase_1_y_2