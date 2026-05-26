# LectorCarpetas.py
import os
import numpy as np # type:ignore

def procesar_data_test_exp_meta(n_carpeta):
    """
    Lee archivos .txt de la carpeta 'c{n_carpeta}' y retorna la lista de matrices
    junto con la lista de sus IDs respectivos (números de los archivos).

    Args:
        n_carpeta (int/str): Número de la carpeta a leer (ej. 1 para 'c1').

    Returns:
        tuple: (lista_matrices, lista_ids)
    """
    nombre_carpeta = f"c{n_carpeta}"
    lista_matrices = []
    lista_ids = []  # <--- NUEVA LISTA

    # 1. Obtener y ordenar archivos numéricamente
    try:
        archivos = [f for f in os.listdir(nombre_carpeta) if f.endswith('.txt')]
        archivos.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except FileNotFoundError:
        print(f"Error: La carpeta '{nombre_carpeta}' no existe en el directorio actual.")
        return [], []  # Retornamos dos listas vacías si falla

    print(f"Procesando {len(archivos)} archivos en la carpeta '{nombre_carpeta}'...")

    for archivo in archivos:
        ruta_completa = os.path.join(nombre_carpeta, archivo)
        
        try:
            # 2. Leer Header (Tipo) y Matriz
            with open(ruta_completa, 'r', encoding='utf-8') as f:
                _ = f.readline().strip() 
                D = np.loadtxt(f)

            # 3. Guardar matriz e ID
            lista_matrices.append(D)
            
            # Extraemos el número del archivo (ej: de "1.txt" extrae el int 1)
            id_archivo = int(os.path.splitext(archivo)[0])
            lista_ids.append(id_archivo)

        except Exception as e:
            print(f"Error procesando archivo {archivo}: {e}")

    # Retornamos ambas listas
    return lista_matrices, lista_ids