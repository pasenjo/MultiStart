import os
import pandas as pd #type: ignore
from datetime import datetime

def consolidar_resumenes_ejecutivos(carpeta_resultados='resultados meta'):
    """
    Busca todos los archivos de Resumen Ejecutivo parciales,
    los concatena y genera un Master Excel ordenado por ID.
    """
    print(f"Buscando archivos en la carpeta '{carpeta_resultados}'...")
    
    # 1. Encontrar todos los Excel de Resumen Ejecutivo
    archivos_resumen = []
    try:
        for f in os.listdir(carpeta_resultados):
            # Filtramos solo los que nos interesan
            if f.startswith('Resumen_Ejecutivo_') and f.endswith('.xlsx'):
                archivos_resumen.append(os.path.join(carpeta_resultados, f))
    except FileNotFoundError:
        print(f"❌ Error: La carpeta '{carpeta_resultados}' no existe.")
        return

    if not archivos_resumen:
        print("⚠️ No se encontraron archivos 'Resumen_Ejecutivo' para consolidar.")
        return

    print(f"Se encontraron {len(archivos_resumen)} resúmenes. Consolidando...")

    # 2. Leer y concatenar
    lista_dataframes = []
    for archivo in archivos_resumen:
        try:
            df_parcial = pd.read_excel(archivo)
            lista_dataframes.append(df_parcial)
            print(f"  -> Leído: {os.path.basename(archivo)} ({len(df_parcial)} filas)")
        except Exception as e:
            print(f"  -> Error leyendo {archivo}: {e}")

    df_master = pd.concat(lista_dataframes, ignore_index=True)

    # 3. Ordenar lógicamente
    # Primero agrupamos por Estrategia (para que todo FI quede junto y luego todo BI),
    # y luego ordenamos numéricamente por el ID (1, 2, 3... 120)
    df_master.sort_values(by=['Estrategia', 'ID'], inplace=True)
    
    # Reiniciamos el índice de Pandas para que quede limpio (0, 1, 2...)
    df_master.reset_index(drop=True, inplace=True)

    # 4. Exportar el Excel Master
    timestamp = datetime.now().strftime('%d%m%y_%H%M')
    nombre_salida = os.path.join(carpeta_resultados, f'MASTER_Resumen_Ejecutivo_{timestamp}.xlsx')
    
    df_master.to_excel(nombre_salida, index=False)
    
    print("\n✅ ¡CONSOLIDACIÓN EXITOSA!")
    print(f"📊 Instancias totales procesadas: {len(df_master)}")
    print(f"📁 Archivo Master guardado en: {nombre_salida}")

# Ejecutar la función
if __name__ == "__main__":
    consolidar_resumenes_ejecutivos()