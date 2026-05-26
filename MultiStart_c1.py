# MultiStart_c1.py
from MultiStart import pipeline_maestro_final
from QuasiRobinson import procesar_benchmark_a_df

print("=== LANZANDO LOTE 1 (Carpeta c1) ===")

# 1. Leer carpeta 1
carpeta = 'c1'

matrices, df = procesar_benchmark_a_df(carpeta, t=0.0001) 

# 2. Ejecutar experimento
# Ajusta tus parámetros t_max, t_mst, etc., según tu diseño
resumen, prep, crudo_fi, crudo_bi = pipeline_maestro_final(
    matrices=matrices, 
    lista_ids=range(len(matrices)), 
    t_max=600,      
    t_mst=10,         
    n_pert=200,       
    epsilon=0.25
)
print("=== LOTE 1 COMPLETADO ===")