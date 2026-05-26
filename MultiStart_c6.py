# MultiStart_c6.py
from LectorCarpetas import procesar_data_test_exp_meta # (Usa el nombre de tu archivo lector)
from MultiStart import pipeline_maestro_final

print("=== LANZANDO LOTE 6 (Carpeta c6) ===")

# 1. Leer carpeta 6
matrices_c6, ids_c6 = procesar_data_test_exp_meta(n_carpeta=6)

# 2. Ejecutar experimento
# Ajusta tus parámetros t_max, t_mst, etc., según tu diseño
resumen, prep, crudo_fi, crudo_bi = pipeline_maestro_final(
    matrices=matrices_c6, 
    lista_ids=ids_c6, 
    t_max=600,      
    t_mst=10,         
    n_pert=200,       
    epsilon=0.25
)
print("=== LOTE 6 COMPLETADO ===")