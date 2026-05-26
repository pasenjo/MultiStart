# MultiStart_c5.py
from LectorCarpetas import procesar_data_test_exp_meta # (Usa el nombre de tu archivo lector)
from MultiStart import pipeline_maestro_final

print("=== LANZANDO LOTE 5 (Carpeta c5) ===")

# 1. Leer carpeta 5
matrices_c5, ids_c5 = procesar_data_test_exp_meta(n_carpeta=5)

# 2. Ejecutar experimento
# Ajusta tus parámetros t_max, t_mst, etc., según tu diseño
resumen, prep, crudo_fi, crudo_bi = pipeline_maestro_final(
    matrices=matrices_c5, 
    lista_ids=ids_c5, 
    t_max=600,      
    t_mst=10,         
    n_pert=200,       
    epsilon=0.25
)
print("=== LOTE 5 COMPLETADO ===")