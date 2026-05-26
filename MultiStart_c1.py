# MultiStart_c1.py
from LectorCarpetas import procesar_data_test_exp_meta # (Usa el nombre de tu archivo lector)
from MultiStart import pipeline_maestro_final

print("=== LANZANDO LOTE 1 (Carpeta c1) ===")

# 1. Leer carpeta 1
matrices_c1, ids_c1 = procesar_data_test_exp_meta(n_carpeta=1)

# 2. Ejecutar experimento
# Ajusta tus parámetros t_max, t_mst, etc., según tu diseño
resumen, prep, crudo_fi, crudo_bi = pipeline_maestro_final(
    matrices=matrices_c1, 
    lista_ids=ids_c1, 
    t_max=600,      
    t_mst=10,         
    n_pert=200,       
    epsilon=0.25
)
print("=== LOTE 1 COMPLETADO ===")