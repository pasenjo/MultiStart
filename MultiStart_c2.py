# MultiStart_c2.py
from LectorCarpetas import procesar_data_test_exp_meta # (Usa el nombre de tu archivo lector)
from MultiStart import pipeline_maestro_final

print("=== LANZANDO LOTE 2 (Carpeta c2) ===")

# 1. Leer carpeta 2
matrices_c2, ids_c2 = procesar_data_test_exp_meta(n_carpeta=2)

# 2. Ejecutar experimento
# Ajusta tus parámetros t_max, t_mst, etc., según tu diseño
resumen, prep, crudo_fi, crudo_bi = pipeline_maestro_final(
    matrices=matrices_c2, 
    lista_ids=ids_c2, 
    t_max=600,      
    t_mst=10,         
    n_pert=200,       
    epsilon=0.25
)
print("=== LOTE 2 COMPLETADO ===")