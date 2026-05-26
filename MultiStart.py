import os
import time
import pandas as pd # type: ignore
import numpy as np # type: ignore
from datetime import datetime

# Importamos tu librería principal
import QuasiRobinson as qr

# Importamos el módulo constructivo que creamos
from Constructive_Phase import ConstructionPhase

def pipeline_maestro_final(matrices, lista_ids, t_max, t_mst, n_pert, epsilon=0.25, distribucion='normal'):
    """
    Ejecuta el Framework Multistart completo.
    - Tiempo_BL: Fotografía tras revisar TODO el pool inicial.
    - Tiempo_MS: Fotografía final tras gastar el tiempo sobrante en árboles aleatorios.
    """
    resumen_ejecutivo = []
    reporte_preparacion = []
    registros_crudos_FI = []
    registros_crudos_BI = []
    
    print(f"=== INICIANDO PIPELINE MAESTRO MULTISTART ===")
    print(f"Límite global (t_max): {t_max}s | t_mst: {t_mst}s | n_pert: {n_pert} | eps: {epsilon}\n")

    for i, D in enumerate(matrices):
        id_real = lista_ids[i]
        n = len(D)
        
        print(f"[{id_real}] Iniciando procesamiento (n={n})")
        
        # --- 1. FASE CONSTRUCTIVA ÚNICA POR ID ---
        print("  -> Construyendo pool inicial (se compartirá entre estrategias)...")
        tiempo_inicio_constructivo = time.time()
        
        T_pool_original, metricas_pool_original, reporte_f1_f2 = ConstructionPhase(D, t_mst, n_pert, epsilon, distribucion)
        
        t_constructivo_real = time.time() - tiempo_inicio_constructivo
        Metrica_constructiva = metricas_pool_original[0]
        
        for estrategia_bl in ["First_Improvement", "Best_Improvement"]:
            print(f"  -> Ejecutando estrategia: {estrategia_bl}")
            
            datos_preparacion = {'ID': id_real, 'n': n, 'Estrategia': estrategia_bl}
            datos_preparacion.update(reporte_f1_f2) 
            reporte_preparacion.append(datos_preparacion)
            
            # Simulamos que el MS global empezó hace exactamente 't_constructivo_real' segundos.
            tiempo_inicio_ms = time.time() - t_constructivo_real
            
            T_pool = T_pool_original.copy()
            metricas_pool = metricas_pool_original.copy()
            
            # --- 2. FASE BÚSQUEDA LOCAL (Revisión de TODO el pool inicial) ---
            Metrica_BL = Metrica_constructiva
            T_best_bl = T_pool[0]
            idx_pool = 0 
            
            while (time.time() - tiempo_inicio_ms < t_max) and (idx_pool < len(T_pool)):
                T_init = T_pool[idx_pool]
                Metrica_base_local = metricas_pool[idx_pool]
                
                t_restante_ms = t_max - (time.time() - tiempo_inicio_ms)
                if t_restante_ms <= 0: break
                    
                tiempo_previo_ms = time.time() - tiempo_inicio_ms
                
                if estrategia_bl == "First_Improvement":
                    T_local, reg_local, _ = qr.ejecutar_FI_estricto(D, T_init, id_real, idx_pool+1, Metrica_base_local, t_restante_ms, tiempo_previo_ms)
                    registros_crudos_FI.extend(reg_local)
                else:
                    T_local, reg_local, _ = qr.ejecutar_BI_estricto(D, T_init, id_real, idx_pool+1, Metrica_base_local, t_restante_ms, tiempo_previo_ms)
                    registros_crudos_BI.extend(reg_local)
                
                val_local = qr.reconocer_caminos_robinson(T_local, D)[0]
                pr_local = 2 * val_local / (n * (n - 1))
                
                # Actualizamos el récord de la fase BL
                if pr_local > Metrica_BL:
                    T_best_bl = T_local
                    Metrica_BL = pr_local
                    
                idx_pool += 1

            # == FOTOGRAFÍA 1: Fin de la revisión del Pool inicial ==
            Tiempo_BL = time.time() - tiempo_inicio_ms
            
            # Inicializamos las variables del MS Global con lo mejor que encontramos en el pool
            Metrica_MS = Metrica_BL
            T_best_ms = T_best_bl

            # --- 3. FASE DE EXPLORACIÓN EXTENDIDA (Si sobra tiempo, seguimos con ruido) ---
            while time.time() - tiempo_inicio_ms < t_max:
                
                if distribucion == 'normal':
                    R = np.random.normal(0, 1, size=(n, n))
                else:
                    R = np.random.uniform(-1, 1, size=(n, n))
                    
                R = (R + R.T) / 2.0
                np.fill_diagonal(R, 0)
                
                D_noise = np.abs(D + epsilon * R)
                T_extra = qr.MST(D_noise) 
                
                val_base_extra = qr.reconocer_caminos_robinson(T_extra, D)[0]
                pr_base_extra = 2 * val_base_extra / (n * (n - 1))
                
                t_restante_ext = t_max - (time.time() - tiempo_inicio_ms)
                if t_restante_ext <= 0: break
                
                tiempo_previo_ext = time.time() - tiempo_inicio_ms
                
                if estrategia_bl == "First_Improvement":
                    T_local_ext, reg_local_ext, _ = qr.ejecutar_FI_estricto(D, T_extra, id_real, idx_pool+1, pr_base_extra, t_restante_ext, tiempo_previo_ext)
                    registros_crudos_FI.extend(reg_local_ext)
                else:
                    T_local_ext, reg_local_ext, _ = qr.ejecutar_BI_estricto(D, T_extra, id_real, idx_pool+1, pr_base_extra, t_restante_ext, tiempo_previo_ext)
                    registros_crudos_BI.extend(reg_local_ext)
                
                val_local_ext = qr.reconocer_caminos_robinson(T_local_ext, D)[0]
                pr_local_ext = 2 * val_local_ext / (n * (n - 1))
                
                # Actualizamos el récord global
                if pr_local_ext > Metrica_MS:
                    T_best_ms = T_local_ext
                    Metrica_MS = pr_local_ext
                    
                idx_pool += 1
                
            # == FOTOGRAFÍA 2: Fin absoluto (t_max alcanzado) ==
            Tiempo_MS = time.time() - tiempo_inicio_ms
            
            # --- 4. REPORTE DE DATOS MS ---
            resumen_ejecutivo.append({
                'ID': id_real,
                'N': n,
                'Estrategia': estrategia_bl,
                'Metrica_constructivo': round(Metrica_constructiva, 4),
                'Tiempo_constructivo': round(t_constructivo_real, 4),
                'Metrica_BL': round(Metrica_BL, 4),
                'Tiempo_BL': round(Tiempo_BL, 4),
                'Metrica_MS': round(Metrica_MS, 4),
                'Tiempo_MS': round(Tiempo_MS, 4)
            })

    # ==========================================
    # EXPORTACIÓN DE LOS 4 ARCHIVOS EXCEL
    # ==========================================
    carpeta_salida = 'resultados meta NEW'
    os.makedirs(carpeta_salida, exist_ok=True)
    timestamp = datetime.now().strftime('%d%m%y_%H%M')
    
    df_resumen = pd.DataFrame(resumen_ejecutivo)
    nombre_resumen = os.path.join(carpeta_salida, f'Resumen_Ejecutivo_{timestamp}.xlsx')
    df_resumen.to_excel(nombre_resumen, index=False)
    print(f"\n✅ Resumen Ejecutivo generado en: {nombre_resumen}")
    
    df_prep = pd.DataFrame(reporte_preparacion)
    nombre_prep = os.path.join(carpeta_salida, f'Resumen_Fase1_y_Ruido_{timestamp}.xlsx')
    df_prep.to_excel(nombre_prep, index=False)
    print(f"✅ Resumen MST+Ruido generado en: {nombre_prep}")
    
    cols_crudas = ['ID', 'n', 'mst', 'tiempo', 'iteracion', 'pr_actual', 'mejora', 'pr_best', 'evento']
    df_FI = pd.DataFrame(registros_crudos_FI)
    df_BI = pd.DataFrame(registros_crudos_BI)
    
    if not df_FI.empty:
        df_FI = df_FI[cols_crudas]
        nombre_fi = os.path.join(carpeta_salida, f'Crudo_FI_{timestamp}.xlsx')
        df_FI.to_excel(nombre_fi, index=False)
        print(f"✅ Datos Crudos FI generados en: {nombre_fi}")
        
    if not df_BI.empty:
        df_BI = df_BI[cols_crudas]
        nombre_bi = os.path.join(carpeta_salida, f'Crudo_BI_{timestamp}.xlsx')
        df_BI.to_excel(nombre_bi, index=False)
        print(f"✅ Datos Crudos BI generados en: {nombre_bi}")

    print("\n=== PIPELINE MAESTRO COMPLETADO CON ÉXITO ===")
    return df_resumen, df_prep, df_FI, df_BI