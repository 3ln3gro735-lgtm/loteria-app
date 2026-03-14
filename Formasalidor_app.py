# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import time 
from collections import defaultdict, Counter
import unicodedata

# --- CONFIGURACION DE LA RUTA ---
RUTA_CSV = 'Flotodo.csv'
RUTA_CACHE = 'cache_perfiles_florida.csv'

# --- CONFIGURACION DE LA PAGINA ---
st.set_page_config(
    page_title="Florida - Análisis de Sorteos",
    page_icon="🌴",
    layout="wide"
)

st.title("🌴 Florida - Análisis de Sorteos")
st.markdown("Visualización de Estado Actual (Solo Tarde y Noche).")

# --- FUNCIONES AUXILIARES Y DE CARGA ---

def remove_accents(input_str):
    if not isinstance(input_str, str): return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def inicializar_archivo(ruta, columnas):
    if not os.path.exists(ruta):
        try:
            with open(ruta, 'w', encoding='latin-1') as f:
                f.write(";".join(columnas) + "\n")
        except Exception as e:
            st.error(f"Error inicializando {ruta}: {e}")

@st.cache_data(ttl="10m") 
def cargar_datos_flotodo(_ruta_csv):
    try:
        inicializar_archivo(_ruta_csv, ["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"])
        
        try:
            df = pd.read_csv(_ruta_csv, sep=';', encoding='latin-1', header=0, on_bad_lines='skip', dtype=str)
        except Exception as e:
            st.error(f"Error leyendo CSV: {e}")
            return pd.DataFrame()

        if df.empty: return pd.DataFrame()
        
        df.columns = [str(c).strip() for c in df.columns]
        
        rename_map = {}
        for col in df.columns:
            c = str(col).strip()
            if 'Fecha' in c: rename_map[col] = 'Fecha'
            elif 'Noche' in c or 'Tarde' in c: rename_map[col] = 'Tipo_Sorteo'
            elif 'Centena' in c: rename_map[col] = 'Centena'
            elif 'Fijo' in c and 'Corrido' not in c: rename_map[col] = 'Fijo'
            elif '1er' in c or 'Primer' in c: rename_map[col] = 'Primer_Corrido'
            elif '2do' in c or 'Segundo' in c: rename_map[col] = 'Segundo_Corrido'
        
        df.rename(columns=rename_map, inplace=True)
        
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Fecha'], inplace=True)
        
        # Mapeo solo para Tarde y Noche
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper().map({
            'TARDE': 'T', 'T': 'T', 
            'NOCHE': 'N', 'N': 'N'
        }).fillna('OTRO')
        df = df[df['Tipo_Sorteo'].isin(['T', 'N'])].copy()
        
        for col in ['Centena', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido']:
            if col not in df.columns: df[col] = '0'
            df[col] = df[col].replace('', '0').fillna('0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df_long = df.melt(id_vars=['Fecha', 'Tipo_Sorteo'], 
                          value_vars=['Centena', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido'],
                          var_name='Posicion', value_name='Numero')
        
        pos_map = {'Centena': 'Centena', 'Fijo': 'Fijo', 
                   'Primer_Corrido': '1er Corrido', 'Segundo_Corrido': '2do Corrido'}
        df_long['Posicion'] = df_long['Posicion'].map(pos_map)
        
        df_historial = df_long.dropna(subset=['Numero']).copy()
        df_historial['Numero'] = df_historial['Numero'].astype(int)
        
        # Ordenamiento Cronológico: Tarde(0), Noche(1)
        draw_order_map = {'T': 0, 'N': 1}
        df_historial['draw_order'] = df_historial['Tipo_Sorteo'].map(draw_order_map)
        df_historial['sort_key'] = df_historial['Fecha'] + pd.to_timedelta(df_historial['draw_order'], unit='h')
        df_historial = df_historial.sort_values(by='sort_key').reset_index(drop=True)
        df_historial.drop(columns=['draw_order', 'sort_key'], inplace=True)
        
        return df_historial
    except Exception as e:
        st.error(f"Error crítico procesando datos: {str(e)}")
        return pd.DataFrame()

def calcular_estado_actual(gap, limite_dinamico):
    if pd.isna(limite_dinamico) or limite_dinamico == 0: return "Normal"
    if gap > limite_dinamico: return "Muy Vencido"
    elif gap > (limite_dinamico * 0.66): return "Vencido"
    else: return "Normal"

def obtener_df_temperatura(contador):
    df = pd.DataFrame.from_dict(contador, orient='index', columns=['Frecuencia'])
    df = df.reset_index().rename(columns={'index': 'Dígito'})
    df = df.sort_values('Frecuencia', ascending=False).reset_index(drop=True)
    df['Temperatura'] = '🧊 Frío'
    if len(df) >= 3: df.loc[0:2, 'Temperatura'] = '🔥 Caliente'
    if len(df) >= 7: df.loc[6:9, 'Temperatura'] = '🧊 Frío'
    if len(df) >= 3: df.loc[3:5, 'Temperatura'] = '🟡 Tibio'
    return df

# --- FUNCIONES DE ANALISIS ---

def analizar_oportunidad_por_digito(df_historial, fecha_referencia):
    if df_historial.empty: return pd.DataFrame(), pd.DataFrame()
    df_base_fijos = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    
    contador_decenas = Counter()
    contador_unidades = Counter()
    for num in df_base_fijos['Numero']:
        contador_decenas[num // 10] += 1
        contador_unidades[num % 10] += 1

    df_temp_dec = obtener_df_temperatura(contador_decenas)
    df_temp_uni = obtener_df_temperatura(contador_unidades)
    
    mapa_temp_dec = pd.Series(df_temp_dec.Temperatura.values, index=df_temp_dec.Dígito).to_dict()
    mapa_temp_uni = pd.Series(df_temp_uni.Temperatura.values, index=df_temp_uni.Dígito).to_dict()
    
    df_hist_estado = df_base_fijos[df_base_fijos['Fecha'] < fecha_referencia].copy()
    
    res_dec, res_uni = [], []
    for i in range(10):
        fechas_d = df_hist_estado[df_hist_estado['Numero'] // 10 == i]['Fecha'].sort_values()
        gap_d, prom_d = 0, 0
        if not fechas_d.empty:
            gaps = fechas_d.diff().dt.days.dropna()
            prom_d = gaps.median() if len(gaps) > 0 else 0
            gap_d = (fecha_referencia - fechas_d.max()).days
        ed = calcular_estado_actual(gap_d, prom_d)
        
        fechas_u = df_hist_estado[df_hist_estado['Numero'] % 10 == i]['Fecha'].sort_values()
        gap_u, prom_u = 0, 0
        if not fechas_u.empty:
            gaps = fechas_u.diff().dt.days.dropna()
            prom_u = gaps.median() if len(gaps) > 0 else 0
            gap_u = (fecha_referencia - fechas_u.max()).days
        eu = calcular_estado_actual(gap_u, prom_u)
        
        p_base_d = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[ed]
        p_base_u = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[eu]
        
        res_dec.append({'Dígito': i, 'Temperatura': mapa_temp_dec.get(i, '🟡 Tibio'), 'Estado': ed, 'Punt. Base': p_base_d})
        res_uni.append({'Dígito': i, 'Temperatura': mapa_temp_uni.get(i, '🟡 Tibio'), 'Estado': eu, 'Punt. Base': p_base_u})

    return pd.DataFrame(res_dec), pd.DataFrame(res_uni)

# --- GESTOR DE CACHE ---
def obtener_historial_perfiles_cacheado(df_full, ruta_cache=None):
    if df_full.empty: return pd.DataFrame()
    
    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    df_cache = pd.DataFrame()
    
    use_file = ruta_cache and os.path.exists(ruta_cache)
    
    if use_file:
        try:
            df_cache = pd.read_csv(ruta_cache, parse_dates=['Fecha'])
        except:
            df_cache = pd.DataFrame() 

    df_fijos['ID_Sorteo'] = df_fijos['Fecha'].astype(str) + "_" + df_fijos['Tipo_Sorteo']
    
    ids_en_cache = set()
    if not df_cache.empty:
        sorteo_map_inv = {'Noche': 'N', 'Tarde': 'T'}
        df_cache['ID_Sorteo'] = df_cache['Fecha'].astype(str) + "_" + df_cache['Sorteo'].map(sorteo_map_inv)
        ids_en_cache = set(df_cache['ID_Sorteo'])
    
    df_nuevos = df_fijos[~df_fijos['ID_Sorteo'].isin(ids_en_cache)].copy()
    
    if df_nuevos.empty:
        if 'ID_Sorteo' in df_cache.columns: df_cache.drop(columns=['ID_Sorteo'], inplace=True)
        return df_cache
    
    df_nuevos = df_nuevos.sort_values(by=['Fecha', 'Tipo_Sorteo'])

    hist_decenas = defaultdict(list)
    hist_unidades = defaultdict(list)
    
    if not df_cache.empty:
        # Orden T(0), N(1)
        sort_val_inv = {'Tarde': 0, 'Noche': 1}
        df_cache['sort_val'] = df_cache['Sorteo'].map(sort_val_inv)
        df_cache_sorted = df_cache.sort_values(by=['Fecha', 'sort_val'])
        for _, row in df_cache_sorted.iterrows():
            num = int(row['Numero'])
            fecha = row['Fecha']
            hist_decenas[num // 10].append(fecha)
            hist_unidades[num % 10].append(fecha)
            
    nuevos_registros = []
    
    for idx, row in df_nuevos.iterrows():
        fecha_actual = row['Fecha']
        num_actual = row['Numero']
        tipo_actual = row['Tipo_Sorteo']
        
        dec = num_actual // 10
        uni = num_actual % 10
        
        fechas_dec_ant = [f for f in hist_decenas[dec] if f < fecha_actual]
        if fechas_dec_ant:
            last_dec = max(fechas_dec_ant)
            gap_dec = (fecha_actual - last_dec).days
            sorted_fds = sorted(fechas_dec_ant)
            gaps_d = [(sorted_fds[i] - sorted_fds[i-1]).days for i in range(1, len(sorted_fds))]
            med_d = np.median(gaps_d) if gaps_d else 0
            estado_dec = calcular_estado_actual(gap_dec, med_d)
        else:
            estado_dec = "Normal"
            
        fechas_uni_ant = [f for f in hist_unidades[uni] if f < fecha_actual]
        if fechas_uni_ant:
            last_uni = max(fechas_uni_ant)
            gap_uni = (fecha_actual - last_uni).days
            sorted_fus = sorted(fechas_uni_ant)
            gaps_u = [(sorted_fus[i] - sorted_fus[i-1]).days for i in range(1, len(sorted_fus))]
            med_u = np.median(gaps_u) if gaps_u else 0
            estado_uni = calcular_estado_actual(gap_uni, med_u)
        else:
            estado_uni = "Normal"
            
        perfil = f"{estado_dec}-{estado_uni}"
        nombre_sorteo = {'T': 'Tarde', 'N': 'Noche'}.get(tipo_actual, 'Otro')
        
        nuevos_registros.append({
            'Fecha': fecha_actual,
            'Sorteo': nombre_sorteo,
            'Numero': num_actual,
            'Perfil': perfil
        })
        
        hist_decenas[dec].append(fecha_actual)
        hist_unidades[uni].append(fecha_actual)
    
    if nuevos_registros:
        df_nuevos_cache = pd.DataFrame(nuevos_registros)
        if not df_cache.empty:
            cols_to_drop = [c for c in ['ID_Sorteo', 'sort_val'] if c in df_cache.columns]
            df_final = pd.concat([df_cache.drop(columns=cols_to_drop, errors='ignore'), df_nuevos_cache], ignore_index=True)
        else:
            df_final = df_nuevos_cache
            
        if ruta_cache:
            df_final.to_csv(ruta_cache, index=False)
        return df_final
    else:
        if 'ID_Sorteo' in df_cache.columns: df_cache.drop(columns=['ID_Sorteo'], inplace=True)
        return df_cache

def calcular_estabilidad_historica_digitos(df_full):
    if df_full.empty: return pd.DataFrame()
    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    resultados = []
    
    for i in range(10):
        fechas_d = df_fijos[df_fijos['Numero'] // 10 == i]['Fecha'].sort_values()
        if len(fechas_d) > 1:
            gaps = fechas_d.diff().dt.days.dropna()
            med = gaps.median()
            excesos = sum(g > (med * 1.5) for g in gaps)
            estabilidad = 100 - (excesos / len(gaps) * 100)
        else:
            estabilidad = 50
        resultados.append({'Digito': i, 'Tipo': 'Decena', 'EstabilidadHist': estabilidad})
        
        fechas_u = df_fijos[df_fijos['Numero'] % 10 == i]['Fecha'].sort_values()
        if len(fechas_u) > 1:
            gaps = fechas_u.diff().dt.days.dropna()
            med = gaps.median()
            excesos = sum(g > (med * 1.5) for g in gaps)
            estabilidad = 100 - (excesos / len(gaps) * 100)
        else:
            estabilidad = 50
        resultados.append({'Digito': i, 'Tipo': 'Unidad', 'EstabilidadHist': estabilidad})
        
    return pd.DataFrame(resultados)

# --- ANALISIS ESTADISTICAS PERFILES (P75) ---
def analizar_estadisticas_perfiles(df_historial_perfiles, fecha_referencia):
    historial_fechas_perfiles = defaultdict(list)
    ultimo_suceso_perfil = {}
    transiciones = Counter()
    ultimo_perfil_global = None
    
    sort_val = {'Tarde': 0, 'Noche': 1}
    df_historial_perfiles = df_historial_perfiles.copy()
    df_historial_perfiles['sort_val'] = df_historial_perfiles['Sorteo'].map(sort_val)
    df_historial_perfiles = df_historial_perfiles.sort_values(by=['Fecha', 'sort_val'])

    for _, row in df_historial_perfiles.iterrows():
        perfil = row['Perfil']
        fecha = row['Fecha']
        numero = row['Numero']
        
        historial_fechas_perfiles[perfil].append(fecha)
        ultimo_suceso_perfil[perfil] = row
        
        if ultimo_perfil_global:
            transiciones[(ultimo_perfil_global, perfil)] += 1
        ultimo_perfil_global = perfil
    
    total_salidas_perfil = Counter()
    for (origen, destino), count in transiciones.items():
        total_salidas_perfil[origen] += count
        
    analisis_perfiles = []
    
    for perfil, fechas in historial_fechas_perfiles.items():
        fechas_ordenadas = sorted(fechas)
        ultima_fecha = fechas_ordenadas[-1]
        
        gaps = []
        for k in range(1, len(fechas_ordenadas)):
            gaps.append((fechas_ordenadas[k] - fechas_ordenadas[k-1]).days)
            
        mediana_gap_actual = np.median(gaps) if gaps else 0
        gap_actual = (fecha_referencia - ultima_fecha).days
        
        if len(gaps) >= 4:
            limite_dinamico = int(np.percentile(gaps, 75))
        elif len(gaps) > 0:
            limite_dinamico = int(mediana_gap_actual * 2)
        else:
            limite_dinamico = 0

        estado_actual = calcular_estado_actual(gap_actual, limite_dinamico)
        
        estados_historicos = [calcular_estado_actual(g, limite_dinamico) for g in gaps] if gaps else []
        total_hist = len(estados_historicos)
        count_normal = estados_historicos.count('Normal')
        count_vencido = estados_historicos.count('Vencido')
        count_muy_vencido = estados_historicos.count('Muy Vencido')
        
        muy_vencidos_count = count_muy_vencido
        estabilidad_actual = ((total_hist - muy_vencidos_count) / total_hist * 100) if total_hist > 0 else 0
        
        alerta_recuperacion = False
        if estabilidad_actual > 60 and estado_actual in ['Vencido', 'Muy Vencido']:
            alerta_recuperacion = True
        
        tiempo_limite = limite_dinamico
        
        repeticiones = transiciones.get((perfil, perfil), 0)
        total_salidas = total_salidas_perfil.get(perfil, 0)
        prob_repeticion = (repeticiones / total_salidas * 100) if total_salidas > 0 else 0
        
        semana_activa = "Sí" if estado_actual in ['Vencido', 'Muy Vencido'] else "No"
        
        last_row = ultimo_suceso_perfil[perfil]
        
        estado_ultima_salida = "Normal"
        estabilidad_ultima_salida = 0.0
        exceso_ultima_salida = 0
        
        if len(gaps) >= 1:
            gap_ultima_espera = gaps[-1]
            if len(gaps) > 1:
                gaps_prev = gaps[:-1]
                if len(gaps_prev) >= 4: lim_prev = int(np.percentile(gaps_prev, 75))
                else: lim_prev = int(np.median(gaps_prev) * 2)
                estado_ultima_salida = calcular_estado_actual(gap_ultima_espera, lim_prev)
                
                if estado_ultima_salida == "Muy Vencido":
                    exceso_ultima_salida = int(gap_ultima_espera - lim_prev)
                
                ests_prev = [calcular_estado_actual(g, lim_prev) for g in gaps_prev]
                mv_prev = ests_prev.count('Muy Vencido')
                estabilidad_ultima_salida = ((len(ests_prev) - mv_prev) / len(ests_prev) * 100)
            else:
                estado_ultima_salida = "Normal"
                estabilidad_ultima_salida = 100.0
        
        analisis_perfiles.append({
            'Perfil': perfil,
            'Frecuencia': total_hist + 1,
            'Última Fecha': ultima_fecha,
            'Gap Actual': gap_actual,
            'Mediana Gap': int(mediana_gap_actual),
            'Estado Actual': estado_actual,
            'Estabilidad': round(estabilidad_actual, 1),
            'Tiempo Limite': tiempo_limite,
            'Alerta': '⚠️ RECUPERAR' if alerta_recuperacion else '-',
            'Prob Repeticiones %': round(prob_repeticion, 1),
            'Semana Activa': semana_activa,
            'Último Numero': last_row['Numero'],
            'Último Sorteo': last_row['Sorteo'],
            'Veces Normal': count_normal,
            'Veces Vencido': count_vencido,
            'Veces Muy Vencido': count_muy_vencido,
            'Estado Ultima Salida': estado_ultima_salida,
            'Estabilidad Ultima Salida': round(estabilidad_ultima_salida, 1),
            'Exceso Ultima Salida': exceso_ultima_salida
        })
        
    df_stats = pd.DataFrame(analisis_perfiles)
    return df_stats, transiciones, ultimo_perfil_global

# --- MOTORES DE PREDICCION ---
def obtener_prediccion_numeros_lista(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos):
    scores = []
    
    map_est_dec = estabilidad_digitos[(estabilidad_digitos['Tipo']=='Decena')].set_index('Digito')['EstabilidadHist'].to_dict()
    map_est_uni = estabilidad_digitos[(estabilidad_digitos['Tipo']=='Unidad')].set_index('Digito')['EstabilidadHist'].to_dict()
    
    for _, row in df_stats.iterrows():
        p = row['Perfil']
        score = 0
        estado = row['Estado Actual']
        
        if row['Alerta'] == '⚠️ RECUPERAR': score += 150 
        else:
            if estado == 'Vencido': score += 70 
            elif estado == 'Normal': score += 50 
            elif estado == 'Muy Vencido': score += 30 
        
        score += row['Estabilidad'] * 0.5
        trans_count = transizioni.get((ultimo_perfil, p), 0)
        score += trans_count * 10 
        
        scores.append({'Perfil': p, 'Score': int(score), 'Estado': estado})
    
    df_scores = pd.DataFrame(scores).sort_values('Score', ascending=False)
    top_3 = df_scores.head(3)
    
    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict()
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict()
    
    df_hist_nums = df_historial_perfiles.groupby('Numero')['Fecha'].max()
    candidatos_totales = []
    
    map_temp_dec = df_oport_dec.set_index('Dígito')['Temperatura'].to_dict()
    map_temp_uni = df_oport_uni.set_index('Dígito')['Temperatura'].to_dict()
    temp_val = {'🔥 Caliente': 3, '🟡 Tibio': 2, '🧊 Frío': 1}
    
    for _, row in top_3.iterrows():
        perfil = row['Perfil']
        partes = perfil.split('-')
        ed_req, eu_req = partes[0], partes[1]
        
        decenas_estado = [d for d in range(10) if map_estado_dec.get(d) == ed_req]
        unidades_estado = [u for u in range(10) if map_estado_uni.get(u) == eu_req]
        
        for d in decenas_estado:
            for u in unidades_estado:
                num = int(f"{d}{u}")
                last_seen = df_hist_nums.get(num, pd.Timestamp('2000-01-01'))
                gap_n = (fecha_ref - last_seen).days if isinstance(last_seen, pd.Timestamp) else 999
                
                temp_d = temp_val.get(map_temp_dec.get(d, '🟡 Tibio'), 2)
                temp_u = temp_val.get(map_temp_uni.get(u, '🟡 Tibio'), 2)
                temp_score = temp_d + temp_u
                
                est_d = map_est_dec.get(d, 50)
                est_u = map_est_uni.get(u, 50)
                bonus_
