# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import traceback
from collections import defaultdict, Counter
import unicodedata

# --- CONFIGURACIÓN DE LA RUTA_ABSOLUTA ---
# Cambiado a la ruta absoluta especificada por el usuario
RUTA_CSV = r'C:\Users\Personal\Documents\Loteria\Loterias\Cerebro\Formasalidor\Geosalidor.csv' 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Georgia - Análisis Fusionado",
    page_icon="🍑",
    layout="wide"
)

st.title("🍑 Georgia - Análisis con Estado Actualizado")
st.markdown("Sistema de Predicción que calcula el estado **ACTUAL** de los números y dígitos para los sorteos de Mañana, Tarde y Noche.")
st.info("ℹ️ **Importante:** Los estados se calculan comparando el salto actual de cada elemento con su propio promedio histórico de ausencia (redondeado). **El cálculo del salto ahora considera el sorteo específico (M/T/N) para mayor precisión.**")

# --- FUNCIÓN AUXILIAR PARA ELIMINAR ACENTOS ---
def remove_accents(input_str):
    if not isinstance(input_str, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# --- LÓGICA PARA EL CÁLCULO DE FORMASALID (CORREGIDA) ---
# Diccionario corregido según las 9 formas proporcionadas
DIGITO_A_FORMA = {
    # C: 0, 6, 8, 9
    0: 'C', 6: 'C', 8: 'C', 9: 'C', 
    # R: 1, 4, 7
    1: 'R', 4: 'R', 7: 'R',  
    # A: 2, 3, 5
    2: 'A', 3: 'A', 5: 'A'
}

def calcular_forma_desde_numero(numero):
    if pd.isna(numero): return '??' 
    try:
        s_numero = str(int(float(numero))).zfill(2)
        if len(s_numero) != 2 or not s_numero.isdigit(): return '??'
        decena, unidad = int(s_numero[0]), int(s_numero[1])
        forma_decena = DIGITO_A_FORMA.get(decena, '?')
        forma_unidad = DIGITO_A_FORMA.get(unidad, '?')
        if '?' in (forma_decena, forma_unidad): return '??'
        return f"{forma_decena}{forma_unidad}"
    except (ValueError, TypeError): return '??'

# --- LÓGICA PARA EL CÁLCULO DE ESTADO ACTUAL (CORREGIDA Y MEJORADA) ---
def calcular_estado_actual(gap, promedio_gap):
    """
    Calcula el estado basado en el gap y el promedio de gaps.
    El promedio se redondea para evitar clasificaciones 'Vencido' por diferencias decimales pequeñas.
    """
    if pd.isna(promedio_gap) or promedio_gap == 0:
        return "Normal" # No hay historial para comparar

    promedio_redondeado = round(promedio_gap)
    
    if gap <= promedio_redondeado:
        return "Normal"
    elif gap <= promedio_redondeado * 1.5:
        return "Vencido"
    else:
        return "Muy Vencido"

# --- FUNCIÓN PARA CREAR EL MAPA DE CALOR (FASE 1) - VERSIÓN CORREGIDA Y MÁS ROBUSTA ---
def crear_mapa_de_calor_numeros(df_frecuencia, top_n=30, medio_n=30):
    df_ordenado = df_frecuencia.sort_values(by='Total_Salidas_Historico', ascending=False).reset_index(drop=True).copy()
    df_ordenado['Temperatura'] = '🧊 Frío'
    df_ordenado.loc[top_n : top_n + medio_n - 1, 'Temperatura'] = '🟡 Tibio'
    df_ordenado.loc[0 : top_n - 1, 'Temperatura'] = '🔥 Caliente'
    return df_ordenado

# --- FUNCIÓN CORREGIDA Y LIMPIA ---
def analizar_estados_desde_csv(df_historial, fecha_referencia):
    st.info(f"Analizando estados hasta la fecha: {fecha_referencia.strftime('%d/%m/%Y')}")
    if 'Combinar_Estado' not in df_historial.columns or 'Fecha' not in df_historial.columns:
        st.error("Faltan las columnas 'Combinar_Estado' o 'Fecha' en el CSV para el análisis histórico.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Filtrar historial hasta la fecha de referencia
    df_historial_filtrado = df_historial[df_historial['Fecha'] < fecha_referencia].copy()
    if df_historial_filtrado.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_frecuencia = df_historial_filtrado['Combinar_Estado'].value_counts().reset_index()
    df_frecuencia.columns = ['Estado Combinado', 'Frecuencia Histórica']
    
    promedios = {}
    maximos = {}
    
    for estado in df_historial_filtrado['Combinar_Estado'].unique():
        fechas_estado = df_historial_filtrado[df_historial_filtrado['Combinar_Estado'] == estado]['Fecha'].sort_values()
        gaps = fechas_estado.diff().dt.days.dropna()
        
        if not gaps.empty:
            promedios[estado] = round(gaps.median())
            maximo_pasado = gaps.max()
            ultima_aparicion = fechas_estado.iloc[-1]
            gap_actual = (fecha_referencia - ultima_aparicion).days
            maximos[estado] = max(maximo_pasado, gap_actual)
        else:
            if not fechas_estado.empty:
                gap_actual = (fecha_referencia - fechas_estado.iloc[0]).days
                promedios[estado] = gap_actual
                maximos[estado] = gap_actual
            else:
                promedios[estado] = 0
                maximos[estado] = 0

    df_promedios = pd.DataFrame.from_dict(promedios, orient='index', columns=['Aparece cada (promedio de días)']).reset_index()
    df_maximos = pd.DataFrame.from_dict(maximos, orient='index', columns=['Máximo Histórico (días)']).reset_index()
    
    df_promedios.rename(columns={'index': 'Estado Combinado'}, inplace=True)
    df_maximos.rename(columns={'index': 'Estado Combinado'}, inplace=True)
    
    df_analisis = pd.merge(df_frecuencia, df_promedios, on='Estado Combinado', how='outer').fillna(0)
    df_analisis = pd.merge(df_analisis, df_maximos, on='Estado Combinado', how='outer').fillna(0)
    
    umbral_sospechoso = 20
    for idx, row in df_analisis.iterrows():
        if row['Frecuencia Histórica'] > 100 and row['Aparece cada (promedio de días)'] > umbral_sospechoso:
            df_analisis.loc[idx, 'Aparece cada (promedio de días)'] = f"⚠️ {row['Aparece cada (promedio de días)']} (Revisar datos)"
    
    ultima_aparicion_estados = df_historial_filtrado.groupby('Combinar_Estado')['Fecha'].max().sort_values()
    saltos_debidos = (fecha_referencia - ultima_aparicion_estados).dt.days
    df_debidos = saltos_debidos.reset_index()
    df_debidos.columns = ['Estado Combinado', 'Días de Ausencia (Debido)']
    df_debidos = df_debidos.sort_values(by='Días de Ausencia (Debido)', ascending=False)
    
    df_debidos_completo = pd.merge(df_debidos, df_analisis, on='Estado Combinado', how='left')
    
    return df_analisis, df_debidos_completo, df_frecuencia

# --- NUEVA FUNCIÓN PARA EL ANÁLISIS DE CONSISTENCIA SEMANAL ---
def analizar_consistencia_semanal_paridad(df_historial, fecha_referencia):
    st.info(f"Analizando la consistencia semanal de paridad hasta: {fecha_referencia.strftime('%d/%m/%Y')}")
    df_historial_filtrado = df_historial[df_historial['Fecha'] < fecha_referencia].copy()
    if df_historial_filtrado.empty:
        return pd.DataFrame()

    df_historial_filtrado['YearWeek'] = df_historial_filtrado['Fecha'].dt.year.astype(str) + '-' + df_historial_filtrado['Fecha'].dt.isocalendar().week.astype(str).str.zfill(2)
    df_historial_filtrado['Combinación Paridad'] = df_historial_filtrado['Numero'].apply(calcular_paridad_desde_numero)
    
    semanas_totales = df_historial_filtrado['YearWeek'].nunique()
    consistencia = df_historial_filtrado.groupby('Combinación Paridad')['YearWeek'].nunique().reset_index()
    consistencia.columns = ['Combinación Paridad', 'Semanas que Salió']
    consistencia['Semanas Totales Analizadas'] = semanas_totales
    consistencia['Consistencia (%)'] = (consistencia['Semanas que Salió'] / semanas_totales * 100).round(2)
    
    todas_las_semanas = sorted(df_historial_filtrado['YearWeek'].unique())
    maximos_sequias = {}
    for paridad in consistencia['Combinación Paridad'].unique():
        semanas_de_paridad = set(df_historial_filtrado[df_historial_filtrado['Combinación Paridad'] == paridad]['YearWeek'].unique())
        max_sequia = 0
        sequia_actual = 0
        for semana in todas_las_semanas:
            if semana in semanas_de_paridad:
                max_sequia = max(max_sequia, sequia_actual)
                sequia_actual = 0
            else:
                sequia_actual += 1
        maximos_sequias[paridad] = max(max_sequia, sequia_actual)
        
    df_maximos_sequias = pd.DataFrame.from_dict(maximos_sequias, orient='index', columns=['Máximo de Semanas Consecutivas sin Salir']).reset_index()
    df_maximos_sequias.rename(columns={'index': 'Combinación Paridad'}, inplace=True)
    
    consistencia_final = pd.merge(consistencia, df_maximos_sequias, on='Combinación Paridad', how='left').fillna(0)
    
    return consistencia_final.sort_values(by='Consistencia (%)', ascending=False)

# --- FUNCIÓN PARA EL ANÁLISIS DE PARIDAD ---
def calcular_paridad_desde_numero(numero):
    if pd.isna(numero): return '??'
    try:
        s_numero = str(int(float(numero))).zfill(2)
        if len(s_numero) != 2 or not s_numero.isdigit(): return '??'
        decena, unidad = int(s_numero[0]), int(s_numero[1])
        rango_decena = 'Bajo' if decena <= 4 else 'Alto'
        rango_unidad = 'Bajo' if unidad <= 4 else 'Alto'
        paridad_decena = 'Par' if decena % 2 == 0 else 'Impar'
        paridad_unidad = 'Par' if unidad % 2 == 0 else 'Impar'
        return f"{rango_decena}{rango_unidad}-{paridad_decena}-{paridad_unidad}"
    except (ValueError, TypeError):
        return '??'

# --- NUEVA FUNCIÓN PARA OBTENER MÁXIMOS HISTÓRICOS DE PARIDAD DESDE CSV ---
def obtener_maximos_historicos_paridad(df_historial):
    target_paridad = 'Clasificación Paridad'
    target_saltos = 'Saltos Paridad'
    
    col_paridad = None
    col_saltos = None
    
    clean_target_paridad = remove_accents(target_paridad).strip().lower()
    clean_target_saltos = remove_accents(target_saltos).strip().lower()
    
    for col in df_historial.columns:
        clean_col = remove_accents(col).strip().lower()
        if clean_col == clean_target_paridad:
            col_paridad = col
        if clean_col == clean_target_saltos:
            col_saltos = col

    if not col_paridad or not col_saltos:
        st.error("❌ No se pudieron encontrar las columnas necesarias para el análisis de paridad.")
        st.error(f"Buscando '{target_paridad}' -> Encontrada: '{col_paridad}'")
        st.error(f"Buscando '{target_saltos}' -> Encontrada: '{col_saltos}'")
        st.info("💡 **Sugerencia:** Revisa la sección de 'Modo Diagnóstico' para ver todos los encabezados de tu archivo.")
        return pd.DataFrame()
    
    maximos_paridad = df_historial.groupby(col_paridad)[col_saltos].max().reset_index()
    maximos_paridad.columns = ['Combinación Paridad', 'Máximo Histórico de Ausencia (Sorteos)']
    maximos_paridad = maximos_paridad.sort_values(by='Máximo Histórico de Ausencia (Sorteos)', ascending=False)
    
    return maximos_paridad

# --- NUEVA FUNCIÓN PARA ANALIZAR PARIDADES (SIMILAR A ANALIZAR_ESTADOS) ---
def analizar_paridades_desde_csv(df_historial, fecha_referencia):
    st.info(f"Analizando paridades hasta la fecha: {fecha_referencia.strftime('%d/%m/%Y')}")
    
    # Asegurarse de que la columna de paridad existe
    if 'Combinación Paridad' not in df_historial.columns:
        df_historial['Combinación Paridad'] = df_historial['Numero'].apply(calcular_paridad_desde_numero)
    
    if 'Combinación Paridad' not in df_historial.columns or 'Fecha' not in df_historial.columns:
        st.error("Faltan las columnas 'Combinación Paridad' o 'Fecha' en el CSV para el análisis histórico.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Filtrar historial hasta la fecha de referencia
    df_historial_filtrado = df_historial[df_historial['Fecha'] < fecha_referencia].copy()
    if df_historial_filtrado.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_frecuencia = df_historial_filtrado['Combinación Paridad'].value_counts().reset_index()
    df_frecuencia.columns = ['Combinación Paridad', 'Frecuencia Histórica']
    
    promedios = {}
    maximos = {}
    
    for paridad in df_historial_filtrado['Combinación Paridad'].unique():
        fechas_paridad = df_historial_filtrado[df_historial_filtrado['Combinación Paridad'] == paridad]['Fecha'].sort_values()
        gaps = fechas_paridad.diff().dt.days.dropna()
        
        if not gaps.empty:
            promedios[paridad] = round(gaps.median())
            maximo_pasado = gaps.max()
            ultima_aparicion = fechas_paridad.iloc[-1]
            gap_actual = (fecha_referencia - ultima_aparicion).days
            maximos[paridad] = max(maximo_pasado, gap_actual)
        else:
            if not fechas_paridad.empty:
                gap_actual = (fecha_referencia - fechas_paridad.iloc[0]).days
                promedios[paridad] = gap_actual
                maximos[paridad] = gap_actual
            else:
                promedios[paridad] = 0
                maximos[paridad] = 0

    df_promedios = pd.DataFrame.from_dict(promedios, orient='index', columns=['Aparece cada (promedio de días)']).reset_index()
    df_maximos = pd.DataFrame.from_dict(maximos, orient='index', columns=['Máximo Histórico (días)']).reset_index()
    
    df_promedios.rename(columns={'index': 'Combinación Paridad'}, inplace=True)
    df_maximos.rename(columns={'index': 'Combinación Paridad'}, inplace=True)
    
    df_analisis = pd.merge(df_frecuencia, df_promedios, on='Combinación Paridad', how='outer').fillna(0)
    df_analisis = pd.merge(df_analisis, df_maximos, on='Combinación Paridad', how='outer').fillna(0)
    
    ultima_aparicion_paridades = df_historial_filtrado.groupby('Combinación Paridad')['Fecha'].max().sort_values()
    saltos_debidos = (fecha_referencia - ultima_aparicion_paridades).dt.days
    df_debidos = saltos_debidos.reset_index()
    df_debidos.columns = ['Combinación Paridad', 'Días de Ausencia (Debido)']
    df_debidos = df_debidos.sort_values(by='Días de Ausencia (Debido)', ascending=False)
    
    df_debidos_completo = pd.merge(df_debidos, df_analisis, on='Combinación Paridad', how='left')
    
    return df_analisis, df_debidos_completo, df_frecuencia

# --- NUEVA FUNCIÓN PARA OBTENER NÚMEROS MÁS SALIDORES POR PARIDAD (CORREGIDA) ---
def obtener_numeros_salidores_por_paridad(df_estados_completos):
    """
    Genera un diccionario con los números más salidores para CADA UNA de las 16 combinaciones de paridad.
    Asegura que todas las combinaciones posibles estén representadas, incluso si no tienen números.
    """
    st.info("Generando análisis de números para las 16 combinaciones de paridad...")
    
    # 1. Generar programáticamente las 16 combinaciones de paridad posibles
    rangos = ['Bajo', 'Alto']
    paridades = ['Par', 'Impar']
    paridades_posibles = []
    for r_dec in rangos:
        for r_uni in rangos:
            for p_dec in paridades:
                for p_uni in paridades:
                    paridades_posibles.append(f"{r_dec}{r_uni}-{p_dec}-{p_uni}")

    # 2. Crear un diccionario para almacenar los resultados
    resultados = {}
    
    # 3. Para CADA combinación de paridad posible, encontrar los números correspondientes
    for paridad in sorted(paridades_posibles):
        # Filtrar números que pertenecen a esa paridad desde el dataframe de estados completo
        numeros_de_paridad = df_estados_completos[df_estados_completos['Combinación Paridad'] == paridad].copy()
        
        if not numeros_de_paridad.empty:
            # Ordenar por frecuencia histórica (de mayor a menor)
            numeros_de_paridad = numeros_de_paridad.sort_values(by='Total_Salidas_Historico', ascending=False)
            # Añadir ranking
            numeros_de_paridad['Ranking'] = range(1, len(numeros_de_paridad) + 1)
            # Seleccionar columnas relevantes para mostrar
            resultados[paridad] = numeros_de_paridad[['Numero', 'Ranking', 'Total_Salidas_Historico', 'Combinar_Estado_Actual', 'Forma_Calculada']]
        else:
            # Si no hay números para esta paridad, crear un DataFrame vacío con las columnas correctas
            # Esto asegura que la pestaña aparezca y muestre que no hay datos.
            resultados[paridad] = pd.DataFrame(columns=['Numero', 'Ranking', 'Total_Salidas_Historico', 'Combinar_Estado_Actual', 'Forma_Calculada'])
            
    return resultados

# --- NUEVA FUNCIÓN MAESTRA PARA CALCULAR ESTADOS (LÓGICA CORREGIDA CON SORTEO) ---
def get_full_state_dataframe(df_historial, fecha_referencia):
    st.info(f"Calculando estado completo de todos los números hasta: {fecha_referencia.strftime('%d/%m/%Y')}")
    df_historial_filtrado = df_historial[df_historial['Fecha'] < fecha_referencia].copy()
    if df_historial_filtrado.empty:
        return pd.DataFrame()

    # --- CORRECCIÓN CLAVE: CREAR CLAVE DE ORDEN CRONOLÓGICO PRECISA ---
    # Esto permite diferenciar entre los sorteos del mismo día (M, T, N)
    if 'Tipo_Sorteo' in df_historial_filtrado.columns:
        draw_order_map = {'M': 0, 'T': 1, 'N': 2}
        # Rellenar valores nulos con un número alto para que se ordenen al final
        df_historial_filtrado['draw_order'] = df_historial_filtrado['Tipo_Sorteo'].map(draw_order_map).fillna(3)
        df_historial_filtrado['sort_key'] = df_historial_filtrado['Fecha'] + pd.to_timedelta(df_historial_filtrado['draw_order'], unit='h')
    else:
        # Si no hay tipo de sorteo, usar solo la fecha como fallback
        df_historial_filtrado['sort_key'] = df_historial_filtrado['Fecha']

    df_maestro = pd.DataFrame({'Numero': range(100)})
    primera_fecha_historica = df_historial['Fecha'].min()

    # --- PRE-CALCULATE HISTORICAL AVERAGES FOR EACH ELEMENT ---
    st.info("Pre-calculando promedios históricos individuales para cada número, decena y unidad...")
    
    historicos_numero = {}
    for i in range(100):
        fechas_i = df_historial_filtrado[df_historial_filtrado['Numero'] == i]['Fecha'].sort_values()
        gaps = fechas_i.diff().dt.days.dropna()
        if not gaps.empty:
            historicos_numero[i] = gaps.median()
        else:
            historicos_numero[i] = (fecha_referencia - primera_fecha_historica).days

    historicos_decena = {}
    for i in range(10):
        fechas_i = df_historial_filtrado[df_historial_filtrado['Numero'] // 10 == i]['Fecha'].sort_values()
        gaps = fechas_i.diff().dt.days.dropna()
        if not gaps.empty:
            historicos_decena[i] = gaps.median()
        else:
            historicos_decena[i] = (fecha_referencia - primera_fecha_historica).days

    historicos_unidad = {}
    for i in range(10):
        fechas_i = df_historial_filtrado[df_historial_filtrado['Numero'] % 10 == i]['Fecha'].sort_values()
        gaps = fechas_i.diff().dt.days.dropna()
        if not gaps.empty:
            historicos_unidad[i] = gaps.median()
        else:
            historicos_unidad[i] = (fecha_referencia - primera_fecha_historica).days

    # --- CALCULATE CURRENT GAPS AND STATES USING THE PRECISE SORT KEY ---
    df_maestro['Decena'] = df_maestro['Numero'] // 10
    df_maestro['Unidad'] = df_maestro['Numero'] % 10
    
    # Estado Numero
    ultima_aparicion_num_key = df_historial_filtrado.groupby('Numero')['sort_key'].max().reindex(range(100))
    ultima_aparicion_num_key.fillna(primera_fecha_historica, inplace=True)
    gap_num = (fecha_referencia - ultima_aparicion_num_key).dt.days
    df_maestro['Salto_Numero'] = gap_num
    df_maestro['Estado_Numero'] = df_maestro.apply(lambda row: calcular_estado_actual(row['Salto_Numero'], historicos_numero[row['Numero']]), axis=1)
    df_maestro['Última Aparición (Fecha)'] = ultima_aparicion_num_key.dt.strftime('%d/%m/%Y')

    # Estado Decena
    ultima_aparicion_dec_key = df_historial_filtrado.groupby(df_historial_filtrado['Numero'] // 10)['sort_key'].max().reindex(range(10))
    ultima_aparicion_dec_key.fillna(primera_fecha_historica, inplace=True)
    gap_dec = (fecha_referencia - ultima_aparicion_dec_key).dt.days
    df_maestro['Estado_Decena'] = df_maestro.apply(lambda row: calcular_estado_actual(gap_dec[row['Decena']], historicos_decena[row['Decena']]), axis=1)

    # Estado Unidad
    ultima_aparicion_uni_key = df_historial_filtrado.groupby(df_historial_filtrado['Numero'] % 10)['sort_key'].max().reindex(range(10))
    ultima_aparicion_uni_key.fillna(primera_fecha_historica, inplace=True)
    gap_uni = (fecha_referencia - ultima_aparicion_uni_key).dt.days
    df_maestro['Estado_Unidad'] = df_maestro.apply(lambda row: calcular_estado_actual(gap_uni[row['Unidad']], historicos_unidad[row['Unidad']]), axis=1)
    
    # Recalcular el estado combinado ya que los componentes pudieron haber cambiado
    df_maestro['Combinar_Estado_Actual'] = df_maestro['Estado_Decena'] + '-' + df_maestro['Estado_Unidad']
    
    df_maestro['Forma_Calculada'] = df_maestro['Numero'].apply(calcular_forma_desde_numero)

    df_historial_con_p = df_historial_filtrado.copy()
    df_historial_con_p['Combinación Paridad'] = df_historial_con_p['Numero'].apply(calcular_paridad_desde_numero)
    promedio_gap_por_paridad = df_historial_con_p.groupby('Combinación Paridad')['Fecha'].apply(
        lambda x: (x.sort_values().diff().dt.days.median()) 
    ).round()
    
    def get_paridad_state(numero):
        paridad = calcular_paridad_desde_numero(numero)
        if paridad in promedio_gap_por_paridad:
            promedio_gap = promedio_gap_por_paridad[paridad]
            gap = df_maestro.loc[df_maestro['Numero'] == numero, 'Salto_Numero'].iloc[0]
            return calcular_estado_actual(gap, promedio_gap)
        else:
            return 'Normal'
    
    df_maestro['Estado_Paridad'] = df_maestro['Numero'].apply(get_paridad_state)
    df_maestro['Combinación Paridad'] = df_maestro['Numero'].apply(calcular_paridad_desde_numero)
    
    frecuencia = df_historial_filtrado['Numero'].value_counts().reindex(range(100)).fillna(0)
    df_maestro['Total_Salidas_Historico'] = frecuencia

    return df_maestro, historicos_decena, historicos_unidad

# --- FUNCIÓN PARA EL ANÁLISIS DE OPORTUNIDAD POR DÍGITO (CORREGIDA Y MEJORADA) ---
def analizar_oportunidad_por_digito(df_historial, df_estados_completos, historicos_decena, historicos_unidad, modo_temperatura, fecha_inicio_rango, fecha_fin_rango, top_n_candidatos=5):
    st.info(f"Analizando oportunidades por dígito en modo: {modo_temperatura}")
    
    # 1. Determinar el rango de fechas para el análisis de temperatura (ALMANAQUE)
    if modo_temperatura == "Histórico Completo":
        df_temperatura = df_historial.copy()
        st.info("Usando el historial completo para el análisis de temperatura.")
    else: # Personalizado por Rango
        # CORRECCIÓN CLAVE: Asegurarse de que la fecha de fin incluya todo el día
        end_of_day = fecha_fin_rango + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        df_temperatura = df_historial[(df_historial['Fecha'] >= fecha_inicio_rango) & (df_historial['Fecha'] <= end_of_day)].copy()
        
        if df_temperatura.empty:
            st.warning(f"El rango de fechas seleccionado ({fecha_inicio_rango.strftime('%d/%m/%Y')} a {fecha_fin_rango.strftime('%d/%m/%Y')}) no contiene sorteos. Se usará el historial completo.")
            df_temperatura = df_historial.copy()
        else:
            st.success(f"✅ Usando el rango de fechas {fecha_inicio_rango.strftime('%d/%m/%Y')} a {fecha_fin_rango.strftime('%d/%m/%Y')} para el análisis de temperatura. Se encontraron {len(df_temperatura)} sorteos.")

    # 2. Calcular la frecuencia de cada dígito (0-9) SEPARADAMENTE para Decenas y Unidades
    # --- CORRECCIÓN CLAVE: Contadores separados para Decenas y Unidades ---
    contador_decenas = Counter()
    contador_unidades = Counter()
    for num in df_temperatura['Numero']:
        contador_decenas[num // 10] += 1  # Solo para Decenas
        contador_unidades[num % 10] += 1   # Solo para Unidades

    # --- CORRECCIÓN CLAVE: DataFrames separados para Decenas y Unidades ---
    df_frecuencia_decenas = pd.DataFrame.from_dict(contador_decenas, orient='index', columns=['Frecuencia Total']).reset_index()
    df_frecuencia_decenas.rename(columns={'index': 'Dígito'}, inplace=True)
    df_frecuencia_unidades = pd.DataFrame.from_dict(contador_unidades, orient='index', columns=['Frecuencia Total']).reset_index()
    df_frecuencia_unidades.rename(columns={'index': 'Dígito'}, inplace=True)

    # 3. Clasificar Temperatura de cada dígito usando POSICIONES DIRECTA (LÓGICA 3-3-4 CORREGIDA)
    # Para Decenas
    df_frecuencia_decenas = df_frecuencia_decenas.sort_values(by='Frecuencia Total', ascending=False).reset_index(drop=True)
    df_frecuencia_decenas['Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_decenas) >= 3:
        df_frecuencia_decenas.loc[0:3, 'Temperatura'] = '🔥 Caliente'
    if len(df_frecuencia_decenas) >= 6:
        df_frecuencia_decenas.loc[3:6, 'Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_decenas) >= 7:
        df_frecuencia_decenas.loc[6:10, 'Temperatura'] = '🧊 Frío'
    
    # Para Unidades
    df_frecuencia_unidades = df_frecuencia_unidades.sort_values(by='Frecuencia Total', ascending=False).reset_index(drop=True)
    df_frecuencia_unidades['Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_unidades) >= 3:
        df_frecuencia_unidades.loc[0:3, 'Temperatura'] = '🔥 Caliente'
    if len(df_frecuencia_unidades) >= 6:
        df_frecuencia_unidades.loc[3:6, 'Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_unidades) >= 7:
        df_frecuencia_unidades.loc[6:10, 'Temperatura'] = '🧊 Frío'
    
    # --- CORRECCIÓN CLAVE: Mapas de temperatura separados para Decenas y Unidades ---
    mapa_temperatura_decenas = pd.Series(df_frecuencia_decenas.Temperatura.values, index=df_frecuencia_decenas.Dígito).to_dict()
    mapa_temperatura_unidades = pd.Series(df_frecuencia_unidades.Temperatura.values, index=df_frecuencia_unidades.Dígito).to_dict()
    
    # --- NUEVO: Crear un mapa de puntuación por temperatura ---
    puntuacion_temperatura_map = {
        '🔥 Caliente': 30,
        '🟡 Tibio': 20,
        '🧊 Frío': 10
    }

    # 4. Crear DataFrames de Decenas y Unidades con su Temperatura y Estado
    resultados_decenas = []
    resultados_unidades = []
    
    for i in range(10):
        # Obtener estado para la decena y unidad 'i'
        estado_decena = df_estados_completos[df_estados_completos['Decena'] == i]['Estado_Decena'].iloc[0]
        estado_unidad = df_estados_completos[df_estados_completos['Unidad'] == i]['Estado_Unidad'].iloc[0]

        # --- LÓGICA DE PUNTUACIÓN ---
        # Puntuación base por estado
        puntuacion_base_decena = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[estado_decena]
        puntuacion_base_unidad = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[estado_unidad]

        # Puntuación proactiva para los "Normales"
        if estado_decena == 'Normal':
            promedio = historicos_decena.get(i, 1)
            gap_actual = df_estados_completos[df_estados_completos['Decena'] == i]['Salto_Numero'].iloc[0]
            # Puntuación proporcional a qué tan cerca está del promedio (máximo 49 puntos para no igualar a "Vencido")
            puntuacion_proactiva_decena = min(49, (gap_actual / promedio) * 50) if promedio > 0 else 0
        else:
            puntuacion_proactiva_decena = 0

        if estado_unidad == 'Normal':
            promedio = historicos_unidad.get(i, 1)
            gap_actual = df_estados_completos[df_estados_completos['Unidad'] == i]['Salto_Numero'].iloc[0]
            puntuacion_proactiva_unidad = min(49, (gap_actual / promedio) * 50) if promedio > 0 else 0
        else:
            puntuacion_proactiva_unidad = 0

        # --- NUEVO: Obtener puntuación por temperatura de los mapas CORRECTOS ---
        temperatura_decena = mapa_temperatura_decenas.get(i, '🟡 Tibio')
        temperatura_unidad = mapa_temperatura_unidades.get(i, '🟡 Tibio')
        
        puntuacion_temp_decena = puntuacion_temperatura_map.get(temperatura_decena, 20)
        puntuacion_temp_unidad = puntuacion_temperatura_map.get(temperatura_unidad, 20)

        puntuacion_total_decena = puntuacion_base_decena + puntuacion_proactiva_decena + puntuacion_temp_decena
        puntuacion_total_unidad = puntuacion_base_unidad + puntuacion_proactiva_unidad + puntuacion_temp_unidad

        resultados_decenas.append({
            'Dígito': i, 'Rol': 'Decena', 'Temperatura': temperatura_decena, 'Estado': estado_decena,
            'Puntuación Base': puntuacion_base_decena, 'Puntuación Proactiva': round(puntuacion_proactiva_decena, 1),
            'Puntuación Temperatura': puntuacion_temp_decena, # NUEVA COLUMNA
            'Puntuación Total': round(puntuacion_total_decena, 1)
        })
        resultados_unidades.append({
            'Dígito': i, 'Rol': 'Unidad', 'Temperatura': temperatura_unidad, 'Estado': estado_unidad,
            'Puntuación Base': puntuacion_base_unidad, 'Puntuación Proactiva': round(puntuacion_proactiva_unidad, 1),
            'Puntuación Temperatura': puntuacion_temp_unidad, # NUEVA COLUMNA
            'Puntuación Total': round(puntuacion_total_unidad, 1)
        })

    df_oportunidad_decenas = pd.DataFrame(resultados_decenas)
    df_oportunidad_unidades = pd.DataFrame(resultados_unidades)

    # 5. Generar Top N Números Candidatos por Puntuación Combinada MEJORADA
    puntuacion_decena_map = df_oportunidad_decenas.set_index('Dígito')['Puntuación Total'].to_dict()
    puntuacion_unidad_map = df_oportunidad_unidades.set_index('Dígito')['Puntuación Total'].to_dict()

    candidatos = []
    for num in range(100):
        decena = num // 10
        unidad = num % 10
        score_total = puntuacion_decena_map.get(decena, 0) + puntuacion_unidad_map.get(unidad, 0)
        candidatos.append({'Numero': num, 'Puntuación Total': score_total})

    # --- MODIFICACIÓN: Usar el parámetro top_n_candidatos ---
    df_candidatos = pd.DataFrame(candidatos).sort_values(by='Puntuación Total', ascending=False).head(top_n_candidatos)
    df_candidatos['Numero'] = df_candidatos['Numero'].apply(lambda x: f"{x:02d}")

    return df_oportunidad_decenas, df_oportunidad_unidades, df_candidatos


# --- FUNCIÓN PARA BUSCAR PATRONES DE 3 FORMAS ---
def buscar_patron_formas(df_historial, patron_formas):
    """
    Busca un patrón de 3 formas en el historial y encuentra las formas que le siguen.
    """
    if len(patron_formas) != 3:
        st.error("El patrón debe contener exactamente 3 formas.")
        return pd.DataFrame()

    historial_formas = df_historial['Forma_Calculada'].tolist()
    if len(historial_formas) < 4:
        st.warning("No hay suficientes datos en el historial para buscar un patrón de 3 formas.")
        return pd.DataFrame()

    formas_siguientes = []
    # Iterar desde el 3er elemento hasta el penúltimo
    for i in range(2, len(historial_formas) - 1):
        # Comprobar si el bloque actual coincide con el patrón
        if historial_formas[i-2:i+1] == patron_formas:
            # Si coincide, el siguiente elemento es el que buscamos
            siguiente_forma = historial_formas[i+1]
            formas_siguientes.append(siguiente_forma)

    if not formas_siguientes:
        st.info(f"El patrón **{' -> '.join(patron_formas)}** no se ha encontrado nunca en el historial.")
        return pd.DataFrame()

    # Contar las ocurrencias de cada "siguiente forma"
    conteo_formas = Counter(formas_siguientes)
    
    # Crear un DataFrame para mostrar los resultados
    df_resultados = pd.DataFrame(conteo_formas.items(), columns=['4ta Forma', 'Cantidad de Ocurrencias']).sort_values(by='Cantidad de Ocurrencias', ascending=False)
    
    return df_resultados

# --- NUEVA FUNCIÓN PARA BUSCAR PATRONES DE 2 FORMAS ---
def buscar_patron_formas_2(df_historial, patron_formas):
    """
    Busca un patrón de 2 formas en el historial y encuentra las formas que le siguen.
    """
    if len(patron_formas) != 2:
        st.error("El patrón debe contener exactamente 2 formas.")
        return pd.DataFrame()

    historial_formas = df_historial['Forma_Calculada'].tolist()
    if len(historial_formas) < 3:
        st.warning("No hay suficientes datos en el historial para buscar un patrón de 2 formas.")
        return pd.DataFrame()

    formas_siguientes = []
    # Iterar desde el 2do elemento hasta el penúltimo
    for i in range(1, len(historial_formas) - 1):
        # Comprobar si el bloque actual coincide con el patrón
        if historial_formas[i-1:i+1] == patron_formas:
            # Si coincide, el siguiente elemento es el que buscamos
            siguiente_forma = historial_formas[i+1]
            formas_siguientes.append(siguiente_forma)

    if not formas_siguientes:
        st.info(f"El patrón **{' -> '.join(patron_formas)}** no se ha encontrado nunca en el historial.")
        return pd.DataFrame()

    # Contar las ocurrencias de cada "siguiente forma"
    conteo_formas = Counter(formas_siguientes)
    
    # Crear un DataFrame para mostrar los resultados
    df_resultados = pd.DataFrame(conteo_formas.items(), columns=['3ra Forma', 'Cantidad de Ocurrencias']).sort_values(by='Cantidad de Ocurrencias', ascending=False)
    
    return df_resultados


# --- CARGA Y PROCESAMIENTO UNIFICADO DE DATOS ---
@st.cache_resource
def cargar_datos_georgia(_ruta_csv, debug_mode=False):
    try:
        st.info("Cargando y procesando datos históricos de Georgia...")
        # Modificado para usar la ruta absoluta directamente
        ruta_csv_absoluta = _ruta_csv
        
        if not os.path.exists(ruta_csv_absoluta):
            st.error(f"❌ Error: No se encontró el archivo de datos de Georgia.")
            st.error(f"La aplicación buscó el archivo en la ruta: {ruta_csv_absoluta}")
            st.warning("💡 **Solución:** Asegúrate de que la carpeta 'Formasalidor' y el archivo 'Geosalidor.csv' existan.")
            st.stop()
        df_historial = pd.read_csv(ruta_csv_absoluta, sep=';', encoding='latin-1')

        if debug_mode:
            st.subheader("🔍 Examen Completo de los Encabezados del CSV")
            st.markdown("A continuación se muestran **TODOS** los encabezados de columna encontrados en tu archivo. Esto te ayudará a verificar que los nombres son correctos.")
            columnas_encontradas = list(df_historial.columns)
            st.write("**Lista completa de encabezados:**")
            st.code("\n".join(columnas_encontradas))
            st.markdown("---")
            st.subheader("🔍 Diagnóstico Específico para Columnas de Paridad y Saltos")
            target_paridad = 'Clasificación Paridad'
            target_saltos = 'Saltos Paridad'
            clean_target_paridad = remove_accents(target_paridad).strip().lower()
            clean_target_saltos = remove_accents(target_saltos).strip().lower()
            col_paridad_encontrada = None
            col_saltos_encontrada = None
            st.write(f"**Buscando objetivo 1:** `{target_paridad}`")
            for col in columnas_encontradas:
                clean_col = remove_accents(col).strip().lower()
                if clean_col == clean_target_paridad:
                    col_paridad_encontrada = col
                    st.success(f"✅ Coincidencia encontrada: `{col}`")
                    break
            if not col_paridad_encontrada:
                st.error(f"❌ No se encontró ninguna coincidencia para '{target_paridad}'.")
            st.write(f"**Buscando objetivo 2:** `{target_saltos}`")
            for col in columnas_encontradas:
                clean_col = remove_accents(col).strip().lower()
                if clean_col == clean_target_saltos:
                    col_saltos_encontrada = col
                    st.success(f"✅ Coincidencia encontrada: `{col}`")
                    break
            if not col_saltos_encontrada:
                st.error(f"❌ No se encontró ninguna coincidencia para '{target_saltos}'.")
            st.markdown("---")
            st.subheader("Vista Previa de las Columnas Encontradas")
            if col_paridad_encontrada and col_saltos_encontrada:
                st.dataframe(df_historial[[col_paridad_encontrada, col_saltos_encontrada]].head(10))
            else:
                st.warning("No se pueden mostrar las columnas porque una o ambas no fueron encontradas.")

        posibles_cols_numero = ['Resultado', 'Resultado ', 'Numero', 'Numero ', 'Número', 'Número ', 'Ganador', 'Ganador ']
        col_numero_encontrada = None
        for col in df_historial.columns:
            if col in posibles_cols_numero:
                col_numero_encontrada = col
                break
        
        if col_numero_encontrada is None:
            st.error("❌ Error Crítico: No se pudo encontrar una columna de resultados con un nombre reconocido.")
            st.error(f"Nombres buscados: {posibles_cols_numero}")
            st.stop()

        col_sorteo_name = next((col for col in df_historial.columns if df_historial[col].dropna().isin(['M', 'T', 'N']).any()), None)
        rename_map = {
            'Fecha': 'Fecha', 
            col_numero_encontrada: 'Numero',
            'Combinar Estado Dec-Uni': 'Combinar_Estado_Historico',
            'Estado D': 'Estado_Decena_Historico', 
            'Estado U': 'Estado_Unidad_Historico'
        }
        if col_sorteo_name: rename_map[col_sorteo_name] = 'Tipo_Sorteo'

        df_historial.rename(columns=rename_map, inplace=True)
        
        if 'Numero' not in df_historial.columns:
            st.error("❌ Error Crítico: La columna 'Numero' no existe después del renombrado.")
            st.stop()
            
        st.info("Limpiando y formateando la columna de números...")
        df_historial['Numero'] = pd.to_numeric(df_historial['Numero'], errors='coerce')
        df_historial.dropna(subset=['Numero'], inplace=True)
        df_historial['Numero'] = df_historial['Numero'].astype(int)
        
        if 'Combinar_Estado_Historico' in df_historial.columns:
            df_historial['Combinar_Estado'] = df_historial['Combinar_Estado_Historico']
        
        df_historial['Fecha'] = pd.to_datetime(df_historial['Fecha'], dayfirst=True, errors='coerce')
        df_historial.dropna(subset=['Fecha', 'Numero'], inplace=True)
        if df_historial.empty:
            st.error("No se encontraron datos válidos.")
            st.stop()
        
        df_historial['Forma_Calculada'] = df_historial['Numero'].apply(calcular_forma_desde_numero)
        
        # --- CORRECCIÓN CLAVE: ORDENAR CRONOLÓGICAMENTE AQUÍ TAMBIÉN ---
        # Esto asegura que el df_historial principal usado para todo esté ordenado.
        if 'Tipo_Sorteo' in df_historial.columns:
            draw_order_map = {'M': 0, 'T': 1, 'N': 2}
            df_historial['draw_order'] = df_historial['Tipo_Sorteo'].map(draw_order_map).fillna(3)
            df_historial['sort_key'] = df_historial['Fecha'] + pd.to_timedelta(df_historial['draw_order'], unit='h')
            df_historial = df_historial.sort_values(by='sort_key').reset_index(drop=True)
            df_historial.drop(columns=['draw_order', 'sort_key'], inplace=True)
        else:
            df_historial = df_historial.sort_values(by='Fecha').reset_index(drop=True)
        # --- FIN DE LA CORRECCIÓN ---

        st.success("¡Datos de Georgia cargados y procesados con éxito!")
        return df_historial
    except Exception as e:
        st.error(f"Error al cargar y procesar los datos de Georgia: {str(e)}")
        if debug_mode:
            st.error(traceback.format_exc())
        st.stop()

# --- FUNCIONES DE ANÁLISIS ---
def obtener_debidas(df_historico, fecha_referencia, columna_analisis):
    # CORRECCIÓN: Filtrar el historial hasta la fecha de referencia
    df_filtrado = df_historico[df_historico['Fecha'] < fecha_referencia].copy()
    if df_filtrado.empty: 
        return pd.DataFrame()
    
    # CORRECCIÓN: Usar el DataFrame filtrado para calcular la última aparición
    ultima_aparicion = df_filtrado.groupby(columna_analisis)['Fecha'].max().sort_values()
    saltos = (fecha_referencia - ultima_aparicion).dt.days
    df_prediccion = pd.DataFrame({
        columna_analisis: saltos.index, 
        'Días sin Aparecer (Salto)': saltos.values,
        'Última Aparición': ultima_aparicion.dt.strftime('%d/%m/%Y')
    }).sort_values(by='Días sin Aparecer (Salto)', ascending=False)
    return df_prediccion

# --- LÓGICA PRINCIPAL DE LA APLICACIÓN ---
def main():
    st.sidebar.header("⚙️ Opciones de Análisis - Georgia")
    
    debug_mode = st.sidebar.checkbox("🔍 Activar Modo Diagnóstico (CSV)", help="Muestra información detallada del archivo CSV para solucionar problemas.")
    
    modo_analisis = st.sidebar.radio(
        "Modo de Análisis Principal:",
        ["Análisis Actual (usando fecha de hoy)", "Análisis Semanal (usando fecha del domingo)"]
    )

    if modo_analisis == "Análisis Semanal (usando fecha del domingo)":
        domingo_seleccionado = st.sidebar.date_input("Selecciona el Domingo de la semana:", value=datetime.now().date(), format="DD/MM/YYYY")
        fecha_referencia = pd.to_datetime(domingo_seleccionado)
    else:
        fecha_referencia = pd.to_datetime(datetime.now().date())
        st.sidebar.info(f"Analizando con la fecha de hoy: {fecha_referencia.strftime('%d/%m/%Y')}")

    # --- SELECTOR DE MODO DE TEMPERATURA (ALMANAQUE) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌡️ Modo de Temperatura de Dígitos (Almanaque)")
    modo_temperatura = st.sidebar.radio(
        "Selecciona el modo para calcular la temperatura de los dígitos:",
        ["Histórico Completo", "Personalizado por Rango"]
    )
    
    fecha_inicio_rango, fecha_fin_rango = None, None
    if modo_temperatura == "Personalizado por Rango":
        st.sidebar.markdown("**Selecciona el rango de fechas (Almanaque):**")
        fecha_inicio_rango = st.sidebar.date_input("Fecha de Inicio:", value=fecha_referencia - pd.Timedelta(days=30), format="DD/MM/YYYY")
        fecha_fin_rango = st.sidebar.date_input("Fecha de Fin:", value=fecha_referencia - pd.Timedelta(days=1), format="DD/MM/YYYY")
        if fecha_inicio_rango > fecha_fin_rango:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")

    # --- NUEVO SELECTOR PARA TOP N CANDIDATOS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏆 Análisis de Top Números")
    top_n_candidatos = st.slider("Top N de Números Candidatos a mostrar:", min_value=1, max_value=20, value=5, step=1)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Forzar Recarga de Datos"):
        st.cache_resource.clear()
        st.sidebar.success("¡Cache limpio! Recargando...")
        st.rerun()

    df_historial = cargar_datos_georgia(RUTA_CSV, debug_mode)

    if df_historial is not None:
        # --- INFORMACIÓN DE ÚLTIMOS SORTEOS (MAÑANA, TARDE Y NOCHE) ---
        st.sidebar.markdown("---")
        if 'Tipo_Sorteo' in df_historial.columns:
            if 'M' in df_historial['Tipo_Sorteo'].unique():
                ultimo_sorteo_M = df_historial[df_historial['Tipo_Sorteo'] == 'M'].iloc[-1]
                st.sidebar.info(f"Último sorteo **Mañana**: {ultimo_sorteo_M['Fecha'].strftime('%d/%m/%Y')} (Número: {ultimo_sorteo_M['Numero']})")
            if 'T' in df_historial['Tipo_Sorteo'].unique():
                ultimo_sorteo_T = df_historial[df_historial['Tipo_Sorteo'] == 'T'].iloc[-1]
                st.sidebar.info(f"Último sorteo **Tarde**: {ultimo_sorteo_T['Fecha'].strftime('%d/%m/%Y')} (Número: {ultimo_sorteo_T['Numero']})")
            if 'N' in df_historial['Tipo_Sorteo'].unique():
                ultimo_sorteo_N = df_historial[df_historial['Tipo_Sorteo'] == 'N'].iloc[-1]
                st.sidebar.info(f"Último sorteo **Noche**: {ultimo_sorteo_N['Fecha'].strftime('%d/%m/%Y')} (Número: {ultimo_sorteo_N['Numero']})")
        else:
            ultima_fila = df_historial.iloc[-1]
            st.sidebar.info(f"Último sorteo registrado: {ultima_fila['Fecha'].strftime('%d/%m/%Y')} (Número: {ultima_fila['Numero']})")
        
        df_estados_completos, historicos_decena, historicos_unidad = get_full_state_dataframe(df_historial, fecha_referencia)
        
        if df_estados_completos.empty:
            st.error("No se pudo calcular el estado de los números para la fecha de referencia.")
            st.stop()

        frecuencia_numeros_historica = df_historial['Numero'].value_counts().reset_index()
        frecuencia_numeros_historica.columns = ['Numero', 'Total_Salidas_Historico']
        df_clasificacion_general = crear_mapa_de_calor_numeros(frecuencia_numeros_historica)
        
        df_analisis_estados, df_debidos_estados, df_frecuencia_estados = analizar_estados_desde_csv(df_historial, fecha_referencia)
        df_consistencia_paridad = analizar_consistencia_semanal_paridad(df_historial, fecha_referencia)
        df_maximos_paridad = obtener_maximos_historicos_paridad(df_historial)
        
        # --- NUEVO: ANÁLISIS DE PARIDADES SIMILAR A ESTADOS ---
        df_analisis_paridades, df_debidos_paridades, df_frecuencia_paridades = analizar_paridades_desde_csv(df_historial, fecha_referencia)
        # La llamada a la función corregida ya no necesita df_historial
        numeros_por_paridad = obtener_numeros_salidores_por_paridad(df_estados_completos)
        
        # --- NUEVO: ANÁLISIS DE OPORTUNIDAD POR DÍGITO ---
        # CORRECCIÓN: Asegurarse de que las fechas no sean None antes de convertirlas
        fecha_inicio_rango_safe = pd.to_datetime(fecha_inicio_rango) if fecha_inicio_rango else None
        fecha_fin_rango_safe = pd.to_datetime(fecha_fin_rango) if fecha_fin_rango else None
        
        # --- MODIFICACIÓN: Pasar el valor del slider a la función ---
        df_oportunidad_decenas, df_oportunidad_unidades, top_candidatos = analizar_oportunidad_por_digito(
            df_historial, df_estados_completos, historicos_decena, historicos_unidad, 
            modo_temperatura, fecha_inicio_rango_safe, fecha_fin_rango_safe,
            top_n_candidatos
        )
        
        estados_posibles = ["Normal", "Vencido", "Muy Vencido"]
        combinaciones_posibles = [f"{d}-{u}" for d in estados_posibles for u in estados_posibles]
        
        st.header("📊 Análisis Paralelo de Formas y Estados")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔷 Análisis de Formas")
            # CORRECCIÓN: Usar la función obtener_debidas con la fecha_referencia correcta
            df_formas_debidas = obtener_debidas(df_historial, fecha_referencia, 'Forma_Calculada')
            if not df_formas_debidas.empty:
                st.dataframe(df_formas_debidas, width='stretch')
                formas_debidas_top3 = df_formas_debidas.head(3)['Forma_Calculada'].tolist()
                st.success(f"Formas más debidas: **{', '.join(formas_debidas_top3)}**")
                st.session_state['formas_debidas'] = formas_debidas_top3

        with col2:
            st.subheader("🧠 Frecuencia y Promedio Histórico de Estados")
            if not df_analisis_estados.empty:
                df_estados_salidores_completos = df_analisis_estados.sort_values(by='Frecuencia Histórica', ascending=False)
                st.dataframe(df_estados_salidores_completos, width='stretch')
                estados_mas_salidores = df_estados_salidores_completos.head(3)['Estado Combinado'].tolist()
                st.success(f"Top 3 Estados más salidores (Histórico): **{', '.join(estados_mas_salidores)}**")
            
            st.markdown("---")
            st.subheader(f"🎯 Estados Más Debidos (hasta {fecha_referencia.strftime('%d/%m/%Y')})")
            if not df_debidos_estados.empty:
                st.dataframe(df_debidos_estados, width='stretch')
                estados_debidos_top3_act = df_debidos_estados.head(3)['Estado Combinado'].tolist()
                st.success(f"Top 3 Estados más debidos: **{', '.join(estados_debidos_top3_act)}**")
                st.session_state['estados_debidos_act'] = estados_debidos_top3_act

        st.markdown("---")
        st.header("🎯 Análisis de Antecesor y Sucesor")
        ultima_fila = df_historial.iloc[-1]
        ultimo_numero = ultima_fila['Numero']
        ultima_unidad = ultimo_numero % 10
        
        antecesor = (ultima_unidad - 1) % 10
        sucesor = (ultima_unidad + 1) % 10
        
        st.markdown(f"Basado en el último número sorteado (**{ultimo_numero:02d}**), este análisis muestra los números más debidos que contienen el dígito antecesor (**{antecesor}**) y el dígito sucesor (**{sucesor}**), tanto en la posición de la unidad como de la decena.")
        
        df_antecesor_unidad = df_estados_completos[df_estados_completos['Unidad'] == antecesor]
        df_sucesor_unidad = df_estados_completos[df_estados_completos['Unidad'] == sucesor]
        df_antecesor_decena = df_estados_completos[df_estados_completos['Decena'] == antecesor]
        df_sucesor_decena = df_estados_completos[df_estados_completos['Decena'] == sucesor]
        
        df_combinado = pd.concat([df_antecesor_unidad, df_sucesor_unidad, df_antecesor_decena, df_sucesor_decena])
        df_combinado = df_combinado.drop_duplicates(subset='Numero').sort_values(by='Salto_Numero', ascending=False)
        
        if not df_combinado.empty:
            st.dataframe(df_combinado[['Numero', 'Salto_Numero', 'Estado_Numero', 'Forma_Calculada', 'Combinar_Estado_Actual']], width='stretch')
        else:
            st.warning("No se encontraron números para el análisis de antecesor/sucesor.")

        st.markdown("---")
        st.header("📈 Análisis de Ausencia Extrema de Estados")
        st.markdown("Muestra los estados más debidos y su progreso hacia su **máximo histórico de ausencia**. Un alto porcentaje indica una oportunidad extrema.")
        
        if not df_debidos_estados.empty:
            df_debidos_estados['Progreso hacia Máximo (%)'] = (df_debidos_estados['Días de Ausencia (Debido)'] / df_debidos_estados['Máximo Histórico (días)'] * 100).round(2)
            df_debidos_estados['Progreso hacia Máximo (%)'] = df_debidos_estados['Progreso hacia Máximo (%)'].replace([np.inf, -np.inf], 0).fillna(0)
            
            df_ausencia_extrema = df_debidos_estados.sort_values(by='Progreso hacia Máximo (%)', ascending=False)
            st.dataframe(df_ausencia_extrema, width='stretch')
        else:
            st.warning("No hay datos para calcular la ausencia extrema.")

        st.markdown("---")
        st.header("🔄 Análisis de Consistencia Semanal de Paridad")
        st.markdown("Identifica las paridades que son más confiables semanalmente. Un alto 'Consistencia (%)' significa que esa paridad aparece casi todas las semanas.")
        
        with st.sidebar:
            st.subheader("⚙️ Filtro de Consistencia")
            umbral_consistencia = st.slider("Mostrar solo paridades con una consistencia mayor a:", min_value=0, max_value=100, value=75, step=5)
        
        if not df_consistencia_paridad.empty:
            paridades_fiables = df_consistencia_paridad[df_consistencia_paridad['Consistencia (%)'] >= umbral_consistencia]
            st.dataframe(paridades_fiables, width='stretch')
            if paridades_fiables.empty:
                st.warning(f"No hay paridades con una consistencia mayor o igual al {umbral_consistencia}%.")
        else:
            st.warning("No hay datos para analizar la consistencia semanal.")

        st.markdown("---")
        st.header("📊 Máximo Histórico de Ausencia por Paridad")
        st.markdown("Muestra el máximo histórico de ausencia (en sorteos) para cada una de las 16 combinaciones de paridad, basado en los datos precalculados del CSV.")
        
        if not df_maximos_paridad.empty:
            st.dataframe(df_maximos_paridad, width='stretch')
        else:
            st.warning("No hay datos de paridad disponibles para mostrar el máximo histórico de ausencia.")

        # --- NUEVA SECCIÓN: ANÁLISIS COMPLETO DE PARIDADES ---
        st.markdown("---")
        st.header("📊 Análisis Completo de Paridades")
        st.markdown("Análisis detallado de las 16 combinaciones de paridad, similar al análisis de estados combinados.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧠 Frecuencia y Promedio Histórico de Paridades")
            if not df_analisis_paridades.empty:
                df_paridades_salidores_completos = df_analisis_paridades.sort_values(by='Frecuencia Histórica', ascending=False)
                st.dataframe(df_paridades_salidores_completos, width='stretch')
                paridades_mas_salidores = df_paridades_salidores_completos.head(3)['Combinación Paridad'].tolist()
                st.success(f"Top 3 Paridades más salidores (Histórico): **{', '.join(paridades_mas_salidores)}**")
            
            st.markdown("---")
            st.subheader(f"🎯 Paridades Más Debidas (hasta {fecha_referencia.strftime('%d/%m/%Y')})")
            if not df_debidos_paridades.empty:
                st.dataframe(df_debidos_paridades, width='stretch')
                paridades_debidas_top3_act = df_debidos_paridades.head(3)['Combinación Paridad'].tolist()
                st.success(f"Top 3 Paridades más debidas: **{', '.join(paridades_debidas_top3_act)}**")
                st.session_state['paridades_debidos_act'] = paridades_debidas_top3_act
        
        with col2:
            st.subheader("📈 Análisis de Ausencia Extrema de Paridades")
            st.markdown("Muestra las paridades más debidas y su progreso hacia su **máximo histórico de ausencia**. Un alto porcentaje indica una oportunidad extrema.")
            
            if not df_debidos_paridades.empty:
                df_debidos_paridades['Progreso hacia Máximo (%)'] = (df_debidos_paridades['Días de Ausencia (Debido)'] / df_debidos_paridades['Máximo Histórico (días)'] * 100).round(2)
                df_debidos_paridades['Progreso hacia Máximo (%)'] = df_debidos_paridades['Progreso hacia Máximo (%)'].replace([np.inf, -np.inf], 0).fillna(0)
                
                df_ausencia_extrema_paridades = df_debidos_paridades.sort_values(by='Progreso hacia Máximo (%)', ascending=False)
                st.dataframe(df_ausencia_extrema_paridades, width='stretch')
            else:
                st.warning("No hay datos para calcular la ausencia extrema de paridades.")

        st.markdown("---")
        st.header("📊 Números Más Salidores por Paridad")
        st.markdown("A continuación se muestran los números más salidores dentro de cada combinación de paridad, ordenados de mayor a menor frecuencia histórica.")
        
        if numeros_por_paridad:
            # Crear pestañas para cada combinación de paridad
            paridad_tabs = st.tabs(sorted(numeros_por_paridad.keys()))
            
            for i, paridad in enumerate(sorted(numeros_por_paridad.keys())):
                with paridad_tabs[i]:
                    df_to_show = numeros_por_paridad[paridad]
                    if not df_to_show.empty:
                        st.dataframe(df_to_show, width='stretch')
                    else:
                        st.warning(f"No se encontraron números para la paridad '{paridad}'. Esto significa que ningún número con esta combinación ha salido en el historial.")
        else:
            st.warning("No hay datos de números por paridad disponibles.")

        st.markdown("---")
        st.header("🎯 Números 'Calientes' con Oportunidad (Debidos)")
        st.markdown("Intersección de los números más salidores (Top 30) con los que están en estado 'Vencido' o 'Muy Vencido'.")
        
        numeros_calientes_df = df_clasificacion_general[df_clasificacion_general['Temperatura'] == '🔥 Caliente']
        calientes_con_estado = numeros_calientes_df.merge(df_estados_completos[['Numero', 'Estado_Numero']], on='Numero')
        calientes_con_oportunidad = calientes_con_estado[calientes_con_estado['Estado_Numero'].isin(['Vencido', 'Muy Vencido'])]
        
        if calientes_con_oportunidad.empty:
            st.warning(f"Actualmente, ninguno de los números 'Calientes' (Top 30) se encuentra en estado de 'Oportunidad' (Vencido o Muy Vencido) hasta la fecha {fecha_referencia.strftime('%d/%m/%Y')}.")
        else:
            st.success(f"Se encontraron {len(calientes_con_oportunidad)} números 'Calientes' con 'Oportunidad'.")
            st.dataframe(calientes_con_oportunidad[['Numero', 'Total_Salidas_Historico', 'Estado_Numero']], width='stretch', hide_index=True)

        st.markdown("---")
        st.header("🔥 Números 'Calientes' con los Estados Más Salidores")
        st.markdown(f"Muestra los números 'Calientes' cuyo estado en la **fecha de referencia ({fecha_referencia.strftime('%d/%m/%Y')})** coincide con los estados combinados más frecuentes en la historia.")

        with st.sidebar:
            st.subheader("⚙️ Análisis de Estados Salidores")
            top_n_estados = st.slider("Top N de Estados Más Salidores a considerar:", min_value=1, max_value=10, value=3, step=1)

        if not df_analisis_estados.empty:
            estados_top_n = df_analisis_estados.sort_values(by='Frecuencia Histórica', ascending=False).head(top_n_estados)['Estado Combinado'].tolist()
            st.info(f"Analizando coincidencias con los Top {top_n_estados} estados: **{', '.join(estados_top_n)}**")
            
            calientes_con_estado_completo = numeros_calientes_df.merge(df_estados_completos[['Numero', 'Combinar_Estado_Actual']], on='Numero')
            resultado_final = calientes_con_estado_completo[calientes_con_estado_completo['Combinar_Estado_Actual'].isin(estados_top_n)]

            if resultado_final.empty:
                st.warning(f"Ninguno de los números 'Calientes' se encontraba en los Top {top_n_estados} estados más salidores para la fecha de referencia.")
            else:
                resultado_final.loc[:, 'Forma_Calculada'] = resultado_final['Numero'].apply(calcular_forma_desde_numero)
                st.success(f"Se encontraron {len(resultado_final)} números 'Calientes' con estados de alta frecuencia en la fecha de referencia.")
                st.dataframe(
                    resultado_final[['Numero', 'Forma_Calculada', 'Total_Salidas_Historico', 'Combinar_Estado_Actual']], 
                    width='stretch', 
                    hide_index=True
                )
        else:
            st.warning("No hay datos históricos anteriores a la fecha de referencia para realizar este análisis.")

        st.markdown("---")
        st.header("🌡️ Clasificación General de Números (Histórico)")
        st.markdown("Lista permanente de los números clasificados por su frecuencia histórica de salidas.")
        col_cal, col_tib, col_fri = st.columns(3)
        with col_cal:
            st.metric("🔥 Calientes (Top 30)", f"{len(df_clasificacion_general[df_clasificacion_general['Temperatura'] == '🔥 Caliente'])} números")
            calientes_lista = df_clasificacion_general[df_clasificacion_general['Temperatura'] == '🔥 Caliente']['Numero'].tolist()
            st.write(", ".join(map(str, calientes_lista)))
        with col_tib:
            st.metric("🟡 Tibios (Siguientes 30)", f"{len(df_clasificacion_general[df_clasificacion_general['Temperatura'] == '🟡 Tibio'])} números")
            tibios_lista = df_clasificacion_general[df_clasificacion_general['Temperatura'] == '🟡 Tibio']['Numero'].tolist()
            st.write(", ".join(map(str, tibios_lista)))
        with col_fri:
            st.metric("🧊 Fríos (Últimos 40)", f"{len(df_clasificacion_general[df_clasificacion_general['Temperatura'] == '🧊 Frío'])} números")
            frios_lista = df_clasificacion_general[df_clasificacion_general['Temperatura'] == '🧊 Frío']['Numero'].tolist()
            st.write(", ".join(map(str, frios_lista)))
        
        st.markdown("---")
        st.header("🎯 Análisis de Oportunidad por Dígito (Decenas y Unidades)")
        st.markdown(f"Esta tabla muestra la **Temperatura** (frecuencia en el rango) y el **Estado** (urgencia actual) de cada dígito. La **Puntuación Total** combina la urgencia del estado (Base) con la proximidad a vencerse (Proactiva) y la tendencia del período (Temperatura). La distribución de temperatura es exactamente **3 Calientes, 3 Tibios y 4 Fríos** para Decenas y Unidades por separado.")
        
        col_dec, col_uni = st.columns(2)
        with col_dec:
            st.subheader("📊 Oportunidad por Decena")
            st.dataframe(df_oportunidad_decenas.sort_values(by='Puntuación Total', ascending=False), width='stretch', hide_index=True)
        with col_uni:
            st.subheader("📊 Oportunidad por Unidad")
            st.dataframe(df_oportunidad_unidades.sort_values(by='Puntuación Total', ascending=False), width='stretch', hide_index=True)
        
        st.markdown("---")
        # --- MODIFICACIÓN: Usar el valor dinámico del slider en el título y texto ---
        st.subheader(f"🏆 Top {top_n_candidatos} Números Candidatos (Puntuación Combinada Mejorada)")
        st.markdown(f"A continuación se muestran los {top_n_candidatos} números cuya suma de **Puntuación Total** (Decena + Unidad) es más alta. Esta puntuación ahora incluye la influencia de la **Temperatura** del rango de fechas seleccionado.")
        st.dataframe(top_candidatos, width='stretch', hide_index=True)

        st.markdown("---")
        st.header("🔀 Análisis de Paridad")
        st.markdown(f"Estado de cada número (00-99) hasta la fecha de referencia **{fecha_referencia.strftime('%d/%m/%Y')}** basado en el comportamiento de su grupo de **Rango y Paridad (16 combinaciones)**.")
        st.dataframe(df_estados_completos, width='stretch')
        
        st.markdown("---")
        st.header("🔥 Análisis de Fusión: Búsqueda de Coincidencias")
        st.markdown(f"Encuentra los números según su **Estado en la fecha de referencia**, su **Forma**, su **Combinación de Paridad (16)** y su **Estado de Paridad**.")

        with st.sidebar:
            st.subheader("⚙️ Filtros de Búsqueda")
            filtrar_por_forma = st.checkbox("Filtrar por Forma", value=True)
            filtrar_por_estado = st.checkbox("Filtrar por Estado", value=True)
            filtrar_por_paridad = st.checkbox("Filtrar por Paridad", value=False)
            filtrar_por_estado_paridad = st.checkbox("Filtrar por Estado de Paridad", value=False)

        # --- CORRECCIÓN: Inicializar las variables de los filtros para evitar el NameError ---
        # Esto asegura que las variables siempre existan antes de ser usadas.
        formas_seleccionadas = []
        estados_seleccionados = []
        paridades_seleccionadas = []
        estados_paridad_seleccionados = []
        # --- FIN DE LA CORRECCIÓN ---

        if 'formas_debidas' in st.session_state and 'estados_debidos_act' in st.session_state:
            col_fusion1, col_fusion2, col_fusion3, col_fusion4 = st.columns(4)
            with col_fusion1:
                formas_seleccionadas = st.multiselect(
                    "Selecciona las Formas:", 
                    options=sorted(df_historial['Forma_Calculada'].unique()),
                    default=st.session_state.get('formas_debidas', [])
                )
            with col_fusion2:
                default_estados = st.session_state.get('estados_debidos_act', [])
                estados_seleccionados = st.multiselect(
                    "Selecciona los Estados:", 
                    options=combinaciones_posibles, 
                    default=default_estados
                )
            with col_fusion3:
                paridades_posibles = sorted(df_estados_completos['Combinación Paridad'].unique())
                default_paridades = st.session_state.get('paridades_debidos_act', [])
                paridades_seleccionadas = st.multiselect(
                    "Selecciona las Paridades:", 
                    options=paridades_posibles, 
                    default=default_paridades
                )
            with col_fusion4:
                estados_paridad_posibles = sorted(df_estados_completos['Estado_Paridad'].unique())
                default_estados_paridad = st.session_state.get('estados_paridad_debidos_act', [])
                estados_paridad_seleccionados = st.multiselect(
                    "Selecciona Estados de Paridad:", 
                    options=estados_paridad_posibles, 
                    default=default_estados_paridad
                )
            
            if st.button("Buscar Números Candidatos"):
                filtros_activos = {
                    'forma': filtrar_por_forma and formas_seleccionadas,
                    'estado': filtrar_por_estado and estados_seleccionados,
                    'paridad': filtrar_por_paridad and paridades_seleccionadas,
                    'estado_paridad': filtrar_por_estado_paridad and estados_paridad_seleccionados
                }

                if not any(filtros_activos.values()):
                    st.warning("Por favor, activa al menos un filtro y selecciona opciones para realizar la búsqueda.")
                else:
                    df_candidatos = df_estados_completos.copy()
                    mask = pd.Series(True, index=df_candidatos.index)

                    if filtros_activos['forma']:
                        mask &= df_candidatos['Forma_Calculada'].isin(formas_seleccionadas)
                    if filtros_activos['estado']:
                        mask &= df_candidatos['Combinar_Estado_Actual'].isin(estados_seleccionados)
                    if filtros_activos['paridad']:
                        mask &= df_candidatos['Combinación Paridad'].isin(paridades_seleccionadas)
                    if filtros_activos['estado_paridad']:
                        mask &= df_candidatos['Estado_Paridad'].isin(estados_paridad_seleccionados)
                    
                    df_candidatos_filtrados = df_candidatos[mask]

                    if not df_candidatos_filtrados.empty:
                        st.success(f"Se encontraron {len(df_candidatos_filtrados)} números candidatos para la fecha de referencia.")
                        st.dataframe(df_candidatos_filtrados, width='stretch')
                    else:
                        st.warning("No se encontraron números que coincidan con los criterios seleccionados para la fecha de referencia.")
        else:
            st.warning("Por favor, ejecuta primero el análisis de Formas y Estados para obtener los valores predeterminados.")
        
        st.markdown("---")
        st.header("🔬 Explorador de Números con Estado")
        forma_explorador = st.selectbox("Selecciona Forma para explorar:", options=sorted(df_historial['Forma_Calculada'].unique()))
        if forma_explorador:
            df_explorado = df_estados_completos[df_estados_completos['Forma_Calculada'] == forma_explorador].sort_values(by='Numero')
            if df_explorado.empty:
                st.warning(f"No se encontraron números con la forma '{forma_explorador}' en el estado actual.")
            else:
                df_explorado_con_ranking = df_explorado.copy()
                df_explorado_con_ranking['Ranking Histórico (Salidas)'] = df_explorado_con_ranking['Total_Salidas_Historico'].rank(method='dense', ascending=False).astype(int)
                # CORREGIDO: Cambiado 'Forma_Caldelada' por 'Forma_Calculada'
                columnas_orden = ['Numero', 'Ranking Histórico (Salidas)', 'Total_Salidas_Historico', 'Salto_Numero', 'Estado_Numero', 'Estado_Decena', 'Estado_Unidad', 'Combinar_Estado_Actual', 'Forma_Calculada']
                df_explorado_con_ranking = df_explorado_con_ranking[columnas_orden]
                st.subheader(f"Números de la Forma: **{forma_explorador}**")
                df_display = df_explorado_con_ranking.rename(columns={
                    'Salto_Numero': 'Salto Actual', 
                    'Estado_Numero': 'Estado Número (Actual)', 
                    'Estado_Decena': 'Estado Decena (Actual)', 
                    'Estado_Unidad': 'Estado Unidad (Actual)', 
                    'Combinar_Estado_Actual': 'Estado Combinado (Actual)'
                })
                st.dataframe(df_display, width='stretch')

        # --- SECCIÓN 1: BÚSQUEDA DE PATRONES DE 3 FORMAS ---
        st.markdown("---")
        st.header("🔍 Búsqueda de Patrones Secuenciales de Formas")
        st.markdown("Esta herramienta busca patrones de 3 formas consecutivas en todo el historial y te muestra cuál es la forma que más veces le ha seguido.")

        # --- DEPURACIÓN: Mostrar las últimas filas para verificar el orden ---
        with st.expander("🔍 Verificar Últimos Sorteos (Depuración)"):
            st.markdown("A continuación se muestran las últimas 5 filas del historial ordenado cronológicamente. **Verifica que el orden y las formas coinciden con tus expectativas.**")
            df_verificacion = df_historial.tail(5)[['Fecha', 'Tipo_Sorteo', 'Numero', 'Forma_Calculada']].copy()
            df_verificacion['Fecha'] = df_verificacion['Fecha'].dt.strftime('%d/%m/%Y')
            st.dataframe(df_verificacion, hide_index=True)
        # --- FIN DE LA DEPURACIÓN ---

        # Obtener las últimas 3 formas
        if len(df_historial) >= 3:
            ultimas_tres_formas = df_historial.tail(3)['Forma_Calculada'].tolist()
            patron_actual = ' -> '.join(ultimas_tres_formas)
            st.info(f"Patrón detectado (de más antiguo a más reciente): **{patron_actual}**")
            
            # Buscar el patrón en el historial
            df_patrones = buscar_patron_formas(df_historial, ultimas_tres_formas)
            
            if not df_patrones.empty:
                st.subheader("📈 Resultados del Patrón")
                st.dataframe(df_patrones, width='stretch')
                
                # Encontrar la forma que más ha seguido al patrón
                if not df_patrones.empty:
                    forma_mas_comun = df_patrones.iloc[0]['4ta Forma']
                    st.success(f"La forma que más ha seguido al patrón **{patron_actual}** es **{forma_mas_comun}** (seguido {df_patrones.iloc[0]['Cantidad de Ocurrencias']} veces).")
                
                # Opción para ver el detalle de las ocurrencias
                with st.expander("Ver todas las ocurrencias del patrón"):
                    st.markdown(f"A continuación se listan todas las veces que el patrón **{patron_actual}** ocurrió en el historial, mostrando la 4ta forma que le siguió.")
                    
                    # Reconstruir las ocurrencias para mostrarlas
                    historial_formas = df_historial['Forma_Calculada'].tolist()
                    ocurrencias_detalle = []
                    for i in range(2, len(historial_formas) - 1):
                        if historial_formas[i-2:i+1] == ultimas_tres_formas:
                            cuarta_forma = historial_formas[i+1]
                            fecha_ocurrencia = df_historial.iloc[i+1]['Fecha']
                            numero_ocurrencia = df_historial.iloc[i+1]['Numero']
                            ocurrencias_detalle.append({
                                'Posición en el Patrón': '4ta Forma',
                                'Forma': cuarta_forma,
                                'Fecha de Ocurrencia': fecha_ocurrencia.strftime('%d/%m/%Y'),
                                'Número del Sorteo': f"{numero_ocurrencia:02d}"
                            })
                    
                    if ocurrencias_detalle:
                        st.dataframe(pd.DataFrame(ocurrencias_detalle), width='stretch', hide_index=True)
            else:
                st.info("No se encontraron coincidencias para el patrón.")

        else:
            st.warning("Se necesitan al menos 3 sorteos para poder analizar un patrón de 3 formas.")

        # --- NUEVA SECCIÓN 2: BÚSQUEDA DE PATRONES DE 2 FORMAS ---
        st.markdown("---")
        st.header("🔍 Búsqueda de Patrones Secuenciales (2 Formas)")
        st.markdown("Esta herramienta busca patrones de 2 formas consecutivas en todo el historial y te muestra cuál es la forma que más veces le ha seguido.")

        # Obtener las últimas 2 formas
        if len(df_historial) >= 2:
            ultimas_dos_formas = df_historial.tail(2)['Forma_Calculada'].tolist()
            patron_actual_2 = ' -> '.join(ultimas_dos_formas)
            st.info(f"Patrón detectado (de más antiguo a más reciente): **{patron_actual_2}**")
            
            # Buscar el patrón en el historial usando la nueva función
            df_patrones_2 = buscar_patron_formas_2(df_historial, ultimas_dos_formas)
            
            if not df_patrones_2.empty:
                st.subheader("📈 Resultados del Patrón")
                st.dataframe(df_patrones_2, width='stretch')
                
                # Encontrar la forma que más ha seguido al patrón
                if not df_patrones_2.empty:
                    forma_mas_comun_2 = df_patrones_2.iloc[0]['3ra Forma']
                    st.success(f"La forma que más ha seguido al patrón **{patron_actual_2}** es **{forma_mas_comun_2}** (seguido {df_patrones_2.iloc[0]['Cantidad de Ocurrencias']} veces).")
                
                # Opción para ver el detalle de las ocurrencias
                with st.expander("Ver todas las ocurrencias del patrón"):
                    st.markdown(f"A continuación se listan todas las veces que el patrón **{patron_actual_2}** ocurrió en el historial, mostrando la 3ra forma que le siguió.")
                    
                    # Reconstruir las ocurrencias para mostrarlas
                    historial_formas = df_historial['Forma_Calculada'].tolist()
                    ocurrencias_detalle = []
                    for i in range(1, len(historial_formas) - 1):
                        if historial_formas[i-1:i+1] == ultimas_dos_formas:
                            tercera_forma = historial_formas[i+1]
                            fecha_ocurrencia = df_historial.iloc[i+1]['Fecha']
                            numero_ocurrencia = df_historial.iloc[i+1]['Numero']
                            ocurrencias_detalle.append({
                                'Posición en el Patrón': '3ra Forma',
                                'Forma': tercera_forma,
                                'Fecha de Ocurrencia': fecha_ocurrencia.strftime('%d/%m/%Y'),
                                'Número del Sorteo': f"{numero_ocurrencia:02d}"
                            })
                    
                    if ocurrencias_detalle:
                        st.dataframe(pd.DataFrame(ocurrencias_detalle), width='stretch', hide_index=True)
            else:
                st.info("No se encontraron coincidencias para el patrón.")

        else:
            st.warning("Se necesitan al menos 2 sorteos para poder analizar un patrón de 2 formas.")


if __name__ == "__main__":
    main()