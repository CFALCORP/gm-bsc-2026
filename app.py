import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="BSC Gráficas Modernas 2026", layout="wide", page_icon="📊")

# Estilos CSS
st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    .stMetricLabel {font-weight: bold; color: #444;}
    div[data-testid="stMetricDelta"] > svg { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA DE DATOS (NUBE) ---
@st.cache_data(ttl=60)
def load_data():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTExBommhNOhtmWDQy-wzCb24jRtR1CLxfH1xzkmoFvodW-AannKBqZshXAdXIADXq5M9_rdm_XMTgr/pub?gid=228314910&single=true&output=csv"
    try:
        try:
            df = pd.read_csv(url, sep=',')
            if 'Proceso' not in df.columns: raise ValueError()
        except:
            df = pd.read_csv(url, sep=';')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"⚠️ Error conectando a Google Sheets: {e}")
        st.stop()

df_raw = load_data()

# --- 3. LIMPIEZA Y NORMALIZACIÓN ---
df = df_raw.copy()

def normalizar_porcentaje(x):
    if pd.isna(x): return 0.0
    
    tiene_simbolo = False
    if isinstance(x, str):
        if '%' in x: tiene_simbolo = True
        clean_val = x.replace('%', '').replace('"', '').replace(',', '.').strip()
        if clean_val == '' or clean_val == '-': return 0.0
        try:
            val = float(clean_val)
        except: return 0.0
    else:
        val = float(x)
    
    if tiene_simbolo: return val
    if abs(val) <= 1.5 and val != 0: return val * 100
    return val

MESES_OFICIALES = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
cols_a_limpiar = ['Meta', 'Prom. Año', 'Cumpl. Año'] + MESES_OFICIALES

for col in cols_a_limpiar:
    if col in df.columns:
        df[col] = df[col].apply(normalizar_porcentaje)

if 'Cumpl. Año' in df.columns:
    df['Cumpl. Año'] = df['Cumpl. Año'].apply(lambda x: x*100 if x <= 2.0 and x != 0 else x)


# --- 4. BARRA LATERAL ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    
    st.divider()
    st.header("🔍 Filtros Generales")
    
    lista_procesos = ["Todos"] + sorted(list(df['Proceso'].unique()))
    proceso_sel = st.selectbox("📂 1. Proceso:", lista_procesos)
    df_temp1 = df[df['Proceso'] == proceso_sel] if proceso_sel != "Todos" else df

    lista_pilares = ["Todos"] + sorted(list(df_temp1['Pilar'].unique()))
    pilar_sel = st.selectbox("🏛️ 2. Pilar:", lista_pilares)
    df_temp2 = df_temp1[df_temp1['Pilar'] == pilar_sel] if pilar_sel != "Todos" else df_temp1

    lista_indicadores = ["Todos"] + sorted(list(df_temp2['Indicador'].unique()))
    indicador_sel = st.selectbox("🎯 3. Indicador:", lista_indicadores)
    
    st.divider()
    st.subheader("📅 Filtro de Tiempo")
    
    meses_disponibles = [m for m in MESES_OFICIALES if m in df.columns]
    
    ver_todos = st.checkbox("Ver todo el año (Resumen)", value=True)
    
    if ver_todos:
        meses_seleccionados = meses_disponibles
        mostrar_meses_tabla = False
    else:
        meses_seleccionados = st.multiselect(
            "Selecciona los meses:", 
            meses_disponibles,
            default=meses_disponibles[:1]
        )
        meses_seleccionados = sorted(meses_seleccionados, key=lambda x: MESES_OFICIALES.index(x))
        mostrar_meses_tabla = True

    st.caption("🟢 Conectado a Google Sheets")

# Aplicar filtros
df_filtered = df_temp2.copy()
if indicador_sel != "Todos":
    df_filtered = df_filtered[df_filtered['Indicador'] == indicador_sel]

# --- 5. TÍTULO Y ESTADÍSTICAS ---
st.title("📊 Tablero de Mando Integral 2026")
subtitulo = indicador_sel if indicador_sel != "Todos" else (proceso_sel if proceso_sel != "Todos" else "Visión Global")
st.markdown(f"**Vista Actual:** {subtitulo}")
st.divider()

if len(df_filtered) > 1:
    col1, col2, col3, col4 = st.columns(4)
    
    promedio_cumpl = df_filtered['Cumpl. Año'].mean()
    total_kpis = len(df_filtered)
    
    en_meta = df_filtered[
        df_filtered['Estado Actual'].astype(str).str.contains("Cumple", case=False) & 
        ~df_filtered['Estado Actual'].astype(str).str.contains("No", case=False)
    ]
    fuera_meta = df_filtered[
        df_filtered['Estado Actual'].astype(str).str.contains("No", case=False)
    ]
    
    if not df_filtered.empty:
        idx_mejor = df_filtered['Cumpl. Año'].idxmax()
        mejor_kpi = df_filtered.loc[idx_mejor]['Indicador']
        val_mejor = df_filtered.loc[idx_mejor]['Cumpl. Año']
    else:
        mejor_kpi = "-"
        val_mejor = 0
    
    col1.metric("Cumplimiento Global", f"{promedio_cumpl:.1f}%")
    col2.metric("Total Indicadores", total_kpis)
    col3.metric("✅ En Meta", len(en_meta))
    col4.metric("⚠️ Fuera de Meta", len(fuera_meta), delta_color="inverse")
    
    st.markdown("---")
    st.info(f"🏆 **Mejor Desempeño:** {mejor_kpi} (**{val_mejor:.1f}%**)")
    st.divider()

# --- 6. SECCIÓN DE ALERTAS ---
kpis_rojos = df_filtered[df_filtered['Estado Actual'].astype(str).str.contains("No", case=False)]

if indicador_sel == "Todos" and len(kpis_rojos) > 0:
    st.subheader("🔥 Alertas Prioritarias (Fuera de Meta)")
    st.warning(f"Se requieren acciones correctivas en {len(kpis_rojos)} indicadores.")
    
    cols_alerta_base = ['Indicador', 'Proceso', 'Meta']
    cols_alerta_final = ['Prom. Año', 'Cumpl. Año', 'Estado Actual']
    
    if mostrar_meses_tabla:
        cols_alerta_mostrar = cols_alerta_base + meses_seleccionados + cols_alerta_final
    else:
        cols_alerta_mostrar = cols_alerta_base + cols_alerta_final
    
    format_dict_meses = {m: "{:.2f}%" for m in meses_seleccionados}
    format_dict_meta = {'Meta': "{:.2f}%", 'Prom. Año': "{:.2f}%", 'Cumpl. Año': "{:.0f}%"}
    format_total = {**format_dict_meta, **format_dict_meses}

    st.dataframe(
        kpis_rojos[cols_alerta_mostrar].style
        .bar(subset=['Cumpl. Año'], color='#00C4FF', vmin=0, vmax=120)
        .format(format_total),
        use_container_width=True,
        hide_index=True
    )
    st.divider()

# --- 7. GRÁFICO TENDENCIA (Para selección individual) ---
if indicador_sel != "Todos" and len(df_filtered) == 1:
    row = df_filtered.iloc[0]
    st.subheader(f"📈 Tendencia Mensual: {row['Indicador']}")
    
    vals = []
    meses_grafica = []
    for m in meses_seleccionados:
        if m in row:
            vals.append(row[m])
            meses_grafica.append(m)
        else:
            vals.append(0)
            meses_grafica.append(m)
            
    fig_ind = go.Figure()
    fig_ind.add_trace(go.Scatter(
        x=meses_grafica, y=vals, mode='lines+markers+text',
        name='Real', line=dict(color='#00C4FF', width=3),
        text=[f"{v:.2f}%" for v in vals], textposition="top center"
    ))
    fig_ind.add_hline(y=row['Meta'], line_dash="dash", line_color="red", annotation_text=f"Meta: {row['Meta']}%")
    fig_ind.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig_ind, use_container_width=True)

# --- 8. TABLA DE DETALLE ---
st.subheader("📋 Detalle de Indicadores")

cols_base = ['Indicador', 'Meta']
cols_final = ['Prom. Año', 'Cumpl. Año', 'Estado Actual']

if mostrar_meses_tabla:
    cols_mostrar = cols_base + meses_seleccionados + cols_final
else:
    cols_mostrar = cols_base + cols_final

def colorear_estado(val):
    color = '#d32f2f' if 'No' in str(val) else '#2e7d32' 
    return f'color: {color}; font-weight: bold'

format_dict_meses = {m: "{:.2f}%" for m in meses_seleccionados}
format_dict_gral = {'Meta': "{:.2f}%", 'Prom. Año': "{:.2f}%", 'Cumpl. Año': "{:.0f}%"}
todos_los_formatos = {**format_dict_gral, **format_dict_meses}

column_config_dinamica = {
    "Indicador": st.column_config.TextColumn("Indicador", width="medium"),
    "Meta": st.column_config.NumberColumn("Meta", format="%.2f%%"),
    "Prom. Año": st.column_config.NumberColumn("Resultado Año", format="%.2f%%"),
    "Cumpl. Año": st.column_config.ProgressColumn(
        "Cumplimiento", format="%.0f%%", min_value=0, max_value=120
    ),
}

if mostrar_meses_tabla:
    for m in meses_seleccionados:
        column_config_dinamica[m] = st.column_config.NumberColumn(m, format="%.2f%%")

st.dataframe(
    df_filtered[cols_mostrar].style
    .bar(subset=['Cumpl. Año'], color='#00C4FF', vmin=0, vmax=120)
    .applymap(colorear_estado, subset=['Estado Actual'])
    .format(todos_los_formatos), 
    use_container_width=True,
    hide_index=True,
    column_config=column_config_dinamica
)

# --- 9. GRÁFICO COMPARATIVO (AHORA SIEMPRE VISIBLE) ---
# Hemos quitado el "if len > 1" para que salga siempre.
if not df_filtered.empty:
    st.subheader("📊 Comparativo: Meta vs Resultado")
    
    fig_bar = go.Figure()
    
    fig_bar.add_trace(go.Bar(
        x=df_filtered['Indicador'], y=df_filtered['Meta'], 
        name='Meta', marker_color='lightgray', 
        text=df_filtered['Meta'], texttemplate='%{text:.2f}%'
    ))
    
    fig_bar.add_trace(go.Bar(
        x=df_filtered['Indicador'], y=df_filtered['Prom. Año'], 
        name='Resultado Real', marker_color='#00C4FF', 
        text=df_filtered['Prom. Año'], texttemplate='%{text:.2f}%'
    ))
    
    fig_bar.update_layout(barmode='group', height=500, xaxis_tickangle=-45)
    
    st.plotly_chart(fig_bar, use_container_width=True)