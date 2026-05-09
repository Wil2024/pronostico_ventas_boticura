import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="BotiCura · Simulador de Pronóstico de Ventas",
    page_icon="💊",
    layout="wide",
)

# CSS corporativo BotiCura
st.markdown("""
<style>
    .main { background-color: #F7F9FC; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-left: 4px solid #1A73E8;
        padding: 14px 18px;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .metric-card.green  { border-left-color: #34A853; }
    .metric-card.orange { border-left-color: #FBBC04; }
    .metric-card.red    { border-left-color: #EA4335; }
    .section-title {
        font-size: 1.1rem; font-weight: 700;
        color: #1A73E8;
        margin-top: 18px; margin-bottom: 6px;
    }
    .insight-box {
        background: #EAF2FF;
        border-radius: 8px;
        padding: 12px 16px; margin-top: 8px;
        font-size: 0.88rem; color: #1A1A2E;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("## 💊")
with col_title:
    st.markdown("## BotiCura S.A.C. · Simulador de pronóstico de ventas")
    st.caption("Análisis Predictivo · Forecasting de Series de Tiempo · 2020 – 2027")

st.divider()

# ──────────────────────────────────────────────
# PROPHET CHECK
# ──────────────────────────────────────────────
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
# Tickets promedio por si falta la columna Unidades
TICKET_PROMEDIO_CATEGORIA = {
    "Medicamentos":   10.0,
    "Vitaminas":      36.0,
    "Dermocosmética": 63.3,
    "Bebés":          62.7,
    "Nutrición":      95.0,
    "Veterinaria":    44.3,
    "General":        30.0,
}

COLOR_CATEGORIA = {
    "Medicamentos":   "#1A73E8",
    "Vitaminas":      "#34A853",
    "Dermocosmética": "#E91E8C",
    "Bebés":          "#FF9800",
    "Nutrición":      "#00BCD4",
    "Veterinaria":    "#9C27B0",
    "General":        "#607D8B",
}

def calcular_kpis(real, pronostico, simbolo="S/."):
    idx = real.index.intersection(pronostico.index)
    r, p = real[idx], pronostico[idx]
    if len(r) == 0:
        return {}
  
    mae  = mean_absolute_error(r, p)
    rmse = np.sqrt(mean_squared_error(r, p))
    mape = np.mean(np.abs((r - p) / r.replace(0, np.nan))) * 100
    rd   = r.diff().dropna()
    pd_  = p.diff().dropna()
    direction = ((rd > 0) == (pd_ > 0)).mean() * 100
    sesgo = np.mean(p - r)
    return {
        f"MAE ({simbolo})":              round(mae, 2),
        f"RMSE ({simbolo})":             round(rmse, 2),
        "MAPE (%)":               round(mape, 2),
        "Precisión Dirección (%)": round(direction, 2),
        f"Sesgo ({simbolo})":            round(sesgo, 2),
    }

def run_forecast(series, model_type, steps, confianza):
    """Retorna (forecast_series, conf_int_df)"""
    alpha = 1 - confianza / 100
    z = {90: 1.645, 91: 1.70, 92: 1.75, 93: 1.81, 94: 1.88,
         95: 1.96, 96: 2.05, 97: 2.17, 98: 2.33, 99: 2.576}.get(confianza, 1.96)

    if model_type == "Holt-Winters":
        m = ExponentialSmoothing(series, seasonal="add", seasonal_periods=12, trend="add").fit()
        fc = m.forecast(steps)
        sd = np.std(m.resid)
        ci = pd.DataFrame({"Limite_Inferior": fc - z * sd, "Limite_Superior": fc + z * sd})
        return fc, ci

    elif model_type == "ARIMA":
        m = ARIMA(series, order=(1, 1, 1)).fit()
        fo = m.get_forecast(steps)
        fc = fo.predicted_mean
        ci = fo.conf_int(alpha=alpha)
        ci.columns = ["Limite_Inferior", "Limite_Superior"]
        return fc, ci

    elif model_type == "SARIMA":
        m = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        fo = m.get_forecast(steps)
        fc = fo.predicted_mean
        ci = fo.conf_int(alpha=alpha)
        ci.columns = ["Limite_Inferior", "Limite_Superior"]
        return fc, ci

    elif model_type == "Prophet" and prophet_available:
        df_p = series.reset_index()
        df_p.columns = ["ds", "y"]
        df_p["ds"] = pd.to_datetime(df_p["ds"])
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_p)
        future_df = pd.DataFrame({"ds": pd.date_range(series.index[-1], periods=steps + 1, freq="MS")[1:]})
        fc_df = m.predict(future_df)
        fc = fc_df.set_index("ds")["yhat"]
        ci = fc_df.set_index("ds")[["yhat_lower", "yhat_upper"]]
        ci.columns = ["Limite_Inferior", "Limite_Superior"]
        return fc, ci

    raise ValueError("Modelo no soportado")

def generar_excel(dfs_dict):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, df in dfs_dict.items():
            df.to_excel(writer, sheet_name=sheet[:31])
    return buf.getvalue()

# ──────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────
st.markdown('<p class="section-title">📂 Paso 1 · Cargar Dataset</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Suba **ventas_mensuales_boticura.xlsx**", type=["xlsx", "csv"])

if not uploaded_file:
    st.info("💡 Suba **ventas_mensuales_boticura.xlsx** para acceder al análisis completo.")
    st.stop()

try:
    if uploaded_file.name.endswith(".csv"):
        data_raw = pd.read_csv(uploaded_file, parse_dates=["Fecha"])
    else:
        data_raw = pd.read_excel(uploaded_file, parse_dates=["Fecha"])
except Exception as e:
    st.error(f"Error al leer el archivo: {e}")
    st.stop()

data_raw["Fecha"] = pd.to_datetime(data_raw["Fecha"])
data_raw.sort_values("Fecha", inplace=True)
data_raw.ffill(inplace=True)

required_cols = ["Fecha", "Departamento", "Distrito", "Canal_Venta", "Ventas"]
for c in required_cols:
    if c not in data_raw.columns:
        st.error(f"Columna requerida ausente: **{c}**")
        st.stop()

has_categoria = "Categoria" in data_raw.columns
has_producto  = "Producto"  in data_raw.columns
has_unidades  = "Unidades" in data_raw.columns

st.success(f"✅ Dataset cargado · {len(data_raw):,} filas · {data_raw['Fecha'].min().strftime('%b %Y')} → {data_raw['Fecha'].max().strftime('%b %Y')}")

# ──────────────────────────────────────────────
# BARRA LATERAL – FILTROS GLOBALES
# ──────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/pharmacy-shop.png", width=60)
st.sidebar.title("⚙️ Configuración")
st.sidebar.markdown("---")

# NUEVO: Selector de Métrica (Ventas vs Unidades)
st.sidebar.markdown("**🎯 Variable Objetivo**")
metrica_label = st.sidebar.radio("Pronosticar por:", ["Ventas (S/.)", "Unidades (Cant.)"])
target_col = "Ventas" if "Ventas" in metrica_label else "Unidades"
simbolo = "S/." if target_col == "Ventas" else "Unds"

st.sidebar.markdown("---")
st.sidebar.markdown("**🌎 Filtros Geográficos**")
deptos = ["Todos"] + sorted(data_raw["Departamento"].dropna().unique().tolist())
sel_depto = st.sidebar.selectbox("Departamento", deptos)

dist_opts = ["Todos"]
if sel_depto != "Todos":
    dist_opts += sorted(data_raw[data_raw["Departamento"] == sel_depto]["Distrito"].dropna().unique().tolist())
sel_dist = st.sidebar.selectbox("Distrito", dist_opts)

canales = ["Todos"] + sorted(data_raw["Canal_Venta"].dropna().unique().tolist())
sel_canal = st.sidebar.selectbox("Canal de Venta", canales)

st.sidebar.markdown("---")
st.sidebar.markdown("**📦 Granularidad de Análisis**")
nivel_opciones = ["Total (consolidado)"]
if has_categoria:
    nivel_opciones.append("Por Categoría")
    nivel_opciones.append("Por Categoría – Top Productos")
nivel = st.sidebar.radio("Nivel de análisis", nivel_opciones)

sel_categoria = "General"
sel_producto = "Todos"

if "Categoría" in nivel and has_categoria:
    categorias = ["Todas"] + sorted(data_raw["Categoria"].dropna().unique().tolist())
    sel_categoria = st.sidebar.selectbox("Categoría", categorias)
    
    if has_producto:
        if sel_categoria != "Todas":
            productos = ["Todos"] + sorted(data_raw[data_raw["Categoria"] == sel_categoria]["Producto"].dropna().unique().tolist())
        else:
            productos = ["Todos"] + sorted(data_raw["Producto"].dropna().unique().tolist())
        sel_producto = st.sidebar.selectbox("Producto", productos)

st.sidebar.markdown("---")
st.sidebar.markdown("**🔮 Modelo de Pronóstico**")
model_opts = ["Holt-Winters", "ARIMA", "SARIMA"]
if prophet_available:
    model_opts.append("Prophet")
model_type = st.sidebar.selectbox("Modelo", model_opts)
confianza  = st.sidebar.slider("Intervalo de Confianza (%)", 90, 99, 95)

horizonte_label = st.sidebar.selectbox("Horizonte de Pronóstico", ["12 meses (hasta dic 2026)", "20 meses (hasta dic 2026 + 2027 parcial)", "32 meses (hasta dic 2027)"])
horizonte_map = {"12 meses (hasta dic 2026)": 12, "20 meses (hasta dic 2026 + 2027 parcial)": 20, "32 meses (hasta dic 2027)": 32}
horizonte = horizonte_map[horizonte_label]

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Diseñado por **Wilton Torvisco** · Business Intelligence")

# ──────────────────────────────────────────────
# FILTRAR DATOS
# ──────────────────────────────────────────────
df = data_raw.copy()
if sel_depto != "Todos": df = df[df["Departamento"] == sel_depto]
if sel_dist  != "Todos": df = df[df["Distrito"]     == sel_dist]
if sel_canal != "Todos": df = df[df["Canal_Venta"]  == sel_canal]

if has_categoria and "Categoría" in nivel:
    if sel_categoria != "Todas": df = df[df["Categoria"] == sel_categoria]
    if has_producto and sel_producto != "Todos": df = df[df["Producto"] == sel_producto]

if df.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# NUEVO: Lógica de agregación para incluir Ventas y Unidades reales
agg_dict = {"Ventas": "sum"}
if has_unidades:
    agg_dict["Unidades"] = "sum"
if "Festivo" in df.columns:
    agg_dict["Festivo"] = "max"

ts_mensual = df.resample("MS", on="Fecha").agg(agg_dict).reset_index().set_index("Fecha")

# Si se pide Unidades pero el Excel no las tiene, las estimamos
if target_col == "Unidades" and not has_unidades:
    ticket = TICKET_PROMEDIO_CATEGORIA.get(sel_categoria, 28.5) if sel_categoria != "Todas" else 30.0
    ts_mensual["Unidades"] = (ts_mensual["Ventas"] / ticket).round().astype(int)

# ──────────────────────────────────────────────
# TABS PRINCIPALES
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard Histórico",
    "🔮 Pronóstico Predictivo",
    "📦 Análisis por Categoría",
    "🏆 Top Productos",
    "📚 Guía Metodológica MBA",
])

# ══════════════════════════════════════════════
# TAB 1 – DASHBOARD HISTÓRICO
# ══════════════════════════════════════════════
with tab1:
    texto_titulo = f"📊 Evolución Histórica ({metrica_label}) · BotiCura 2020–2026"
    if has_categoria and "Categoría" in nivel:
        if sel_categoria != "Todas": texto_titulo += f" | Cat: {sel_categoria}"
        if sel_producto != "Todos":  texto_titulo += f" | Prod: {sel_producto}"
            
    st.markdown(f'<p class="section-title">{texto_titulo}</p>', unsafe_allow_html=True)

    # KPI Cards dinámicos
    total_historico = ts_mensual[target_col].sum()
    ultimo_mes      = ts_mensual[target_col].iloc[-1]
    mes_hace12      = ts_mensual[target_col].iloc[-13] if len(ts_mensual) > 12 else ts_mensual[target_col].iloc[0]
    crecimiento_yoy = ((ultimo_mes - mes_hace12) / mes_hace12 * 100) if mes_hace12 else 0
    mes_peak_str    = ts_mensual[target_col].idxmax().strftime("%b %Y")
    valor_peak      = ts_mensual[target_col].max()

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric(f"💰 Total Histórico ({simbolo})", f"{total_historico:,.0f} {simbolo}")
    with k2: st.metric(f"📅 Último Mes Registrado", f"{ultimo_mes:,.0f} {simbolo}", delta=f"{crecimiento_yoy:+.1f}% vs año ant.")
    with k3: st.metric("🏆 Mes con Mayor Volumen", mes_peak_str, delta=f"{valor_peak:,.0f} {simbolo}")
    with k4: st.metric("📈 Promedio Mensual", f"{ts_mensual[target_col].mean():,.0f} {simbolo}")

    st.divider()

    # Gráfico principal
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=ts_mensual.index, y=ts_mensual[target_col],
            mode="lines+markers", name=f"{target_col}",
            line=dict(color=COLOR_CATEGORIA.get(sel_categoria, "#1A73E8"), width=2.5),
        ))
        mm3 = ts_mensual[target_col].rolling(3).mean()
        fig_hist.add_trace(go.Scatter(
            x=ts_mensual.index, y=mm3, mode="lines", name="Media Móvil 3m",
            line=dict(dash="dot", color="#FF9800", width=1.8),
        ))
        fig_hist.update_layout(title=f"Evolución de {target_col} Mensuales", height=380, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        ventas_canal = df.groupby("Canal_Venta")[target_col].sum().reset_index()
        fig_canal = px.pie(ventas_canal, names="Canal_Venta", values=target_col, title=f"Participación por Canal ({simbolo})", color_discrete_sequence=["#1A73E8", "#34A853"])
        fig_canal.update_layout(height=380)
        st.plotly_chart(fig_canal, use_container_width=True)

    # Heatmap
    st.markdown(f'<p class="section-title">🌡️ Heatmap de Estacionalidad ({target_col} por Mes × Año)</p>', unsafe_allow_html=True)
    heat_df = ts_mensual[target_col].reset_index()
    heat_df["Año"] = heat_df["Fecha"].dt.year
    heat_df["Mes"] = heat_df["Fecha"].dt.strftime("%b")
    meses_orden = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    heat_pivot = heat_df.pivot_table(index="Año", columns="Mes", values=target_col, aggfunc="sum")
    cols_presentes = [m for m in meses_orden if m in heat_pivot.columns]
    heat_pivot = heat_pivot[cols_presentes]
    fig_heat = px.imshow(heat_pivot, text_auto=".2s", color_continuous_scale="Blues", title=f"Heatmap de {target_col}")
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – PRONÓSTICO (AHORA CON DESCARGA EXCEL)
# ══════════════════════════════════════════════
with tab2:
    st.markdown(f'<p class="section-title">🔮 Pronóstico de {target_col} · Modelo: {model_type}</p>', unsafe_allow_html=True)

    corte_train = "2024-12-31"
    corte_test  = "2025-01-01"
    train_series = ts_mensual.loc[:corte_train, target_col]
    test_series  = ts_mensual.loc[corte_test:,  target_col]

    if len(train_series) < 24:
        st.warning("Se necesitan al menos 24 meses de datos históricos para entrenar el modelo.")
        st.stop()

    with st.spinner(f"Entrenando modelo {model_type}..."):
        try:
            fc_test, ci_test = run_forecast(train_series, model_type, len(test_series), confianza)
            fc_test.index = test_series.index[:len(fc_test)]
            if ci_test is not None: ci_test.index = test_series.index[:len(ci_test)]
        except Exception as e:
            st.error(f"Error en el modelo: {e}")
            st.stop()

    kpis = calcular_kpis(test_series, fc_test, simbolo)

    st.markdown("**📐 Métricas de Evaluación (back-test 2025)**")
    kc = st.columns(5)
    icons = ["📏", "📐", "📊", "🎯", "⚖️"]
    colores = ["metric-card", "metric-card", "metric-card orange", "metric-card green", "metric-card"]
    for i, (k, v) in enumerate(kpis.items()):
        with kc[i]:
            st.markdown(f'<div class="{colores[i]}">{icons[i]} <b>{k}</b><br><span style="font-size:1.3rem">{v:,.2f}</span></div>', unsafe_allow_html=True)

    st.divider()

    # Pronóstico Futuro
    st.markdown(f'<p class="section-title">📅 Proyección de Pedidos/Ventas Futuras · {horizonte_label}</p>', unsafe_allow_html=True)

    full_series = ts_mensual[target_col]
    future_idx = pd.date_range(start=full_series.index[-1] + pd.DateOffset(months=1), periods=horizonte, freq="MS")

    with st.spinner("Generando pronóstico futuro..."):
        fc_fut, ci_fut = run_forecast(full_series, model_type, horizonte, confianza)
        fc_fut.index = future_idx
        ci_fut.index = future_idx

    fc_fut = fc_fut.clip(lower=0)
    ci_fut["Limite_Inferior"] = ci_fut["Limite_Inferior"].clip(lower=0)

    # Gráfico Futuro
    fig_fut = go.Figure()
    fig_fut.add_trace(go.Scatter(x=full_series.index, y=full_series, mode="lines+markers", name=f"Histórico ({simbolo})", line=dict(color="#607D8B", width=2)))
    fig_fut.add_trace(go.Scatter(x=fc_fut.index, y=fc_fut, mode="lines+markers", name="Pronóstico", line=dict(color="#FF9800", width=2.5, dash="dot")))
    if ci_fut is not None:
        fig_fut.add_trace(go.Scatter(
            x=ci_fut.index.tolist() + ci_fut.index[::-1].tolist(),
            y=ci_fut["Limite_Superior"].tolist() + ci_fut["Limite_Inferior"][::-1].tolist(),
            fill="toself", fillcolor="rgba(255,152,0,0.15)", line=dict(color="rgba(255,255,255,0)"), name=f"Confianza {confianza}%"
        ))
    fig_fut.update_layout(height=450, legend=dict(orientation="h", y=-0.2), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_fut, use_container_width=True)

    # NUEVO: Descarga en Excel
    df_export = pd.DataFrame({
        "Mes": fc_fut.index.strftime("%Y-%m"),
        f"Pronostico_{target_col}": fc_fut.round(2),
        "Stock_Seguridad_Max (Lim.Sup)": ci_fut["Limite_Superior"].round(2),
        "Minimo_Requerido (Lim.Inf)": ci_fut["Limite_Inferior"].round(2)
    })
    
    excel_data = generar_excel({
        "Pronostico_Proyectado": df_export,
        "Historico_Consolidado": ts_mensual[[target_col]],
        "Metricas_Evaluacion": pd.DataFrame([kpis])
    })
    
    st.download_button(
        label="📥 Descargar Reporte Completo (Excel)",
        data=excel_data,
        file_name=f"Pronostico_{target_col}_{sel_categoria}_{sel_producto}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ══════════════════════════════════════════════
# TAB 3 – ANÁLISIS POR CATEGORÍA
# ══════════════════════════════════════════════
with tab3:
    if not has_categoria:
        st.warning("El dataset no incluye columna de Categoría.")
    else:
        st.markdown(f'<p class="section-title">📊 Composición de {target_col} por Categoría</p>', unsafe_allow_html=True)
        
        resumen_cat_hist = df.groupby("Categoria").agg(
            Total_Metrica=(target_col, "sum")
        ).reset_index().sort_values("Total_Metrica", ascending=False)
        
        resumen_cat_hist["Participación (%)"] = (resumen_cat_hist["Total_Metrica"] / resumen_cat_hist["Total_Metrica"].sum()) * 100
        
        c1, c2 = st.columns([1, 1])
        with c1: st.dataframe(resumen_cat_hist.style.format({"Total_Metrica": f"{simbolo} {{:,.0f}}", "Participación (%)": "{:.1f}%"}), use_container_width=True, hide_index=True)
        with c2:
            fig_pie_cat = px.pie(resumen_cat_hist, names="Categoria", values="Total_Metrica", hole=0.4, color="Categoria", color_discrete_map=COLOR_CATEGORIA)
            fig_pie_cat.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie_cat, use_container_width=True)

        st.markdown(f'<p class="section-title">🔮 Pronóstico por Categoría ({target_col})</p>', unsafe_allow_html=True)
        fig_fc_cat = go.Figure()
        
        for cat in resumen_cat_hist["Categoria"].tolist():
            serie_c = df[df["Categoria"] == cat].groupby("Fecha")[target_col].sum().resample("MS").sum().asfreq("MS", fill_value=0)
            if len(serie_c) >= 24:
                try:
                    fc_c, _ = run_forecast(serie_c, model_type, horizonte, confianza)
                    fig_fc_cat.add_trace(go.Scatter(x=fc_c.index, y=fc_c.clip(lower=0), name=cat, mode="lines+markers", line=dict(color=COLOR_CATEGORIA.get(cat, "#000"), width=2)))
                except Exception: pass
        
        fig_fc_cat.update_layout(height=450, legend=dict(orientation="h", y=-0.2), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_fc_cat, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 – TOP PRODUCTOS
# ══════════════════════════════════════════════
with tab4:
    if not has_producto:
        st.warning("⚠️ El dataset no contiene la columna **Producto**.")
    else:
        st.markdown(f'<p class="section-title">🏆 Pronóstico de {target_col} por Producto</p>', unsafe_allow_html=True)
        
        cats_disponibles = sorted(df["Categoria"].dropna().unique().tolist()) if has_categoria else ["General"]
        sel_cat_tab4 = st.selectbox("📦 Seleccionar Categoría:", cats_disponibles, key="cat_tab4")
        
        df_prod_cat = df[df["Categoria"] == sel_cat_tab4].copy() if has_categoria else df.copy()

        ranking_prod = df_prod_cat.groupby("Producto")[target_col].sum().reset_index().sort_values(target_col, ascending=False)

        st.markdown(f"**1️⃣ Ranking Histórico de {target_col}**")
        st.dataframe(ranking_prod.style.format({target_col: f"{simbolo} {{:,.0f}}"}), hide_index=True, use_container_width=True)

        st.markdown(f"**2️⃣ Pronóstico Futuro – Top 3 Productos**")
        top_3_prods = ranking_prod["Producto"].head(3).tolist()
        fig_fc_prod = go.Figure()

        for prod in top_3_prods:
            serie_p = df_prod_cat[df_prod_cat["Producto"] == prod].groupby("Fecha")[target_col].sum().resample("MS").sum().asfreq("MS", fill_value=0)
            if len(serie_p) >= 24:
                try:
                    fc_p, _ = run_forecast(serie_p, model_type, horizonte, confianza)
                    fig_fc_prod.add_trace(go.Scatter(x=fc_p.index, y=fc_p.clip(lower=0), name=prod, mode="lines+markers", line=dict(width=2)))
                except Exception: pass

        fig_fc_prod.update_layout(height=400, legend=dict(orientation="h", y=-0.2), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_fc_prod, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 – GUÍA METODOLÓGICA MBA
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">🎓 Guía de Toma de Decisiones Estratégicas para Estudiantes MBA</p>', unsafe_allow_html=True)
    
    with st.expander("💼 ¿Cómo utilizar este dashboard para la estrategia corporativa?", expanded=True):
        st.markdown("""
- 📉 **Optimización de Inventarios (Unidades)**: Descargue el Excel y compare el *Límite Inferior* vs. *Límite Superior*. Si su producto tiene alta rentabilidad y el costo de almacenamiento es bajo, pida el volumen cercano al Límite Superior para evitar quiebres de stock.
- 🛍️ **Estrategia Comercial (Ventas en S/.)**: Identifique los picos financieros. Planifique sus campañas de Trade Marketing basándose en los valles y picos del modelo.
- 💰 **Presupuesto 2026–2027**: Utilice las métricas exportadas en Excel como input cuantitativo e irrefutable para la defensa de su Plan Operativo Anual (POA).
        """)

# FOOTER
st.divider()
st.markdown("<div style='text-align:center; font-size:12px; color:#607D8B; margin-top:20px;'>BotiCura S.A.C. · Caso de Estudio MBA</div>", unsafe_allow_html=True)