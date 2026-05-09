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
        color: #1A73E8; margin-top: 18px; margin-bottom: 6px;
    }
    .insight-box {
        background: #EAF2FF; border-radius: 8px;
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
# Tickets promedio reales extraídos del dataset de transacciones
TICKET_PROMEDIO_CATEGORIA = {
    "Medicamentos":   10.0,   # (Ibuprofeno S/12 + Paracetamol S/8) / 2
    "Vitaminas":      36.0,   # (Vitamina C S/25 + Omega3 S/48 + Magnesio S/35) / 3
    "Dermocosmética": 63.3,   # (Agua Micelar S/45 + Bloqueador S/65 + Crema Facial S/80) / 3
    "Bebés":          62.7,   # (Leche S/95 + Pañales S/75 + Toallitas S/18) / 3
    "Nutrición":      95.0,   # Colágeno S/95
    "Veterinaria":    44.3,   # (Antipulgas S/65 + Shampoo S/28 + Vitaminas Mascotas S/40) / 3
    "General":        30.0,
}

# Productos reales del dataset de transacciones por categoría
PRODUCTOS_ESTRELLA = {
    "Medicamentos":   ["Ibuprofeno", "Paracetamol"],
    "Vitaminas":      ["Vitamina C", "Omega 3", "Magnesio"],
    "Dermocosmética": ["Crema Facial", "Bloqueador Solar", "Agua Micelar"],
    "Bebés":          ["Leche Infantil", "Pañales", "Toallitas Húmedas"],
    "Nutrición":      ["Colágeno"],
    "Veterinaria":    ["Antipulgas", "Vitaminas Mascotas", "Shampoo Canino"],
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

def calcular_kpis(real, pronostico):
    idx = real.index.intersection(pronostico.index)
    r, p = real[idx], pronostico[idx]
    if len(r) == 0:
        return {}
    mae  = mean_absolute_error(r, p)
    rmse = np.sqrt(mean_squared_error(r, p))
    mape = np.mean(np.abs((r - p) / r.replace(0, np.nan))) * 100
    rd   = r.diff().dropna(); pd_ = p.diff().dropna()
    direction = ((rd > 0) == (pd_ > 0)).mean() * 100
    sesgo = np.mean(p - r)
    return {
        "MAE (S/.)":              round(mae, 2),
        "RMSE (S/.)":             round(rmse, 2),
        "MAPE (%)":               round(mape, 2),
        "Precisión Dirección (%)": round(direction, 2),
        "Sesgo (S/.)":            round(sesgo, 2),
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
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for sheet, df in dfs_dict.items():
            df.to_excel(w, sheet_name=sheet[:31])
    buf.seek(0)
    return buf

# ──────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────
st.markdown('<p class="section-title">📂 Paso 1 · Cargar Dataset</p>', unsafe_allow_html=True)

with st.expander("ℹ️ Formato esperado del archivo", expanded=False):
    st.markdown("""
    El archivo Excel debe contener las columnas: `Fecha`, `Departamento`, `Distrito`,
    `Canal_Venta`, `Ventas`, `Festivo`, `Eventos`.  
    Opcionalmente: `Categoria`, `Producto`, `Costos_Operativos`, `Marketing`.  
    Una fila = un mes × distrito × canal (o categoría/producto si están disponibles).
    """)

uploaded_file = st.file_uploader("Suba **ventas_mensuales_boticura.xlsx**", type=["xlsx"])

if not uploaded_file:
    st.info("💡 Suba **ventas_mensuales_boticura.xlsx** (productos y categorías alineados con el dataset de transacciones reales de BotiCura) para acceder al análisis completo.")
    st.stop()

# ──────────────────────────────────────────────
# PROCESAMIENTO BASE
# ──────────────────────────────────────────────
try:
    data_raw = pd.read_excel(uploaded_file, parse_dates=["Fecha"])
except Exception as e:
    st.error(f"Error al leer el archivo: {e}")
    st.stop()

data_raw["Fecha"] = pd.to_datetime(data_raw["Fecha"])
data_raw.sort_values("Fecha", inplace=True)
data_raw.ffill(inplace=True)

required_cols = ["Fecha", "Departamento", "Distrito", "Canal_Venta", "Ventas", "Festivo", "Eventos"]
for c in required_cols:
    if c not in data_raw.columns:
        st.error(f"Columna requerida ausente: **{c}**")
        st.stop()

has_categoria = "Categoria" in data_raw.columns
has_producto  = "Producto"  in data_raw.columns
has_costos    = "Costos_Operativos" in data_raw.columns

st.success(f"✅ Dataset cargado · {len(data_raw):,} filas · {data_raw['Fecha'].min().strftime('%b %Y')} → {data_raw['Fecha'].max().strftime('%b %Y')}")

# ──────────────────────────────────────────────
# BARRA LATERAL – FILTROS GLOBALES
# ──────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/pharmacy-shop.png", width=60)
st.sidebar.title("⚙️ Configuración")
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

categorias_disponibles = list(PRODUCTOS_ESTRELLA.keys())
sel_categoria = "General"
if "Categoría" in nivel and has_categoria:
    sel_categoria = st.sidebar.selectbox(
        "Categoría", sorted(data_raw["Categoria"].dropna().unique().tolist())
    )

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
st.sidebar.caption("© 2026 Diseñado por **Wilton Torvisco** · Business Intelligence & Business Analytics")

# ──────────────────────────────────────────────
# FILTRAR DATOS
# ──────────────────────────────────────────────
df = data_raw.copy()
if sel_depto != "Todos": df = df[df["Departamento"] == sel_depto]
if sel_dist  != "Todos": df = df[df["Distrito"]     == sel_dist]
if sel_canal != "Todos": df = df[df["Canal_Venta"]  == sel_canal]
if has_categoria and "Categoría" in nivel:
    df = df[df["Categoria"] == sel_categoria]

if df.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# Serie mensual principal
ts_mensual = df.resample("MS", on="Fecha").agg(
    Ventas=("Ventas", "sum"),
    Festivo=("Festivo", "max"),
).reset_index().set_index("Fecha")

# Estimación de unidades
ticket = TICKET_PROMEDIO_CATEGORIA.get(sel_categoria, 28.5)
ts_mensual["Unidades"] = (ts_mensual["Ventas"] / ticket).round().astype(int)

if has_costos:
    ts_costos = df.resample("MS", on="Fecha")["Costos_Operativos"].sum()
    ts_mensual["Margen_Bruto"] = ts_mensual["Ventas"] - ts_costos.reindex(ts_mensual.index, fill_value=0)

# ──────────────────────────────────────────────
# TABS PRINCIPALES
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard Histórico",
    "🔮 Pronóstico de Ventas",
    "📦 Análisis por Categoría",
    "🏆 Top Productos",
    "📚 Guía Metodológica MBA",
])

# ══════════════════════════════════════════════
# TAB 1 – DASHBOARD HISTÓRICO
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">📊 Evolución Histórica de Ventas · BotiCura 2020–2026</p>', unsafe_allow_html=True)

    # KPI Cards
    ventas_total     = ts_mensual["Ventas"].sum()
    ventas_ultimo    = ts_mensual["Ventas"].iloc[-1]
    ventas_hace12    = ts_mensual["Ventas"].iloc[-13] if len(ts_mensual) > 12 else ts_mensual["Ventas"].iloc[0]
    crecimiento_yoy  = ((ventas_ultimo - ventas_hace12) / ventas_hace12 * 100) if ventas_hace12 else 0
    mes_peak         = ts_mensual["Ventas"].idxmax().strftime("%b %Y")
    ventas_peak      = ts_mensual["Ventas"].max()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("💰 Ventas Totales Históricas", f"S/ {ventas_total:,.0f}")
    with k2:
        st.metric("📅 Último Mes Registrado", f"S/ {ventas_ultimo:,.0f}", delta=f"{crecimiento_yoy:+.1f}% vs año anterior")
    with k3:
        st.metric("🏆 Mes con Mayor Venta", mes_peak, delta=f"S/ {ventas_peak:,.0f}")
    with k4:
        prom_mensual = ts_mensual["Ventas"].mean()
        st.metric("📈 Promedio Mensual", f"S/ {prom_mensual:,.0f}")

    st.divider()

    # Gráfico principal + descomposición
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=ts_mensual.index, y=ts_mensual["Ventas"],
            mode="lines+markers", name="Ventas S/.",
            line=dict(color=COLOR_CATEGORIA.get(sel_categoria, "#1A73E8"), width=2.5),
            marker=dict(size=4),
        ))
        # Media móvil 3m
        mm3 = ts_mensual["Ventas"].rolling(3).mean()
        fig_hist.add_trace(go.Scatter(
            x=ts_mensual.index, y=mm3,
            mode="lines", name="Media Móvil 3m",
            line=dict(dash="dot", color="#FF9800", width=1.8),
        ))
        # Anotaciones de shocks
        shocks = {
            "2020-03-01": ("COVID-19", "#EA4335"),
            "2020-12-01": ("Navidad Pico", "#34A853"),
            "2022-04-01": ("Huelga",       "#FBBC04"),
            "2023-07-01": ("Fiestas Patrias Pico", "#34A853"),
        }
        for fecha_s, (label_s, color_s) in shocks.items():
            try:
                y_val = ts_mensual.loc[fecha_s, "Ventas"]
                fig_hist.add_annotation(x=fecha_s, y=y_val, text=label_s,
                    showarrow=True, arrowhead=2, arrowcolor=color_s,
                    font=dict(size=9, color=color_s), ax=0, ay=-30)
            except Exception:
                pass
        fig_hist.update_layout(
            title="Ventas Mensuales BotiCura (S/.) con Media Móvil",
            xaxis_title="Fecha", yaxis_title="Ventas (S/.)",
            legend=dict(orientation="h", y=-0.2), height=380,
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        # Ventas por canal
        ventas_canal = df.groupby("Canal_Venta")["Ventas"].sum().reset_index()
        fig_canal = px.pie(ventas_canal, names="Canal_Venta", values="Ventas",
                           title="Participación por Canal",
                           color_discrete_sequence=["#1A73E8", "#34A853"])
        fig_canal.update_layout(height=380)
        st.plotly_chart(fig_canal, use_container_width=True)

    # Heatmap mensual
    st.markdown('<p class="section-title">🌡️ Heatmap de Estacionalidad (Ventas por Mes × Año)</p>', unsafe_allow_html=True)
    heat_df = ts_mensual["Ventas"].reset_index()
    heat_df["Año"] = heat_df["Fecha"].dt.year
    heat_df["Mes"] = heat_df["Fecha"].dt.strftime("%b")
    meses_orden = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    heat_pivot = heat_df.pivot_table(index="Año", columns="Mes", values="Ventas", aggfunc="sum")
    cols_presentes = [m for m in meses_orden if m in heat_pivot.columns]
    heat_pivot = heat_pivot[cols_presentes]
    fig_heat = px.imshow(
        heat_pivot, text_auto=".2s",
        color_continuous_scale="Blues",
        title="Ventas Mensuales por Año (S/.)",
        labels=dict(color="Ventas S/."),
    )
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Observe cómo los meses de julio (Fiestas Patrias), diciembre (Navidad) y mayo (Día de la Madre) muestran picos consistentes. Esto es clave para la <b>planificación de inventario y presupuesto</b> en cadenas de farmacia.</div>', unsafe_allow_html=True)

    # Ventas por departamento
    st.markdown('<p class="section-title">🗺️ Ventas por Departamento (Acumulado)</p>', unsafe_allow_html=True)
    ventas_depto = df.groupby("Departamento")["Ventas"].sum().reset_index().sort_values("Ventas", ascending=True)
    fig_depto = px.bar(ventas_depto, x="Ventas", y="Departamento", orientation="h",
                       title="Ventas Acumuladas por Departamento (S/.)",
                       color="Ventas", color_continuous_scale="Blues",
                       text_auto=".2s")
    fig_depto.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_depto, use_container_width=True)

    # YoY por año
    st.markdown('<p class="section-title">📈 Crecimiento Año a Año (YoY)</p>', unsafe_allow_html=True)
    ventas_anual = ts_mensual["Ventas"].resample("YE").sum().reset_index()
    ventas_anual.columns = ["Fecha", "Ventas"]
    ventas_anual["Año"] = pd.to_datetime(ventas_anual["Fecha"]).dt.year
    ventas_anual["YoY (%)"] = ventas_anual["Ventas"].pct_change() * 100
    fig_yoy = make_subplots(specs=[[{"secondary_y": True}]])
    fig_yoy.add_trace(go.Bar(x=ventas_anual["Año"], y=ventas_anual["Ventas"],
                              name="Ventas (S/.)", marker_color="#1A73E8"), secondary_y=False)
    fig_yoy.add_trace(go.Scatter(x=ventas_anual["Año"], y=ventas_anual["YoY (%)"],
                                  mode="lines+markers+text", name="Var. YoY (%)",
                                  text=ventas_anual["YoY (%)"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else ""),
                                  textposition="top center",
                                  line=dict(color="#EA4335", width=2),
                                  marker=dict(size=8)), secondary_y=True)
    fig_yoy.update_layout(title="Ventas Anuales y Variación YoY", height=350,
                           plot_bgcolor="white", paper_bgcolor="white")
    fig_yoy.update_yaxes(title_text="Ventas (S/.)", secondary_y=False)
    fig_yoy.update_yaxes(title_text="Variación YoY (%)", secondary_y=True)
    st.plotly_chart(fig_yoy, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – PRONÓSTICO
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">🔮 Pronóstico de Ventas · Modelo: ' + model_type + '</p>', unsafe_allow_html=True)

    # División train / test
    corte_train = "2024-12-31"
    corte_test  = "2025-01-01"
    train_series = ts_mensual.loc[:corte_train, "Ventas"]
    test_series  = ts_mensual.loc[corte_test:,  "Ventas"]

    if len(train_series) < 24:
        st.warning("Se necesitan al menos 24 meses de datos históricos para entrenar el modelo.")
        st.stop()

    with st.spinner(f"Entrenando modelo {model_type}..."):
        try:
            fc_test, ci_test = run_forecast(train_series, model_type, len(test_series), confianza)
            fc_test.index = test_series.index[:len(fc_test)]
            ci_test.index = test_series.index[:len(ci_test)]
        except Exception as e:
            st.error(f"Error en el modelo: {e}")
            st.stop()

    kpis = calcular_kpis(test_series, fc_test)

    # KPI de evaluación
    st.markdown("**📐 Métricas de Evaluación (back-test 2025)**")
    kc = st.columns(5)
    icons = ["📏", "📐", "📊", "🎯", "⚖️"]
    colores = ["metric-card", "metric-card", "metric-card orange", "metric-card green", "metric-card"]
    for i, (k, v) in enumerate(kpis.items()):
        with kc[i]:
            st.markdown(f'<div class="{colores[i]}">{icons[i]} <b>{k}</b><br><span style="font-size:1.3rem">{v:,.2f}</span></div>', unsafe_allow_html=True)

    st.divider()

    # Gráfico back-test
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=train_series.index, y=train_series,
                                 mode="lines", name="Histórico (entrenamiento)",
                                 line=dict(color="#607D8B", width=1.5)))
    fig_bt.add_trace(go.Scatter(x=test_series.index, y=test_series,
                                 mode="lines+markers", name="Real 2025",
                                 line=dict(color="#34A853", width=2.5)))
    fig_bt.add_trace(go.Scatter(x=fc_test.index, y=fc_test,
                                 mode="lines+markers", name=f"Pronóstico ({model_type})",
                                 line=dict(color="#1A73E8", width=2, dash="dash")))
    if ci_test is not None:
        fig_bt.add_trace(go.Scatter(
            x=ci_test.index.tolist() + ci_test.index[::-1].tolist(),
            y=ci_test["Limite_Superior"].tolist() + ci_test["Limite_Inferior"][::-1].tolist(),
            fill="toself", fillcolor="rgba(26,115,232,0.12)",
            line=dict(color="rgba(255,255,255,0)"), name=f"IC {confianza}%",
        ))
    fig_bt.update_layout(
        title=f"Back-test: Pronóstico vs Ventas Reales · {model_type}",
        xaxis_title="Fecha", yaxis_title="Ventas (S/.)",
        plot_bgcolor="white", paper_bgcolor="white", height=420,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Lectura del back-test:</b> Un MAPE < 10% se considera excelente en retail farmacéutico. El sesgo positivo indica sobreestimación sistemática; el negativo, subestimación. Compare modelos cambiando la selección en la barra lateral.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Pronóstico futuro ──
    st.markdown(f'<p class="section-title">📅 Pronóstico Futuro · {horizonte_label}</p>', unsafe_allow_html=True)

    full_series = ts_mensual["Ventas"]
    last_date   = full_series.index[-1]
    future_idx  = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizonte, freq="MS")

    with st.spinner("Generando pronóstico futuro..."):
        try:
            fc_fut, ci_fut = run_forecast(full_series, model_type, horizonte, confianza)
            fc_fut.index = future_idx
            ci_fut.index = future_idx
        except Exception as e:
            st.error(f"Error generando pronóstico: {e}")
            st.stop()

    fc_fut = fc_fut.clip(lower=0)
    ci_fut["Limite_Inferior"] = ci_fut["Limite_Inferior"].clip(lower=0)

    fig_fc = go.Figure()
    # Histórico último año
    hist_reciente = full_series.iloc[-24:]
    fig_fc.add_trace(go.Scatter(x=hist_reciente.index, y=hist_reciente,
                                 mode="lines", name="Histórico",
                                 line=dict(color="#607D8B", width=2)))
    fig_fc.add_trace(go.Scatter(x=fc_fut.index, y=fc_fut,
                                 mode="lines+markers", name=f"Pronóstico ({model_type})",
                                 line=dict(color="#1A73E8", width=2.5)))
    fig_fc.add_trace(go.Scatter(
        x=ci_fut.index.tolist() + ci_fut.index[::-1].tolist(),
        y=ci_fut["Limite_Superior"].tolist() + ci_fut["Limite_Inferior"][::-1].tolist(),
        fill="toself", fillcolor="rgba(26,115,232,0.12)",
        line=dict(color="rgba(255,255,255,0)"), name=f"IC {confianza}%",
    ))
    # Línea vertical de corte (shapes es más compatible que add_vline con fechas)
    fig_fc.add_shape(
        type="line",
        x0=last_date, x1=last_date,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#EA4335", dash="dot", width=1.5),
    )
    fig_fc.add_annotation(
        x=last_date, y=1, xref="x", yref="paper",
        text="Último dato real", showarrow=False,
        font=dict(color="#EA4335", size=10),
        xanchor="left", yanchor="bottom",
    )
    fig_fc.update_layout(
        title=f"Pronóstico de Ventas BotiCura · {model_type} ({horizonte} meses)",
        xaxis_title="Fecha", yaxis_title="Ventas (S/.)",
        plot_bgcolor="white", paper_bgcolor="white", height=450,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Tabla de resultados
    resultado_tbl = pd.DataFrame({
        "Fecha": fc_fut.index.strftime("%b %Y"),
        "Ventas Pronosticadas (S/.)": fc_fut.values.round(2),
        "Límite Inferior (S/.)": ci_fut["Limite_Inferior"].values.round(2),
        "Límite Superior (S/.)": ci_fut["Limite_Superior"].values.round(2),
        "Unidades Estimadas": (fc_fut.values / ticket).round().astype(int),
    })
    resultado_tbl["Variación vs Mes Anterior (%)"] = resultado_tbl["Ventas Pronosticadas (S/.)"].pct_change().mul(100).round(1)

    with st.expander("📋 Ver tabla de pronóstico completa", expanded=True):
        st.dataframe(resultado_tbl.style.format({
            "Ventas Pronosticadas (S/.)": "{:,.2f}",
            "Límite Inferior (S/.)": "{:,.2f}",
            "Límite Superior (S/.)": "{:,.2f}",
            "Unidades Estimadas": "{:,}",
        }).background_gradient(subset=["Ventas Pronosticadas (S/.)"], cmap="Blues"), use_container_width=True)

    # Gráfico dual: Ventas S/. + Unidades
    st.markdown("**📦 Pronóstico de Unidades Estimadas**")
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dual.add_trace(go.Bar(
        x=fc_fut.index, y=fc_fut.values,
        name="Ventas (S/.)", marker_color="#1A73E8", opacity=0.75,
    ), secondary_y=False)
    unidades_fc_fut = (fc_fut.values / ticket).round().astype(int)
    fig_dual.add_trace(go.Scatter(
        x=fc_fut.index, y=unidades_fc_fut,
        name="Unidades Estimadas", mode="lines+markers",
        line=dict(color="#EA4335", width=2.5), marker=dict(size=7),
    ), secondary_y=True)
    fig_dual.update_layout(
        title="Ventas (S/.) y Unidades Estimadas por Mes",
        plot_bgcolor="white", paper_bgcolor="white", height=380,
        legend=dict(orientation="h", y=-0.2),
    )
    fig_dual.update_yaxes(title_text="Ventas (S/.)", secondary_y=False)
    fig_dual.update_yaxes(title_text="Unidades", secondary_y=True)
    st.plotly_chart(fig_dual, use_container_width=True)
    st.caption(f"🔑 Ticket promedio usado: S/ {ticket:.2f} por unidad (categoría: {sel_categoria})")

    # Resumen anual
    st.markdown("**📅 Resumen Anual del Pronóstico**")
    resultado_tbl["Año"] = pd.to_datetime(fc_fut.index).year
    anual_fc = resultado_tbl.groupby("Año").agg(
        Ventas_Anuales=("Ventas Pronosticadas (S/.)", "sum"),
        Unidades_Anuales=("Unidades Estimadas", "sum"),
    ).reset_index()
    anual_fc["Ventas_Anuales"] = anual_fc["Ventas_Anuales"].map("S/ {:,.2f}".format)
    anual_fc["Unidades_Anuales"] = anual_fc["Unidades_Anuales"].map("{:,}".format)
    st.dataframe(anual_fc, use_container_width=True)

    # Descarga
    excel_fc = generar_excel({
        "Pronostico_Mensual": resultado_tbl.drop(columns=["Año"]),
        "KPIs_Backtest": pd.DataFrame(list(kpis.items()), columns=["KPI", "Valor"]),
    })
    st.download_button(
        "⬇️ Descargar Pronóstico Excel",
        data=excel_fc,
        file_name=f"pronostico_boticura_{model_type}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ══════════════════════════════════════════════
# TAB 3 – ANÁLISIS POR CATEGORÍA  (datos reales)
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">📦 Análisis de Ventas por Categoría · Datos Reales del Dataset</p>', unsafe_allow_html=True)

    has_cat_col = "Categoria" in df.columns

    if not has_cat_col:
        st.warning("⚠️ El dataset no contiene la columna **Categoria**. Sube **ventas_mensuales_boticura_v3.xlsx** para ver este análisis con datos reales por categoría y producto.")
        st.stop()

    # ── Filtrar igual que el resto de la app ──────────────────────
    df_cat_filt = data_raw.copy()
    if sel_depto != "Todos": df_cat_filt = df_cat_filt[df_cat_filt["Departamento"] == sel_depto]
    if sel_dist  != "Todos": df_cat_filt = df_cat_filt[df_cat_filt["Distrito"]     == sel_dist]
    if sel_canal != "Todos": df_cat_filt = df_cat_filt[df_cat_filt["Canal_Venta"]  == sel_canal]

    # Excluir filas de totales de categoría si existieran
    if "Producto" in df_cat_filt.columns:
        df_cat_filt = df_cat_filt[df_cat_filt["Producto"] != "TOTAL_CATEGORIA"]

    # Márgenes reales calculados del dataset de transacciones
    # (precio - costo_unitario) / precio × 100
    margen_cat_real = {
        "Medicamentos":   25,   # Ibuprofeno ~25%, Paracetamol ~25%
        "Vitaminas":      28,   # Vitamina C ~25%, Omega3 ~25%, Magnesio ~25%
        "Dermocosmética": 25,   # Crema Facial ~26%, Bloqueador ~26%, Micelar ~25%
        "Bebés":          26,   # Pañales ~25%, Leche ~25%, Toallitas ~25%
        "Nutrición":      25,   # Colágeno ~25%
        "Veterinaria":    26,   # Antipulgas ~26%, Shampoo ~25%
    }

    agg_cat = {"Ventas_Acumuladas": ("Ventas", "sum")}
    if "Unidades" in df_cat_filt.columns:
        agg_cat["Unidades_Totales"] = ("Unidades", "sum")
    else:
        agg_cat["Unidades_Totales"] = ("Ventas", "count")
    if "Precio_Unitario" in df_cat_filt.columns:
        agg_cat["Ticket_Promedio"] = ("Precio_Unitario", "mean")
    else:
        agg_cat["Ticket_Promedio"] = ("Ventas", "mean")

    resumen_cat_hist = (
        df_cat_filt.groupby("Categoria")
        .agg(**agg_cat)
        .reset_index()
    )
    total_ventas_cat = resumen_cat_hist["Ventas_Acumuladas"].sum()
    resumen_cat_hist["Participacion_%"] = (resumen_cat_hist["Ventas_Acumuladas"] / total_ventas_cat * 100).round(1)
    resumen_cat_hist["Margen_%"] = resumen_cat_hist["Categoria"].map(margen_cat_real)
    resumen_cat_hist["Margen_Generado_S/."] = (resumen_cat_hist["Ventas_Acumuladas"] * resumen_cat_hist["Margen_%"] / 100).round(2)
    resumen_cat_hist = resumen_cat_hist.sort_values("Ventas_Acumuladas", ascending=False)

    # KPI cards por categoría
    n_cats = len(resumen_cat_hist)
    cols_k = st.columns(min(n_cats, 6))
    for i, row in resumen_cat_hist.iterrows():
        col_idx = list(resumen_cat_hist.index).index(i) % len(cols_k)
        with cols_k[col_idx]:
            st.markdown(
                f'<div class="metric-card">'
                f'<b style="color:{COLOR_CATEGORIA.get(row["Categoria"],"#607D8B")}">{row["Categoria"]}</b><br>'
                f'S/ {row["Ventas_Acumuladas"]:,.0f}<br>'
                f'<small>{row["Participacion_%"]}% · Margen {row["Margen_%"]}%</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Gráficos históricos ───────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig_cb = px.bar(
            resumen_cat_hist, x="Categoria", y="Ventas_Acumuladas",
            color="Categoria", color_discrete_map=COLOR_CATEGORIA,
            title="Ventas Acumuladas por Categoría (S/.)", text_auto=".2s",
        )
        fig_cb.update_layout(showlegend=False, height=370, plot_bgcolor="white")
        st.plotly_chart(fig_cb, use_container_width=True)

    with c2:
        fig_cp = px.pie(
            resumen_cat_hist, names="Categoria", values="Participacion_%",
            color="Categoria", color_discrete_map=COLOR_CATEGORIA,
            title="Mix de Ventas por Categoría (%)", hole=0.38,
        )
        fig_cp.update_layout(height=370)
        st.plotly_chart(fig_cp, use_container_width=True)

    # Tabla resumen
    st.dataframe(
        resumen_cat_hist.rename(columns={
            "Categoria":"Categoría","Ventas_Acumuladas":"Ventas Acum. (S/.)","Unidades_Totales":"Unidades",
            "Ticket_Promedio":"Ticket Prom. (S/.)","Participacion_%":"Part. (%)","Margen_%":"Margen (%)",
            "Margen_Generado_S/.":"Margen Generado (S/.)",
        }).style.format({
            "Ventas Acum. (S/.)":"{:,.0f}", "Unidades":"{:,.0f}",
            "Ticket Prom. (S/.)":"{:.2f}", "Margen Generado (S/.)":"{:,.0f}",
        }).background_gradient(subset=["Ventas Acum. (S/.)"], cmap="Blues"),
        use_container_width=True,
    )

    # ── Evolución mensual por categoría ──────────────────────────
    st.markdown('<p class="section-title">📈 Evolución Mensual por Categoría (datos históricos)</p>', unsafe_allow_html=True)

    ts_por_cat = (
        df_cat_filt.groupby(["Fecha", "Categoria"])["Ventas"]
        .sum().reset_index()
    )
    fig_evol = px.line(
        ts_por_cat, x="Fecha", y="Ventas", color="Categoria",
        color_discrete_map=COLOR_CATEGORIA,
        title="Ventas Mensuales por Categoría (S/.)",
        labels={"Ventas":"Ventas (S/.)", "Fecha":""},
    )
    fig_evol.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
                            legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_evol, use_container_width=True)

    # ── Pronóstico por categoría con datos reales ─────────────────
    st.markdown(f'<p class="section-title">🔮 Pronóstico por Categoría · {horizonte} meses · {model_type}</p>', unsafe_allow_html=True)

    future_idx_cat = pd.date_range(
        start=ts_mensual.index[-1] + pd.DateOffset(months=1),
        periods=horizonte, freq="MS",
    )

    fig_fc_cat = go.Figure()
    resumen_fc_cat = []

    for cat in resumen_cat_hist["Categoria"].tolist():
        serie_c = (
            df_cat_filt[df_cat_filt["Categoria"] == cat]
            .groupby("Fecha")["Ventas"].sum()
            .resample("MS").sum()
            .asfreq("MS", fill_value=0)
        )
        serie_c.name = "Ventas"
        serie_c.index.name = "Fecha"
        if len(serie_c) < 24:
            continue
        try:
            fc_c, ci_c = run_forecast(serie_c, model_type, horizonte, confianza)
            fc_c.index = future_idx_cat
            ci_c.index = future_idx_cat
            fc_c = fc_c.clip(lower=0)
            ci_c["Limite_Inferior"] = ci_c["Limite_Inferior"].clip(lower=0)

            ticket_c = TICKET_PROMEDIO_CATEGORIA.get(cat, 28.5)
            margen_c = margen_cat_real.get(cat, 35)

            fig_fc_cat.add_trace(go.Scatter(
                x=fc_c.index, y=fc_c, name=cat, mode="lines",
                line=dict(color=COLOR_CATEGORIA.get(cat, "#607D8B"), width=2.5),
            ))
            fig_fc_cat.add_trace(go.Scatter(
                x=ci_c.index.tolist() + ci_c.index[::-1].tolist(),
                y=ci_c["Limite_Superior"].tolist() + ci_c["Limite_Inferior"][::-1].tolist(),
                fill="toself", showlegend=False,
                fillcolor=f"rgba(0,0,0,0.04)",
                line=dict(color="rgba(255,255,255,0)"),
            ))

            resumen_fc_cat.append({
                "Categoría":                    cat,
                "Ventas Pron. Total (S/.)":      round(fc_c.sum(), 2),
                "Ventas Pron. Mensual Prom.":   round(fc_c.mean(), 2),
                "Unidades Estimadas":            int(fc_c.sum() / ticket_c),
                "Margen Estimado (S/.)":         round(fc_c.sum() * margen_c / 100, 2),
                "Mes Pico":                     fc_c.idxmax().strftime("%b %Y"),
            })
        except Exception:
            pass

    fig_fc_cat.update_layout(
        title=f"Pronóstico por Categoría · {model_type} ({horizonte} meses)",
        xaxis_title="Fecha", yaxis_title="Ventas (S/.)",
        plot_bgcolor="white", paper_bgcolor="white", height=430,
        legend=dict(orientation="h", y=-0.22),
    )
    st.plotly_chart(fig_fc_cat, use_container_width=True)

    if resumen_fc_cat:
        df_fc_cat = pd.DataFrame(resumen_fc_cat).sort_values("Ventas Pron. Total (S/.)", ascending=False)
        st.dataframe(
            df_fc_cat.style.format({
                "Ventas Pron. Total (S/.)":   "{:,.0f}",
                "Ventas Pron. Mensual Prom.": "{:,.0f}",
                "Unidades Estimadas":         "{:,}",
                "Margen Estimado (S/.)":      "{:,.0f}",
            }).background_gradient(subset=["Ventas Pron. Total (S/.)"], cmap="Greens"),
            use_container_width=True,
        )

        # Gráfico margen vs ventas
        fig_mv = px.scatter(
            df_fc_cat, x="Ventas Pron. Total (S/.)", y="Margen Estimado (S/.)",
            size="Unidades Estimadas", color="Categoría",
            color_discrete_map=COLOR_CATEGORIA,
            text="Categoría",
            title="Relación Ventas vs Margen Estimado por Categoría (tamaño = Unidades)",
        )
        fig_mv.update_traces(textposition="top center")
        fig_mv.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig_mv, use_container_width=True)

        excel_cat_out = generar_excel({
            "Pronostico_Categorias": df_fc_cat,
            "Historico_Categorias":  resumen_cat_hist,
        })
        st.download_button(
            "⬇️ Descargar Análisis por Categoría",
            data=excel_cat_out,
            file_name="analisis_categorias_boticura.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Medicamentos tiene el mayor volumen pero Cosmética y Suplementos generan mayor margen por sol vendido. Un <b>rebalanceo del mix</b> hacia categorías de mayor margen puede incrementar la rentabilidad de BotiCura sin necesidad de crecer en ventas totales.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 – TOP PRODUCTOS  (datos reales)
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">🏆 Pronóstico por Producto · Datos Reales del Dataset</p>', unsafe_allow_html=True)

    has_prod_col = "Producto" in df.columns

    if not has_prod_col:
        st.warning("⚠️ El dataset no contiene la columna **Producto**. Sube **ventas_mensuales_boticura_v3.xlsx** para ver pronósticos reales por producto.")
        st.stop()

    # ── Filtros de esta tab ───────────────────────────────────────
    df_prod_filt = data_raw.copy()
    if sel_depto != "Todos": df_prod_filt = df_prod_filt[df_prod_filt["Departamento"] == sel_depto]
    if sel_dist  != "Todos": df_prod_filt = df_prod_filt[df_prod_filt["Distrito"]     == sel_dist]
    if sel_canal != "Todos": df_prod_filt = df_prod_filt[df_prod_filt["Canal_Venta"]  == sel_canal]
    df_prod_filt = df_prod_filt[df_prod_filt["Producto"] != "TOTAL_CATEGORIA"]

    cats_disponibles = sorted(df_prod_filt["Categoria"].dropna().unique().tolist())
    sel_cat_tab4 = st.selectbox("📦 Seleccionar Categoría:", cats_disponibles, key="cat_tab4")

    df_prod_cat = df_prod_filt[df_prod_filt["Categoria"] == sel_cat_tab4].copy()

    # ── Ranking histórico de productos ────────────────────────────
    agg_prod = {"Ventas_Acum": ("Ventas", "sum")}
    if "Unidades" in df_prod_cat.columns:
        agg_prod["Unidades"] = ("Unidades", "sum")
    else:
        agg_prod["Unidades"] = ("Ventas", "count")
    if "Precio_Unitario" in df_prod_cat.columns:
        agg_prod["Precio_Unitario"] = ("Precio_Unitario", "mean")
    else:
        agg_prod["Precio_Unitario"] = ("Ventas", "mean")

    ranking_hist = (
        df_prod_cat.groupby("Producto")
        .agg(**agg_prod)
        .reset_index()
        .sort_values("Ventas_Acum", ascending=False)
    )
    total_cat_ventas = ranking_hist["Ventas_Acum"].sum()
    ranking_hist["Part_%"] = (ranking_hist["Ventas_Acum"] / total_cat_ventas * 100).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig_rank_hist = px.bar(
            ranking_hist.head(10), x="Ventas_Acum", y="Producto",
            orientation="h", title=f"Top 10 Productos · {sel_cat_tab4} (Ventas Hist. S/.)",
            color="Ventas_Acum", color_continuous_scale="Blues", text_auto=".2s",
        )
        fig_rank_hist.update_layout(showlegend=False, height=400, plot_bgcolor="white", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_rank_hist, use_container_width=True)

    with c2:
        fig_rank_u = px.bar(
            ranking_hist.head(10), x="Unidades", y="Producto",
            orientation="h", title=f"Top 10 Productos · {sel_cat_tab4} (Unidades Vendidas)",
            color="Unidades", color_continuous_scale="Oranges", text_auto=",",
        )
        fig_rank_u.update_layout(showlegend=False, height=400, plot_bgcolor="white", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_rank_u, use_container_width=True)

    # ── Evolución mensual por producto ────────────────────────────
    st.markdown(f'<p class="section-title">📈 Evolución Histórica por Producto · {sel_cat_tab4}</p>', unsafe_allow_html=True)

    top_prods = ranking_hist["Producto"].head(6).tolist()

    ts_prod_hist = (
        df_prod_cat[df_prod_cat["Producto"].isin(top_prods)]
        .groupby(["Fecha", "Producto"])["Ventas"]
        .sum().reset_index()
    )
    fig_evol_prod = px.line(
        ts_prod_hist, x="Fecha", y="Ventas", color="Producto",
        title=f"Ventas Mensuales · Top {len(top_prods)} Productos de {sel_cat_tab4} (S/.)",
        labels={"Ventas":"Ventas (S/.)", "Fecha":""},
    )
    fig_evol_prod.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
                                 legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_evol_prod, use_container_width=True)

    # ── Pronóstico por producto ───────────────────────────────────
    st.markdown(f'<p class="section-title">🔮 Pronóstico por Producto · {sel_cat_tab4} · {horizonte} meses · {model_type}</p>', unsafe_allow_html=True)

    future_idx_prod = pd.date_range(
        start=ts_mensual.index[-1] + pd.DateOffset(months=1),
        periods=horizonte, freq="MS",
    )

    fig_fc_prod = go.Figure()
    tabla_fc_prod = []

    for prod in top_prods:
        serie_p = (
            df_prod_cat[df_prod_cat["Producto"] == prod]
            .groupby("Fecha")["Ventas"].sum()
            .resample("MS").sum()
            .asfreq("MS", fill_value=0)
        )
        serie_p.name = "Ventas"
        serie_p.index.name = "Fecha"
        if len(serie_p) < 24:
            continue
        try:
            fc_p, ci_p = run_forecast(serie_p, model_type, horizonte, confianza)
            fc_p.index = future_idx_prod
            ci_p.index = future_idx_prod
            fc_p = fc_p.clip(lower=0)
            ci_p["Limite_Inferior"] = ci_p["Limite_Inferior"].clip(lower=0)

            # Precio unitario real del producto
            precio_u = ranking_hist.loc[ranking_hist["Producto"] == prod, "Precio_Unitario"].values
            precio_u = float(precio_u[0]) if len(precio_u) > 0 and precio_u[0] > 0 else TICKET_PROMEDIO_CATEGORIA.get(sel_cat_tab4, 28.5)
            margen_p = margen_cat_real.get(sel_cat_tab4, 35)

            unidades_fc = (fc_p / precio_u).round().astype(int)

            fig_fc_prod.add_trace(go.Scatter(
                x=fc_p.index, y=fc_p, name=prod, mode="lines+markers",
                line=dict(width=2),
            ))
            # IC sombreado
            fig_fc_prod.add_trace(go.Scatter(
                x=ci_p.index.tolist() + ci_p.index[::-1].tolist(),
                y=ci_p["Limite_Superior"].tolist() + ci_p["Limite_Inferior"][::-1].tolist(),
                fill="toself", showlegend=False, fillcolor="rgba(0,0,0,0.04)",
                line=dict(color="rgba(255,255,255,0)"),
            ))

            tabla_fc_prod.append({
                "Producto":                    prod,
                "Precio Unit. (S/.)":          round(precio_u, 2),
                "Ventas Pron. Total (S/.)":    round(fc_p.sum(), 2),
                "Unidades Pron. Total":        int(unidades_fc.sum()),
                "Unidades Pron./Mes Prom.":    int(unidades_fc.mean()),
                "Margen Estimado (S/.)":       round(fc_p.sum() * margen_p / 100, 2),
                "Mes Pico":                    fc_p.idxmax().strftime("%b %Y"),
                "Part. Histórica (%)":         ranking_hist.loc[ranking_hist["Producto"]==prod,"Part_%"].values[0],
            })
        except Exception:
            pass

    fig_fc_prod.update_layout(
        title=f"Pronóstico de Ventas por Producto · {sel_cat_tab4} · {model_type}",
        xaxis_title="Fecha", yaxis_title="Ventas (S/.)",
        plot_bgcolor="white", paper_bgcolor="white", height=440,
        legend=dict(orientation="h", y=-0.28, font=dict(size=10)),
    )
    st.plotly_chart(fig_fc_prod, use_container_width=True)

    if tabla_fc_prod:
        df_fc_prod = pd.DataFrame(tabla_fc_prod).sort_values("Ventas Pron. Total (S/.)", ascending=False)

        st.markdown("**📋 Tabla de Pronóstico por Producto · Plan de Compras**")
        st.dataframe(
            df_fc_prod.style.format({
                "Precio Unit. (S/.)":         "{:.2f}",
                "Ventas Pron. Total (S/.)":   "{:,.0f}",
                "Unidades Pron. Total":       "{:,}",
                "Unidades Pron./Mes Prom.":   "{:,}",
                "Margen Estimado (S/.)":      "{:,.0f}",
                "Part. Histórica (%)":        "{:.1f}",
            }).background_gradient(subset=["Ventas Pron. Total (S/.)"], cmap="Oranges"),
            use_container_width=True,
        )

        # Gráfico unidades vs ventas (para plan de compras)
        fig_uv = make_subplots(specs=[[{"secondary_y": True}]])
        fig_uv.add_trace(go.Bar(
            x=df_fc_prod["Producto"], y=df_fc_prod["Ventas Pron. Total (S/.)"],
            name="Ventas (S/.)", marker_color="#1A73E8",
        ), secondary_y=False)
        fig_uv.add_trace(go.Scatter(
            x=df_fc_prod["Producto"], y=df_fc_prod["Unidades Pron. Total"],
            name="Unidades", mode="lines+markers",
            line=dict(color="#EA4335", width=2), marker=dict(size=8),
        ), secondary_y=True)
        fig_uv.update_layout(
            title=f"Ventas vs Unidades Proyectadas · {sel_cat_tab4}",
            plot_bgcolor="white", paper_bgcolor="white", height=380,
            legend=dict(orientation="h", y=-0.2),
        )
        fig_uv.update_yaxes(title_text="Ventas (S/.)", secondary_y=False)
        fig_uv.update_yaxes(title_text="Unidades", secondary_y=True)
        st.plotly_chart(fig_uv, use_container_width=True)

        # Plan de compras mensual
        st.markdown("**🛒 Plan de Compras Mensual Estimado (unidades)**")
        plan_rows = []
        for prod in top_prods:
            serie_p = (
                df_prod_cat[df_prod_cat["Producto"] == prod]
                .groupby("Fecha")["Ventas"].sum()
                .resample("MS").sum()
                .asfreq("MS", fill_value=0)
            )
            serie_p.name = "Ventas"
            serie_p.index.name = "Fecha"
            if len(serie_p) < 24:
                continue
            try:
                fc_p2, _ = run_forecast(serie_p, model_type, horizonte, confianza)
                fc_p2.index = future_idx_prod
                fc_p2 = fc_p2.clip(lower=0)
                precio_u = ranking_hist.loc[ranking_hist["Producto"] == prod, "Precio_Unitario"].values
                precio_u = float(precio_u[0]) if len(precio_u) > 0 and precio_u[0] > 0 else 28.5
                unidades_mes = (fc_p2 / precio_u).round().astype(int)
                row_plan = {"Producto": prod}
                for fecha, uds in zip(unidades_mes.index, unidades_mes.values):
                    row_plan[fecha.strftime("%b %Y")] = int(uds)
                plan_rows.append(row_plan)
            except Exception:
                pass

        if plan_rows:
            df_plan = pd.DataFrame(plan_rows).set_index("Producto")
            st.dataframe(
                df_plan.style.background_gradient(cmap="YlOrRd", axis=1),
                use_container_width=True,
            )

        excel_prod_out = generar_excel({
            "Pronostico_Productos": df_fc_prod,
            "Plan_Compras_Mensual": pd.DataFrame(plan_rows) if plan_rows else pd.DataFrame(),
            "Ranking_Historico":    ranking_hist,
        })
        st.download_button(
            "⬇️ Descargar Plan de Compras + Pronóstico",
            data=excel_prod_out,
            file_name=f"plan_compras_{sel_cat_tab4.lower().replace(' ','_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown(
        '<div class="insight-box">💡 <b>Insight · Plan de Compras:</b> Las <b>Unidades Pron./Mes Prom.</b> son el input directo para el cálculo del <b>EOQ (cantidad óptima de pedido)</b>. '
        'Multiplica unidades mensuales × Lead Time del proveedor para estimar el <b>stock mínimo</b> requerido. '
        'Para productos críticos como Paracetamol o Pañales, agrega un <b>stock de seguridad</b> de 15–20% sobre la demanda pronosticada.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════
# TAB 5 – GUÍA MBA
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">📚 Guía Metodológica para Estudiantes MBA</p>', unsafe_allow_html=True)

    with st.expander("📐 ¿Cómo interpretar las métricas de error?", expanded=True):
        st.markdown("""
| Métrica | Fórmula | Interpretación | Referencia Farmacia |
|---|---|---|---|
| **MAE** | Media errores absolutos | Cuánto se equivoca en promedio en S/. | < S/ 5,000 mensual = bueno |
| **RMSE** | Raíz del error cuadrático medio | Penaliza errores grandes | < 1.5× MAE = estable |
| **MAPE (%)** | Error porcentual absoluto | Compara entre locales de distinto tamaño | < 10% excelente, < 15% aceptable |
| **Precisión Dirección** | % meses con tendencia correcta | ¿Sabe cuándo sube o baja? | > 65% = útil para decisiones |
| **Sesgo** | Promedio (pronóstico − real) | Positivo = sobreestima, negativo = subestima | Ideal: cercano a 0 |
        """)

    with st.expander("🔁 ¿Cuándo usar cada modelo?"):
        st.markdown("""
| Modelo | Fortaleza | Limitación | Recomendado cuando... |
|---|---|---|---|
| **Holt-Winters** | Simple, captura tendencia + estacionalidad | Sensible a outliers | Serie con ciclo anual claro (Navidad, Fiestas Patrias) |
| **ARIMA** | Flexible para series no estacionarias | No captura estacionalidad por sí solo | Serie con tendencia pero sin ciclo marcado |
| **SARIMA** | ARIMA con componente estacional | Lento, muchos parámetros | Serie con estacionalidad fuerte (ej. medicamentos antigripales) |
| **Prophet** | Maneja festivos y cambios de régimen | Requiere instalación adicional | Planificación estratégica multi-año con eventos conocidos |
        """)

    with st.expander("📦 Implicancias para la Gestión de Inventario"):
        st.markdown("""
El pronóstico de ventas es el **input principal** para:

1. **Stock de seguridad** = Z × σ_demanda × √Lead_time  
   — donde Z depende del nivel de servicio deseado (98% para medicamentos críticos)

2. **Punto de reorden** = Demanda_diaria_promedio × Lead_time + Stock_seguridad

3. **Presupuesto de compras** = Σ (Unidades_pronosticadas × Costo_unitario)

4. **Capital de trabajo en inventario** = Costo_inventario_promedio / Rotación_inventario

Para BotiCura con 150 locales, un error de pronóstico del 15% en medicamentos de alta rotación puede significar **desabastecimiento o sobrestock** de S/ 2–4 millones mensuales a nivel nacional.
        """)

    with st.expander("🏢 Aplicación Estratégica para BotiCura"):
        st.markdown("""
**Decisiones que este simulador apoya:**

- 📍 **Expansión geográfica**: ¿Qué departamentos tienen mayor potencial de crecimiento?
- 🛒 **Mix de producto**: ¿Aumentar el mix de Cosmética/Suplementos para mejorar margen?
- 📱 **Canal Online**: El canal digital crece ~3% anual más que el físico — ¿cuándo invertir más en e-commerce?
- 🗓️ **Campañas estacionales**: Planificar inventario para Día de la Madre (mayo), Fiestas Patrias (julio) y Navidad (diciembre)
- 💰 **Presupuesto 2026–2027**: Las proyecciones por categoría y producto alimentan el **Plan Operativo Anual (POA)**
        """)

    with st.expander("🔍 Comparación de Modelos – ¿Cómo elegir el mejor?"):
        st.markdown("""
**Metodología recomendada para MBA:**

1. Entrenar todos los modelos con datos 2020–2024
2. Evaluar con back-test sobre 2025 (datos reales disponibles)
3. Comparar MAPE y Precisión de Dirección
4. Seleccionar el modelo con menor MAPE **y** sesgo cercano a cero
5. Usar el modelo ganador para el pronóstico 2026–2027
6. Documentar supuestos y limitaciones en el informe ejecutivo

> **Regla práctica**: Para productos con estacionalidad fuerte (pañales, protector solar) use SARIMA o Holt-Winters. Para líneas nuevas con pocos datos, prefiera Holt-Winters simple.
        """)

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; font-size:12px; color:#888; margin-top:10px;'>
    💊 <b>BotiCura S.A.C.</b> · Simulador de Pronóstico de Ventas v2.0 |
    © 2025 Diseñado por <b>Wilton Torvisco</b> · Todos los derechos reservados.
</div>
""", unsafe_allow_html=True)
