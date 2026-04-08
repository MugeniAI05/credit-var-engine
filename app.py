import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from data.fetch_data import fetch_credit_spreads
from models.portfolio import build_portfolio
from models.historical_var import HistoricalVaR
from models.monte_carlo_var import MonteCarloVaR
from models.backtesting import run_backtest

os.environ["FRED_API_KEY"] = st.secrets.get("FRED_API_KEY", "")

st.set_page_config(
    page_title="Credit VaR Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252836);
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .metric-label { color: #8b8fa8; font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase; }
    .metric-value { color: #e8eaf6; font-size: 1.8rem; font-weight: 700; margin: 0.2rem 0; }
    .metric-delta { font-size: 0.8rem; }
    .pos { color: #ef5350; }
    .neg { color: #26a69a; }
    h1 { color: #e8eaf6 !important; opacity: 1 !important; }
    .stSelectbox label, .stSlider label, .stMultiSelect label, .stNumberInput label { color: #8b8fa8 !important; }
    div[data-testid="stSidebarContent"] { background-color: #161b2e; }
    .main .stMarkdown strong { color: #e8eaf6 !important; }
    div[data-testid="stSidebarContent"] h2 { color: #e8eaf6 !important; opacity: 1 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Model Configuration")
    st.markdown("---")

    confidence_level = st.selectbox(
        "Confidence Level",
        options=[0.95, 0.99, 0.999],
        index=1,
        format_func=lambda x: f"{x*100:.1f}%"
    )

    holding_period = st.selectbox(
        "Holding Period (days)",
        options=[1, 10],
        index=1
    )

    lookback_window = st.slider(
        "Historical Lookback (days)",
        min_value=250,
        max_value=1000,
        value=500,
        step=50
    )

    n_simulations = st.selectbox(
        "MC Simulations",
        options=[5_000, 10_000, 50_000],
        index=1,
        format_func=lambda x: f"{x:,}"
    )

    st.markdown("---")
    st.markdown("## Portfolio Weights")

    w_ig_bonds = st.slider("IG Corporate Bonds", 0, 100, 35)
    w_hy_bonds = st.slider("HY Corporate Bonds", 0, 100, 20)
    w_ig_cds   = st.slider("IG CDS Protection", 0, 100, 25)
    w_hy_cds   = st.slider("HY CDS Protection", 0, 100, 20)

    total_w = w_ig_bonds + w_hy_bonds + w_ig_cds + w_hy_cds
    if total_w != 100:
        st.warning(f"Weights sum to {total_w}% (should be 100%)")

    portfolio_value = st.number_input(
        "Portfolio Value ($M)",
        min_value=1,
        max_value=10_000,
        value=100,
        step=10
    )

    run_btn = st.button("▶  Run VaR Engine", type="primary", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# Credit VaR Engine")
st.markdown("*Market Risk Analytics | Bonds & CDS Portfolio*")
st.markdown("---")

if not run_btn:
    st.info("  Configure your portfolio in the sidebar and click **Run VaR Engine** to begin.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What This Engine Calculates")
        st.markdown("""
- **Historical Simulation VaR** – uses 2 years of real credit spread data from FRED  
- **Monte Carlo VaR** – correlated spread shocks via Cholesky decomposition  
- **Stressed VaR** – recalibrated on the 2008 financial crisis window  
- **Backtesting** – Basel traffic-light test with exceedance tracking  
- **IRC Proxy** – jump-to-default charge estimate  
        """)
    with col2:
        st.markdown("### Credit Instruments Modelled")
        st.markdown("""
- **IG Corporate Bonds** – driven by ICE BofA IG OAS spread (FRED: BAMLC0A0CM)  
- **HY Corporate Bonds** – driven by ICE BofA HY OAS spread (FRED: BAMLH0A0HYM2)  
- **IG CDS** – proxied from IG OAS with basis adjustment  
- **HY CDS** – proxied from HY OAS with basis adjustment  
        """)
    st.stop()

# ── Run computation ───────────────────────────────────────────────────────────
with st.spinner("Fetching credit spread data from FRED…"):
    raw_spreads = fetch_credit_spreads(lookback_days=lookback_window + 260)

weights = np.array([w_ig_bonds, w_hy_bonds, w_ig_cds, w_hy_cds]) / 100.0
portfolio = build_portfolio(raw_spreads, weights, portfolio_value * 1_000_000)

with st.spinner("Running Historical Simulation…"):
    hist_var = HistoricalVaR(portfolio, confidence=confidence_level, hp=holding_period)
    h_var_val, h_pnl = hist_var.compute()

with st.spinner("Running Monte Carlo Simulation…"):
    mc_var = MonteCarloVaR(portfolio, confidence=confidence_level, hp=holding_period, n_sims=n_simulations)
    mc_var_val, mc_pnl = mc_var.compute()

with st.spinner("Running Backtesting…"):
    bt_results = run_backtest(portfolio, confidence=confidence_level, hp=holding_period)

stressed_var = hist_var.stressed_var()
irc_proxy    = hist_var.irc_proxy()

# ── KPI row ───────────────────────────────────────────────────────────────────
kpis = [
    ("Historical VaR", f"${h_var_val/1e6:.2f}M",  f"{h_var_val/portfolio.notional*100:.2f}% of NAV", "pos"),
    ("Monte Carlo VaR", f"${mc_var_val/1e6:.2f}M", f"{mc_var_val/portfolio.notional*100:.2f}% of NAV", "pos"),
    ("Stressed VaR",   f"${stressed_var/1e6:.2f}M", f"{stressed_var/portfolio.notional*100:.2f}% of NAV", "pos"),
    ("IRC Proxy",      f"${irc_proxy/1e6:.2f}M",   "Jump-to-Default charge", "pos"),
    ("Exceedances",    str(bt_results['exceedances']), f"/ {bt_results['total_obs']} obs ({bt_results['exc_pct']:.1f}%)",
     "neg" if bt_results['exc_pct'] < (1 - confidence_level) * 1.5 * 100 else "pos"),
    ("Basel Zone",     bt_results['zone'],           bt_results['zone_desc'], "neg" if bt_results['zone'] == "Green" else "pos"),
]

cols = st.columns(6)
for col, (label, val, delta, cls) in zip(cols, kpis):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val}</div>
        <div class="metric-delta {cls}">{delta}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Charts row 1 ─────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("#### Historical P&L Distribution with VaR")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=h_pnl / 1e6, nbinsx=80,
        marker_color="#5c6bc0", opacity=0.8, name="P&L"
    ))
    fig.add_vline(x=-h_var_val / 1e6, line_color="#ef5350", line_width=2,
                  annotation_text=f"HistVaR {confidence_level*100:.0f}%",
                  annotation_position="top right")
    fig.add_vline(x=-mc_var_val / 1e6, line_color="#ffa726", line_width=2, line_dash="dash",
                  annotation_text=f"MC VaR",
                  annotation_position="top left")
    fig.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="P&L ($M)", yaxis_title="Frequency",
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown("#### Monte Carlo P&L Distribution")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=mc_pnl / 1e6, nbinsx=100,
        marker_color="#26a69a", opacity=0.8, name="MC P&L"
    ))
    fig2.add_vline(x=-mc_var_val / 1e6, line_color="#ffa726", line_width=2,
                   annotation_text=f"MC VaR {confidence_level*100:.0f}%",
                   annotation_position="top right")
    fig2.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="P&L ($M)", yaxis_title="Frequency",
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Charts row 2 ─────────────────────────────────────────────────────────────
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.markdown("#### Credit Spread History")
    spread_df = portfolio.spread_df
    fig3 = go.Figure()
    colors = ["#5c6bc0", "#ef5350", "#26a69a", "#ffa726"]
    for i, col_name in enumerate(spread_df.columns):
        fig3.add_trace(go.Scatter(
            x=spread_df.index, y=spread_df[col_name],
            name=col_name, line=dict(color=colors[i], width=1.5)
        ))
    fig3.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="OAS Spread (bps)",
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_r2:
    st.markdown("#### Backtesting: VaR Exceedances")
    bt_df = bt_results['detail_df']
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df['pnl'] / 1e6,
        name="Daily P&L", line=dict(color="#5c6bc0", width=1), opacity=0.7
    ))
    fig4.add_trace(go.Scatter(
        x=bt_df.index, y=-bt_df['var'] / 1e6,
        name="−VaR Threshold", line=dict(color="#ef5350", width=1.5, dash="dot")
    ))
    exc = bt_df[bt_df['exceedance']]
    if len(exc):
        fig4.add_trace(go.Scatter(
            x=exc.index, y=exc['pnl'] / 1e6,
            mode="markers", name="Exceedance",
            marker=dict(color="#ffa726", size=8, symbol="x")
        ))
    fig4.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="P&L ($M)",
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── VaR term structure ────────────────────────────────────────────────────────
st.markdown("#### VaR Term Structure & Confidence Level Sensitivity")
col_ts, col_cs = st.columns(2)

with col_ts:
    hps = [1, 2, 5, 10, 20]
    hist_ts, mc_ts = [], []
    for hp in hps:
        v, _ = HistoricalVaR(portfolio, confidence=confidence_level, hp=hp).compute()
        m, _ = MonteCarloVaR(portfolio, confidence=confidence_level, hp=hp, n_sims=2000).compute()
        hist_ts.append(v / 1e6)
        mc_ts.append(m / 1e6)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=hps, y=hist_ts, name="Historical VaR",
                              line=dict(color="#5c6bc0", width=2), mode="lines+markers"))
    fig5.add_trace(go.Scatter(x=hps, y=mc_ts, name="Monte Carlo VaR",
                              line=dict(color="#26a69a", width=2), mode="lines+markers"))
    fig5.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title="Holding Period (days)", yaxis_title="VaR ($M)",
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130"
    )
    st.plotly_chart(fig5, use_container_width=True)

with col_cs:
    cls = [0.90, 0.95, 0.975, 0.99, 0.999]
    hist_cl, mc_cl = [], []
    for cl in cls:
        v, _ = HistoricalVaR(portfolio, confidence=cl, hp=holding_period).compute()
        m, _ = MonteCarloVaR(portfolio, confidence=cl, hp=holding_period, n_sims=2000).compute()
        hist_cl.append(v / 1e6)
        mc_cl.append(m / 1e6)

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=[f"{c*100:.1f}%" for c in cls], y=hist_cl,
                              name="Historical VaR", line=dict(color="#5c6bc0", width=2),
                              mode="lines+markers"))
    fig6.add_trace(go.Scatter(x=[f"{c*100:.1f}%" for c in cls], y=mc_cl,
                              name="Monte Carlo VaR", line=dict(color="#26a69a", width=2),
                              mode="lines+markers"))
    fig6.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title="Confidence Level", yaxis_title="VaR ($M)",
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130"
    )
    st.plotly_chart(fig6, use_container_width=True)

# ── Correlation heatmap ───────────────────────────────────────────────────────
st.markdown("#### Spread Return Correlation Matrix")
corr = portfolio.spread_returns.corr()
fig7 = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale="RdBu", zmid=0,
    text=np.round(corr.values, 2),
    texttemplate="%{text}",
))
fig7.update_layout(
    template="plotly_dark", height=300,
    margin=dict(l=20, r=20, t=10, b=20),
    paper_bgcolor="#1e2130", plot_bgcolor="#1e2130"
)
st.plotly_chart(fig7, use_container_width=True)

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("Show Backtesting Detail Table"):
    st.dataframe(
        bt_results['detail_df'].tail(60).style.format({
            'pnl':        '${:,.0f}',
            'var':        '${:,.0f}',
            'exceedance': '{}'
        }).map(lambda v: 'color: #ef5350' if v else '', subset=['exceedance']),
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<small>Data: FRED (Federal Reserve Economic Data) | "
    "Models: Historical Simulation, Monte Carlo (Cholesky), Stressed VaR | "
    "Built for portfolio demonstration purposes.</small>",
    unsafe_allow_html=True
)
