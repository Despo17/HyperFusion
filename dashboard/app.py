import sys
import os
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
port = int(os.environ.get("PORT", 8501))
# ==============================
# PATH FIX
# ==============================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.market_data import update_market_data, ASSETS
from src.features.volatility_features import add_features
from src.live.live_predict import build_live_sequence
from src.inference.predictor_multi import MultiAssetPredictor

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="HyperFusion Pro", layout="wide")

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.signal-buy { color: #22c55e; font-size: 22px; font-weight: bold; }
.signal-sell { color: #ef4444; font-size: 22px; font-weight: bold; }
.signal-hold { color: #facc15; font-size: 22px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE + ASSET HEADER
# ==============================
st.title("🚀 HyperFusion AI Trading Dashboard")

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Controls")

asset = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
refresh_rate = st.sidebar.slider("Refresh Interval", 5, 60, 10)

# ✅ SHOW ASSET ON MAIN SCREEN
st.markdown(f"""
<div style="
    background-color:#111827;
    padding:15px;
    border-radius:10px;
    margin-bottom:15px;
">
    <h3 style="margin:0;">📌 Active Asset: {asset}</h3>
</div>
""", unsafe_allow_html=True)

# ==============================
# DATA
# ==============================
df = update_market_data(asset)
df_feat = add_features(df)

if len(df_feat) < 30:
    st.error("Not enough data")
    st.stop()

seq = build_live_sequence(df_feat)
seq = np.squeeze(seq)
if len(seq.shape) == 2:
    seq = np.expand_dims(seq, axis=0)

# ==============================
# MODEL
# ==============================
model = MultiAssetPredictor()
vol_mean_20 = df_feat["vol_mean_20"].iloc[-1]
pred = model.predict(seq, asset, vol_mean_20)

# ==============================
# SCALING FIX
# ==============================
latest_vol = df_feat["volatility"].iloc[-1]

pred_pct = pred * 100
latest_pct = latest_vol * 100
delta_pct = pred_pct - latest_pct

# ==============================
# KPI CARDS
# ==============================
col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="metric-card">
<h4>📊 Predicted</h4>
<h2>{pred_pct:.2f}%</h2>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="metric-card">
<h4>📉 Current</h4>
<h2>{latest_pct:.2f}%</h2>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="metric-card">
<h4>📈 Delta</h4>
<h2>{delta_pct:.2f}%</h2>
</div>
""", unsafe_allow_html=True)

# ==============================
# SIGNAL
# ==============================
st.subheader("📌 Trading Signal")

if pred_pct > latest_pct * 1.2:
    signal = "SELL"
    css_class = "signal-sell"
elif pred_pct < latest_pct * 0.8:
    signal = "BUY"
    css_class = "signal-buy"
else:
    signal = "HOLD"
    css_class = "signal-hold"

st.markdown(f'<div class="{css_class}">{signal}</div>', unsafe_allow_html=True)

# ==============================
# CONFIDENCE
# ==============================
confidence = max(0, min(100, 100 - abs(delta_pct) * 5))

st.subheader("🎯 Confidence")
st.progress(confidence / 100)
st.write(f"{confidence:.1f}% confidence")

# ==============================
# AI INSIGHT
# ==============================
st.subheader("🤖 AI Insight")

if signal == "BUY":
    st.success("Volatility expected to decrease → good entry opportunity")
elif signal == "SELL":
    st.error("Volatility spike expected → high risk")
else:
    st.warning("Market stable → no strong signal")

# ==============================
# 🔥 PLOTLY CHART (UPGRADED)
# ==============================
st.subheader("📈 Volatility Trend")

chart_data = df_feat.tail(60).copy()
chart_data["volatility"] = chart_data["volatility"] * 100

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=chart_data.index,
    y=chart_data["volatility"],
    mode='lines',
    line=dict(width=3),
    fill='tozeroy',
    hovertemplate='<b>Date</b>: %{x}<br><b>Volatility</b>: %{y:.2f}%<extra></extra>'
))

fig.update_layout(
    template="plotly_dark",
    height=400,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="",
    yaxis_title="Volatility (%)",
    showlegend=False
)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

st.plotly_chart(fig, use_container_width=True)

# ==============================
# AUTO REFRESH
# ==============================
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("⚡ HyperFusion Pro Dashboard")