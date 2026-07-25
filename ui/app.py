"""
ui/app.py
---------
Streamlit dashboard covering:
  - Model up-time / live status
  - Data visualizations (dataset feature stories)
  - Single-image prediction
  - Bulk upload + retrain trigger

Run with: streamlit run ui/app.py
Set API_URL env var if the API isn't at http://localhost:8000
"""

import os
import time
import requests
import streamlit as st
import pandas as pd
import plotly.express as px

API_URL = os.environ.get("API_URL", "http://localhost:8000")
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

st.set_page_config(page_title="Scene Classifier", layout="wide")
st.title("🏞️ Scene Classifier — MLOps Dashboard")

tab_status, tab_viz, tab_predict, tab_retrain = st.tabs(
    ["📡 Model Status", "📊 Data Insights", "🔮 Predict", "🔁 Upload & Retrain"]
)

# ---------------------------------------------------------------- STATUS ---
with tab_status:
    st.subheader("Model Up-time & Live Traffic")
    try:
        up = requests.get(f"{API_URL}/uptime", timeout=5).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Uptime", up["uptime_human"])
        c2.metric("Model loaded", "✅" if up["model_loaded"] else "❌")
        c3.metric("Images pending retrain", up["pending_retrain_images"])
    except Exception as e:
        st.error(f"Could not reach API at {API_URL}: {e}")

    try:
        live = requests.get(f"{API_URL}/metrics/live", timeout=5).json()
        preds = pd.DataFrame(live["recent_predictions"])
        if not preds.empty:
            st.markdown("**Recent prediction latency (ms)**")
            st.plotly_chart(px.line(preds.iloc[::-1], y="latency_ms", title="Latency over recent requests"),
                             use_container_width=True)
            st.markdown("**Prediction class distribution (recent traffic)**")
            st.plotly_chart(px.histogram(preds, x="predicted_class"), use_container_width=True)
        else:
            st.info("No predictions logged yet — try the Predict tab.")
    except Exception as e:
        st.warning(f"Metrics unavailable: {e}")

    if st.button("Refresh"):
        st.rerun()

# ------------------------------------------------------------------ VIZ ---
with tab_viz:
    st.subheader("Dataset Feature Stories")
    st.caption("Precomputed from src.preprocessing.get_image_stats_df on the training set. "
               "Run the notebook's EDA section (or `python -m src.preprocessing`) to regenerate "
               "data/image_stats.csv if it's missing.")
    stats_path = "data/image_stats.csv"
    if os.path.exists(stats_path):
        df = pd.read_csv(stats_path)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.box(df, x="class", y="brightness", title="Brightness by scene class"),
                use_container_width=True,
            )
            st.markdown("*Story:* glacier/sea scenes skew brighter (snow, sky, water reflectance) "
                        "while forest/street scenes skew darker (dense canopy, shadowed buildings).")
        with col2:
            df_rgb = df.melt(id_vars="class", value_vars=["mean_r", "mean_g", "mean_b"],
                              var_name="channel", value_name="value")
            st.plotly_chart(
                px.box(df_rgb, x="class", y="value", color="channel", title="Color channel means by class"),
                use_container_width=True,
            )
            st.markdown("*Story:* forest/mountain classes lean green/brown-dominant, sea/glacier "
                        "lean blue-dominant — the model is implicitly learning color signatures.")
        st.plotly_chart(
            px.histogram(df, x="class", title="Class balance in the training set"),
            use_container_width=True,
        )
        st.markdown("*Story:* class balance affects which mistakes the model is more prone to; "
                    "a skewed class often needs oversampling or class weights.")
    else:
        st.info(f"No stats file found at `{stats_path}` yet.")

# -------------------------------------------------------------- PREDICT ---
with tab_predict:
    st.subheader("Predict a single scene image")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.image(uploaded, width=300)
        if st.button("Run prediction"):
            with st.spinner("Calling API..."):
                files = {"file": (uploaded.name, uploaded.getvalue())}
                resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            if resp.ok:
                result = resp.json()
                st.success(f"Prediction: **{result['predicted_class']}** "
                           f"({result['confidence']*100:.1f}% confidence, "
                           f"{result['latency_ms']} ms)")
                st.bar_chart(pd.Series(result["all_scores"]))
            else:
                st.error(resp.text)

# -------------------------------------------------------------- RETRAIN ---
with tab_retrain:
    st.subheader("Bulk upload labeled images for retraining")
    label = st.selectbox("Class label for this batch", CLASS_NAMES)
    bulk_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=True)
    if bulk_files and st.button("Upload batch"):
        files = [("files", (f.name, f.getvalue())) for f in bulk_files]
        resp = requests.post(f"{API_URL}/upload-retrain-data",
                              data={"label": label}, files=files, timeout=60)
        if resp.ok:
            r = resp.json()
            st.success(f"Saved {r['saved']} images. Total pending: {r['total_pending']}.")
            if r["auto_retrain_ready"]:
                st.info("Enough images have accumulated to auto-trigger a retrain.")
        else:
            st.error(resp.text)

    st.divider()
    st.subheader("Trigger retraining")
    status = requests.get(f"{API_URL}/retrain/status", timeout=5).json()
    st.write(f"Pending images: **{status['pending_images']}** | Running: **{status['running']}**")
    if st.button("🚀 Retrain now", disabled=status["running"]):
        resp = requests.post(f"{API_URL}/retrain", timeout=10)
        if resp.ok:
            st.success(resp.json())
        else:
            st.error(resp.text)

    if status["history"]:
        st.markdown("**Retrain job history**")
        hist_df = pd.json_normalize([h["report"] | {"created_at": h["created_at"]} for h in status["history"]])
        st.dataframe(hist_df)
