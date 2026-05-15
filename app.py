from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "CC GENERAL.csv"
SEGMENTS_FILE = BASE_DIR / "credit_card_segments_improved.csv"
PIPELINE_FILE = BASE_DIR / "segmentation_pipeline.joblib"
RANDOM_STATE = 42

FEATURES = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "PRC_FULL_PAYMENT",
    "BALANCE_TO_LIMIT",
    "PURCHASES_TO_LIMIT",
    "CASH_ADVANCE_RATIO",
    "PAYMENT_TO_BALANCE",
    "PURCHASE_FREQUENCY_SCORE",
    "TOTAL_TRX",
]

RECOMMENDATIONS = {
    "Cash Advance Heavy": "Offer debt restructuring, monitor credit risk, and promote alternatives to cash advances.",
    "Transactors": "Reward with cashback, loyalty benefits, and higher-tier card offers.",
    "High-Risk Revolvers": "Use repayment nudges, proactive risk alerts, and avoid aggressive limit increases.",
    "Premium Spenders": "Prioritize retention, travel partnerships, and premium cross-sell offers.",
    "Low Activity": "Run reactivation campaigns with simple first-purchase incentives.",
    "Moderate Users": "Use targeted category offers and spend-based promotions.",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["BALANCE_TO_LIMIT"] = out["BALANCE"] / (out["CREDIT_LIMIT"] + 1)
    out["PURCHASES_TO_LIMIT"] = out["PURCHASES"] / (out["CREDIT_LIMIT"] + 1)
    out["CASH_ADVANCE_RATIO"] = out["CASH_ADVANCE"] / (out["BALANCE"] + 1)
    out["PAYMENT_TO_BALANCE"] = out["PAYMENTS"] / (out["BALANCE"] + 1)
    out["PURCHASE_FREQUENCY_SCORE"] = (
        out["PURCHASES_FREQUENCY"] + out["PURCHASES_INSTALLMENTS_FREQUENCY"]
    ) / 2
    out["TOTAL_TRX"] = out["PURCHASES_TRX"] + out["CASH_ADVANCE_TRX"]
    return out


def label_segments(profile: pd.DataFrame) -> dict[int, str]:
    normalized = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    labels: dict[int, str] = {}
    for cluster, row in normalized.iterrows():
        if row["CASH_ADVANCE_RATIO"] > 0.65 and row["PAYMENT_TO_BALANCE"] < 0.45:
            labels[int(cluster)] = "Cash Advance Heavy"
        elif row["PRC_FULL_PAYMENT"] > 0.65 and row["PURCHASES"] > 0.55:
            labels[int(cluster)] = "Transactors"
        elif row["BALANCE_TO_LIMIT"] > 0.7 and row["PAYMENT_TO_BALANCE"] < 0.35:
            labels[int(cluster)] = "High-Risk Revolvers"
        elif row["CREDIT_LIMIT"] > 0.75 and row["PURCHASES"] > 0.6:
            labels[int(cluster)] = "Premium Spenders"
        elif row["PURCHASES"] < 0.3 and row["TOTAL_TRX"] < 0.3:
            labels[int(cluster)] = "Low Activity"
        else:
            labels[int(cluster)] = "Moderate Users"
    return labels


def train_model() -> dict:
    raw = pd.read_csv(DATA_FILE)
    ids = raw["CUST_ID"] if "CUST_ID" in raw.columns else pd.Series(range(len(raw)))
    numeric = raw.drop(columns=["CUST_ID"], errors="ignore")

    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(imputer.fit_transform(numeric), columns=numeric.columns)
    engineered = add_features(imputed)
    model_df = engineered[FEATURES].copy()

    log_df = model_df.copy()
    log_cols = []
    for col in log_df.columns:
        if log_df[col].skew() > 1 and not log_df[col].between(0, 1).all():
            log_cols.append(col)
            log_df[col] = np.log1p(log_df[col].clip(lower=0))

    bounds = {
        col: (log_df[col].quantile(0.01), log_df[col].quantile(0.99))
        for col in log_df.columns
    }
    capped = log_df.copy()
    for col, (lower, upper) in bounds.items():
        capped[col] = capped[col].clip(lower, upper)

    scaler = RobustScaler()
    scaled = scaler.fit_transform(capped)
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    modeled = pca.fit_transform(scaled)

    k_scores = []
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30).fit_predict(modeled)
        k_scores.append({"k": k, "silhouette": silhouette_score(modeled, labels)})
    k_scores_df = pd.DataFrame(k_scores).sort_values("silhouette", ascending=False)
    best_k = int(k_scores_df.iloc[0]["k"])

    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=30)
    clusters = model.fit_predict(modeled)
    profile = model_df.assign(Cluster=clusters).groupby("Cluster")[FEATURES].mean().round(3)
    segment_names = label_segments(profile)

    plot_xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
    result = raw.copy()
    result["Cluster"] = clusters
    result["Segment"] = [segment_names[int(cluster)] for cluster in clusters]

    exported = pd.read_csv(SEGMENTS_FILE) if SEGMENTS_FILE.exists() else result

    return {
        "raw": raw,
        "ids": ids,
        "numeric_columns": numeric.columns.tolist(),
        "imputer": imputer,
        "bounds": bounds,
        "log_cols": log_cols,
        "scaler": scaler,
        "pca": pca,
        "model": model,
        "best_k": best_k,
        "k_scores": k_scores_df,
        "profile": profile,
        "segment_names": segment_names,
        "plot_df": pd.DataFrame(
            {
                "PCA 1": plot_xy[:, 0],
                "PCA 2": plot_xy[:, 1],
                "Cluster": clusters,
                "Segment": [segment_names[int(cluster)] for cluster in clusters],
            }
        ),
        "result": result,
        "exported": exported,
    }


@st.cache_resource(show_spinner="Loading saved segmentation pipeline...")
def load_model_pack() -> dict:
    if PIPELINE_FILE.exists():
        return joblib.load(PIPELINE_FILE)

    model_pack = train_model()
    joblib.dump(model_pack, PIPELINE_FILE)
    model_pack["result"].to_csv(SEGMENTS_FILE, index=False)
    return model_pack


def prepare_new_customer(model_pack: dict, row: pd.DataFrame) -> np.ndarray:
    full_row = pd.DataFrame(columns=model_pack["numeric_columns"])
    for col in full_row.columns:
        full_row.loc[0, col] = row.iloc[0].get(col, np.nan)
    full_row = full_row.astype(float)

    imputed = pd.DataFrame(
        model_pack["imputer"].transform(full_row), columns=model_pack["numeric_columns"]
    )
    engineered = add_features(imputed)
    selected = engineered[FEATURES].copy()
    for col in model_pack["log_cols"]:
        selected[col] = np.log1p(selected[col].clip(lower=0))
    for col, (lower, upper) in model_pack["bounds"].items():
        selected[col] = selected[col].clip(lower, upper)
    scaled = model_pack["scaler"].transform(selected)
    return model_pack["pca"].transform(scaled)


def configure_page() -> None:
    st.set_page_config(page_title="Credit Card Segmentation Studio", page_icon="CC", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #f7faf8; color: #18332f; }
        [data-testid="stSidebar"] { background: #18332f; }
        [data-testid="stSidebar"] * { color: #faf7ef !important; }
        .metric-box, .note-box {
            background: #fffffb;
            border: 1px solid rgba(24, 51, 47, 0.12);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(24, 51, 47, 0.08);
        }
        .metric-label { color: #5f746e; font-size: .9rem; }
        .metric-value { color: #18332f; font-size: 1.7rem; font-weight: 800; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    model_pack = load_model_pack()
    raw = model_pack["raw"]

    st.title("Credit Card Segmentation Studio")
    st.caption("Single-file GUI connected to the project notebook and data files.")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Data", "EDA", "Model", "Segments", "Predict"],
    )

    if page == "Overview":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric("Customers", f"{len(raw):,}")
        with col2:
            metric("Raw Features", str(raw.drop(columns=["CUST_ID"], errors="ignore").shape[1]))
        with col3:
            metric("Best K", str(model_pack["best_k"]))
        with col4:
            metric("Model", "KMeans")
        st.subheader("Project files used")
        st.write(f"Data: `{DATA_FILE.name}`")
        st.write(f"Saved model pipeline: `{PIPELINE_FILE.name}`")
        st.write(f"Segmented data: `{SEGMENTS_FILE.name}`")
        st.dataframe(model_pack["result"].head(20), use_container_width=True)

    elif page == "Data":
        missing = raw.isna().sum().rename("missing_count").reset_index()
        missing.columns = ["feature", "missing_count"]
        missing["missing_pct"] = (100 * missing["missing_count"] / len(raw)).round(2)
        st.subheader("Missing values")
        st.dataframe(missing.query("missing_count > 0"), use_container_width=True, hide_index=True)
        st.subheader("Raw data preview")
        st.dataframe(raw.head(30), use_container_width=True)

    elif page == "EDA":
        feature = st.selectbox(
            "Feature",
            ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT"],
        )
        fig = px.histogram(raw, x=feature, nbins=45, marginal="box", title=f"{feature} distribution")
        st.plotly_chart(fig, use_container_width=True)
        corr_cols = [col for col in FEATURES if col in add_features(raw.drop(columns=["CUST_ID"], errors="ignore")).columns]
        corr_df = add_features(raw.drop(columns=["CUST_ID"], errors="ignore"))[corr_cols].corr()
        st.plotly_chart(
            px.imshow(corr_df.round(2), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r"),
            use_container_width=True,
        )

    elif page == "Model":
        st.subheader("K selection")
        st.dataframe(model_pack["k_scores"].round(4), use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.line(model_pack["k_scores"].sort_values("k"), x="k", y="silhouette", markers=True),
            use_container_width=True,
        )
        st.subheader("2D cluster map")
        st.plotly_chart(
            px.scatter(model_pack["plot_df"], x="PCA 1", y="PCA 2", color="Segment", opacity=0.75),
            use_container_width=True,
        )

    elif page == "Segments":
        st.subheader("Segment size")
        size_df = model_pack["result"]["Segment"].value_counts().rename_axis("Segment").reset_index()
        size_df.columns = ["Segment", "Customers"]
        st.plotly_chart(px.pie(size_df, names="Segment", values="Customers", hole=0.45), use_container_width=True)
        st.subheader("Cluster profile")
        profile = model_pack["profile"].copy()
        profile.index = [model_pack["segment_names"][int(idx)] for idx in profile.index]
        st.dataframe(profile, use_container_width=True)
        st.subheader("Recommendations")
        for segment in size_df["Segment"]:
            st.markdown(f"**{segment}:** {RECOMMENDATIONS.get(segment, 'Review customer behavior manually.')}")

    elif page == "Predict":
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                balance = st.number_input("BALANCE", min_value=0.0, value=2500.0, step=100.0)
                balance_frequency = st.slider("BALANCE_FREQUENCY", 0.0, 1.0, 0.9, 0.01)
                purchases = st.number_input("PURCHASES", min_value=0.0, value=500.0, step=50.0)
                oneoff_purchases = st.number_input("ONEOFF_PURCHASES", min_value=0.0, value=300.0, step=50.0)
                installments_purchases = st.number_input("INSTALLMENTS_PURCHASES", min_value=0.0, value=200.0, step=50.0)
                cash_advance = st.number_input("CASH_ADVANCE", min_value=0.0, value=3000.0, step=100.0)
            with col2:
                purchases_frequency = st.slider("PURCHASES_FREQUENCY", 0.0, 1.0, 0.3, 0.01)
                oneoff_frequency = st.slider("ONEOFF_PURCHASES_FREQUENCY", 0.0, 1.0, 0.2, 0.01)
                installments_frequency = st.slider("PURCHASES_INSTALLMENTS_FREQUENCY", 0.0, 1.0, 0.2, 0.01)
                cash_frequency = st.slider("CASH_ADVANCE_FREQUENCY", 0.0, 1.0, 0.8, 0.01)
                cash_trx = st.number_input("CASH_ADVANCE_TRX", min_value=0, value=12, step=1)
                purchases_trx = st.number_input("PURCHASES_TRX", min_value=0, value=5, step=1)
            with col3:
                credit_limit = st.number_input("CREDIT_LIMIT", min_value=1.0, value=5000.0, step=100.0)
                payments = st.number_input("PAYMENTS", min_value=0.0, value=400.0, step=50.0)
                minimum_payments = st.number_input("MINIMUM_PAYMENTS", min_value=0.0, value=200.0, step=25.0)
                prc_full_payment = st.slider("PRC_FULL_PAYMENT", 0.0, 1.0, 0.05, 0.01)
                tenure = st.number_input("TENURE", min_value=1, max_value=12, value=10, step=1)
            submitted = st.form_submit_button("Predict Segment")

        if submitted:
            row = pd.DataFrame(
                [
                    {
                        "BALANCE": balance,
                        "BALANCE_FREQUENCY": balance_frequency,
                        "PURCHASES": purchases,
                        "ONEOFF_PURCHASES": oneoff_purchases,
                        "INSTALLMENTS_PURCHASES": installments_purchases,
                        "CASH_ADVANCE": cash_advance,
                        "PURCHASES_FREQUENCY": purchases_frequency,
                        "ONEOFF_PURCHASES_FREQUENCY": oneoff_frequency,
                        "PURCHASES_INSTALLMENTS_FREQUENCY": installments_frequency,
                        "CASH_ADVANCE_FREQUENCY": cash_frequency,
                        "CASH_ADVANCE_TRX": cash_trx,
                        "PURCHASES_TRX": purchases_trx,
                        "CREDIT_LIMIT": credit_limit,
                        "PAYMENTS": payments,
                        "MINIMUM_PAYMENTS": minimum_payments,
                        "PRC_FULL_PAYMENT": prc_full_payment,
                        "TENURE": tenure,
                    }
                ]
            )
            transformed = prepare_new_customer(model_pack, row)
            cluster = int(model_pack["model"].predict(transformed)[0])
            segment = model_pack["segment_names"][cluster]
            st.success(f"Predicted segment: {segment} (Cluster {cluster})")
            st.write(RECOMMENDATIONS.get(segment, "Review customer behavior manually."))


if __name__ == "__main__":
    main()
