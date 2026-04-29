import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
import textwrap

from utils import (
    CONTINUOUS_COLS,
    BINARY_COLS,
    CATEGORICAL_COLS,
    prepare_input_dataframe,
    load_ttvae,
    load_feature_names,
    load_cluster_model,
    load_pseudotime_bounds,
    load_ood_threshold,
    compute_latent,
    compute_pseudotime,
    assign_cluster,
    batched_reconstruction_error,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TB Risk Profiling System",
    page_icon="📊🫁",
    layout="wide",
)

# ============================================================
# CUSTOM UI STYLING
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #060b17;
        --card: rgba(13, 20, 38, 0.88);
        --card-2: rgba(12, 18, 34, 0.96);
        --border: rgba(114, 137, 218, 0.22);
        --text: #eef2ff;
        --muted: #a8b2d1;
        --pink: #ff4d8d;
        --purple: #7c4dff;
        --cyan: #2dd4bf;
        --blue: #3b82f6;
        --green: #10b981;
        --green-2: #059669;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(76, 29, 149, 0.20), transparent 25%),
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.14), transparent 25%),
            linear-gradient(180deg, #040915 0%, #050b17 50%, #06101d 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    /* Safe text color: avoid styling every Streamlit internal div */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: var(--text);
    }

    .hero-wrap {
        padding: 0.8rem 0 1.2rem 0;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.42rem 0.9rem;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(124,77,255,0.25), rgba(59,130,246,0.18));
        border: 1px solid rgba(124,77,255,0.25);
        color: #d9d7ff;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 4rem;
        line-height: 1.02;
        font-weight: 800;
        margin: 0 0 0.85rem 0;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        font-size: 1.35rem;
        line-height: 1.6;
        color: var(--muted);
        max-width: 760px;
        margin-bottom: 1.5rem;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 0.8rem;
    }

    .feature-card {
        background: rgba(13, 20, 38, 0.70);
        border: 1px solid rgba(114, 137, 218, 0.18);
        border-radius: 18px;
        padding: 1rem 1rem 0.95rem 1rem;
        min-height: 120px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.18);
    }

    .feature-title {
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .feature-text {
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .visual-card {
        background:
            radial-gradient(circle at center, rgba(124,77,255,0.26), transparent 45%),
            linear-gradient(180deg, rgba(15,22,40,0.95), rgba(9,14,28,0.95));
        border: 1px solid rgba(114, 137, 218, 0.18);
        border-radius: 28px;
        min-height: 370px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: inset 0 0 60px rgba(62, 90, 255, 0.08);
        position: relative;
        overflow: hidden;
    }

    .visual-card:before,
    .visual-card:after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        border: 1px solid rgba(124,77,255,0.14);
        border-radius: 24px;
        transform: rotate(24deg);
    }

    .visual-card:before {
        top: 18px;
        right: 42px;
    }

    .visual-card:after {
        bottom: 22px;
        left: 40px;
    }

    .lung {
        font-size: 10rem;
        filter: drop-shadow(0 0 25px rgba(124,77,255,0.50));
    }

    .section-card {
        background: linear-gradient(180deg, rgba(13,20,38,0.96), rgba(8,14,28,0.98));
        border: 1px solid rgba(114, 137, 218, 0.20);
        border-radius: 24px;
        padding: 1.15rem 1.15rem 1.25rem 1.15rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .section-head {
        display: flex;
        gap: 0.9rem;
        align-items: flex-start;
        margin-bottom: 0.8rem;
    }

    .section-icon {
        width: 56px;
        height: 56px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        background: linear-gradient(135deg, rgba(124,77,255,0.28), rgba(59,130,246,0.18));
        border: 1px solid rgba(124,77,255,0.22);
        flex-shrink: 0;
    }

    .section-title {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0;
    }

    .section-subtitle {
        color: var(--muted);
        margin-top: 0.2rem;
        font-size: 1rem;
    }

    .soft-note {
        margin-top: 0.85rem;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(8, 13, 26, 0.72);
        border: 1px solid rgba(114, 137, 218, 0.16);
        color: var(--muted);
        font-size: 0.96rem;
    }

    .footer-note {
        margin-top: 1rem;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(8, 13, 26, 0.60);
        border: 1px solid rgba(114, 137, 218, 0.18);
        color: var(--muted);
        font-size: 0.95rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(14, 22, 41, 0.96), rgba(10, 15, 28, 0.96));
        border: 1px solid rgba(114, 137, 218, 0.18);
        border-radius: 18px;
        padding: 0.35rem 0.35rem 0.2rem 0.35rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.16);
    }

    div[data-testid="stMetric"] {
        background: transparent;
        border: none;
        padding: 0.65rem 0.8rem;
        border-radius: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: white;
        font-weight: 800;
    }

    /* Main buttons */
    .stButton > button {
        border: none !important;
        border-radius: 14px !important;
        padding: 0.78rem 1.35rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        color: white !important;
        background: linear-gradient(90deg, var(--pink), var(--purple)) !important;
        box-shadow: 0 10px 25px rgba(124,77,255,0.26) !important;
        transition: all 0.2s ease !important;
        opacity: 1 !important;
    }

    .stButton > button:hover {
        filter: brightness(1.08) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button:focus,
    .stButton > button:active {
        outline: none !important;
        box-shadow: 0 0 0 0.2rem rgba(124, 77, 255, 0.25) !important;
    }

    .stButton > button:disabled {
        opacity: 0.55 !important;
        cursor: not-allowed !important;
    }

    /* File uploader container */
    div[data-testid="stFileUploader"] {
        background: rgba(10, 16, 30, 0.72);
        border: 1px solid rgba(114, 137, 218, 0.18);
        border-radius: 18px;
        padding: 0.9rem;
    }

    /* File uploader button */
    div[data-testid="stFileUploader"] section button,
    div[data-testid="stFileUploader"] button[kind="secondary"] {
        background: linear-gradient(90deg, var(--blue), var(--purple)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.7rem 1.1rem !important;
        opacity: 1 !important;
        box-shadow: 0 8px 20px rgba(59,130,246,0.24) !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stFileUploader"] section button:hover,
    div[data-testid="stFileUploader"] button[kind="secondary"]:hover {
        filter: brightness(1.08) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span {
        color: var(--muted) !important;
    }

    /* Download buttons */
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(90deg, var(--green), var(--green-2)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.78rem 1.2rem !important;
        opacity: 1 !important;
        box-shadow: 0 8px 20px rgba(16,185,129,0.22) !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stDownloadButton"] > button:hover {
        filter: brightness(1.08) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stDownloadButton"] > button:focus,
    div[data-testid="stDownloadButton"] > button:active {
        outline: none !important;
        box-shadow: 0 0 0 0.2rem rgba(16, 185, 129, 0.22) !important;
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stSlider {
        background: transparent;
    }

    /* Tables and expanders */
    .stDataFrame, .stTable, div[data-testid="stExpander"] {
        border-radius: 18px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(114, 137, 218, 0.16);
        background: rgba(11, 16, 30, 0.75);
    }

    /* Expander header hover/focus fix only */
    div[data-testid="stExpander"] details summary {
        color: var(--text) !important;
        background: transparent !important;
        border-radius: 18px !important;
        transition: background 0.2s ease, color 0.2s ease !important;
    }

    div[data-testid="stExpander"] details summary:hover {
        background: rgba(124, 77, 255, 0.10) !important;
        color: #ffffff !important;
    }

    div[data-testid="stExpander"] details summary:focus,
    div[data-testid="stExpander"] details summary:focus-visible,
    div[data-testid="stExpander"] details[open] summary {
        background: rgba(124, 77, 255, 0.14) !important;
        color: #ffffff !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Keep expander content readable */
    div[data-testid="stExpander"] details > div {
        background: transparent !important;
        color: var(--text) !important;
    }

    hr {
        border-color: rgba(114, 137, 218, 0.12);
    }



    /* ============================================================ */
    /* STREAMLIT SELECTBOX / DROPDOWN FIX FOR DARK THEME             */
    /* ============================================================ */
    div[data-baseweb="select"] > div {
        background-color: #071121 !important;
        border: 1px solid rgba(114, 137, 218, 0.35) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }

    div[data-baseweb="select"] input {
        color: #ffffff !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #ffffff !important;
    }

    div[data-baseweb="select"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    div[data-baseweb="popover"] {
        background-color: #0b1324 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(114, 137, 218, 0.35) !important;
        box-shadow: 0 16px 40px rgba(0,0,0,0.35) !important;
        overflow: hidden !important;
    }

    div[data-baseweb="menu"] {
        background-color: #0b1324 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="menu"] ul {
        background-color: #0b1324 !important;
    }

    div[data-baseweb="menu"] li,
    div[role="option"] {
        background-color: #0b1324 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="menu"] li:hover,
    div[role="option"]:hover {
        background-color: rgba(124,77,255,0.35) !important;
        color: #ffffff !important;
    }

    div[role="option"][aria-selected="true"] {
        background-color: rgba(45,212,191,0.25) !important;
        color: #ffffff !important;
    }

    div[data-baseweb="select"] div[aria-disabled="true"],
    div[data-baseweb="menu"] div[aria-disabled="true"] {
        color: #cbd5e1 !important;
        opacity: 1 !important;
    }

    div[role="radiogroup"] label,
    div[role="radiogroup"] span {
        color: #eef2ff !important;
    }

    div[data-testid="stAlert"] * {
        color: inherit !important;
    }

    @media (max-width: 1100px) {
        .hero-title {
            font-size: 2.8rem;
        }

        .feature-grid {
            grid-template-columns: 1fr;
        }

        .visual-card {
            min-height: 240px;
        }

        .lung {
            font-size: 6rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)




# ============================================================
# INTERPRETABLE CLUSTER DEFINITIONS
# Ordered by risk progression from your saved cluster outputs:
# 3 -> 1 -> 4 -> 2 -> 0
# ============================================================
CLUSTER_INFO = {
    3: {
        "name": "High-Risk / Active TB-Like Profile",
        "stage": "Earliest / Most severe risk stage",
        "risk": "High Risk",
        "summary": "Strong cough, chest pain, sputum and other symptomatic TB signals.",
        "key_features": ["cough", "chest_pain", "sputum", "fever", "weight_loss"],
    },
    1: {
        "name": "Symptomatic TB-Like Profile",
        "stage": "Early risk stage",
        "risk": "High Risk",
        "summary": "Moderate cough and chest pain with clinically relevant symptom burden.",
        "key_features": ["cough", "chest_pain", "fever", "weight_loss"],
    },
    4: {
        "name": "Transitional TB Risk Profile",
        "stage": "Intermediate / transition stage",
        "risk": "Moderate Risk",
        "summary": "Milder but emerging symptom profile, suggesting transition along risk progression.",
        "key_features": ["chest_pain", "cough", "weight_loss"],
    },
    2: {
        "name": "Low-Symptom Profile",
        "stage": "Later / relatively stable stage",
        "risk": "Low Risk",
        "summary": "Very low symptom burden and weak clinical activity.",
        "key_features": ["minimal symptoms", "stable radiology"],
    },
    0: {
        "name": "Very Low-Risk / Stable Profile",
        "stage": "Latest / most stable stage",
        "risk": "Very Low Risk",
        "summary": "Minimal symptoms and stable overall profile.",
        "key_features": ["minimal symptoms", "stable profile"],
    },
}

CLUSTER_ORDER = [3, 1, 4, 2, 0]

MAX_ROWS = 30000
BATCH_SIZE = 1024
LATENT_DIM = 16

ALL_COLS = CONTINUOUS_COLS + BINARY_COLS + CATEGORICAL_COLS

# ============================================================
# USER-FACING DEPLOYMENT FEATURES
# ============================================================
# These are the 15 variables shown in the hospital mapping grid.
# The full trained schema is still preserved internally through ALL_COLS
# so the saved preprocessor and TTVAE model remain compatible.
DISPLAY_COLS = [
    # Demographic / baseline
    "age_census",
    "occupation",

    # Core TB symptoms
    "chest_pain",
    "cough",
    "sputum",
    "fever",
    "weight_loss",
    "night_sweats",

    # Symptom duration / progression
    "cough_d",
    "sputum_d",
    "fever_d",
    "wloss_d",

    # Radiology and laboratory indicators
    "xrayres",
    "smear_pos",
    "genexpert",
]



# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_resource
def load_preprocessor():
    try:
        return joblib.load("models/preprocessor.joblib")
    except Exception:
        return joblib.load("models/preprocessor.pkl")


@st.cache_resource
def load_all_artifacts():
    feature_names = load_feature_names()
    model = load_ttvae(input_dim=len(feature_names))
    kmeans = load_cluster_model()
    preprocessor = load_preprocessor()
    pt_bounds = load_pseudotime_bounds()
    ood_info = load_ood_threshold()
    return feature_names, model, kmeans, preprocessor, pt_bounds, ood_info


feature_names, model, kmeans, preprocessor, PT_BOUNDS, OOD_INFO = load_all_artifacts()

OOD_THRESHOLD = float(
    OOD_INFO.get("threshold", OOD_INFO.get("ood_threshold", 0.0))
)
OOD_PERCENTILE = int(OOD_INFO.get("percentile", 95))


# ============================================================
# HELPERS
# ============================================================
def transform_uploaded_data(df_raw: pd.DataFrame):
    df_clean = prepare_input_dataframe(df_raw)
    X = preprocessor.transform(df_clean)

    X_df = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())
    X_df = X_df.reindex(columns=feature_names, fill_value=0.0)

    return df_clean, X_df.values.astype(np.float32)


def progression_position_label(pt_norm: float) -> str:
    if pt_norm < 0.20:
        return "Earliest / Highest-Risk Position"
    if pt_norm < 0.45:
        return "Early Progression Position"
    if pt_norm < 0.70:
        return "Intermediate Progression Position"
    if pt_norm < 0.90:
        return "Late Progression Position"
    return "Most Stable / Lowest-Risk Position"


def risk_bucket_from_cluster(cluster_id: int) -> str:
    return CLUSTER_INFO.get(cluster_id, {}).get("risk", "Unknown")


def reliability_label(flag: bool) -> str:
    return "⚠️ OOD Warning" if flag else "✅ In Distribution"


def build_patient_results(latents, pseudotime_norm, clusters, rec_error, ood_flags):
    """
    Build a user-facing result table.

    Important safety rule:
    Records with reconstruction error above the training threshold are treated as
    out-of-distribution (OOD). For those records, the app abstains from assigning
    a risk profile or progression position and displays dashes instead.
    """
    rows = []

    for i in range(len(clusters)):
        is_ood = bool(ood_flags[i])

        if is_ood:
            rows.append(
                {
                    "Assessment Status": "Not assessed - outside training distribution",
                    "Clinical Profile": "—",
                    "Risk Group": "—",
                    "Risk Position": "—",
                    "Reliability": "⚠️ OOD - abstained",
                }
            )
            continue

        cid = int(clusters[i])
        info = CLUSTER_INFO.get(cid, {})

        rows.append(
            {
                "Assessment Status": "Assessed",
                "Clinical Profile": info.get("name", "Clinical profile"),
                "Risk Group": risk_bucket_from_cluster(cid),
                "Risk Position": progression_position_label(float(pseudotime_norm[i])),
                "Reliability": "✅ Within training distribution",
            }
        )

    return pd.DataFrame(rows)


def plot_profile_distribution(results_df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#0b1324")
    ax.set_facecolor("#0b1324")

    assessed = results_df[results_df["Assessment Status"] == "Assessed"]
    if assessed.empty:
        ax.text(0.5, 0.5, "No assessed records", ha="center", va="center", color="white")
        ax.set_axis_off()
        return fig

    ordered = assessed["Clinical Profile"].value_counts()
    ordered.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of Patients", color="white")
    ax.set_title("Clinical Profile Distribution", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2a3550")
    return fig


def plot_risk_group_distribution(results_df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#0b1324")
    ax.set_facecolor("#0b1324")

    assessed = results_df[results_df["Assessment Status"] == "Assessed"]
    if assessed.empty:
        ax.text(0.5, 0.5, "No assessed records", ha="center", va="center", color="white")
        ax.set_axis_off()
        return fig

    order = ["High Risk", "Moderate Risk", "Low Risk", "Very Low Risk"]
    counts = assessed["Risk Group"].value_counts().reindex(order, fill_value=0)
    counts = counts[counts > 0]
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of Patients", color="white")
    ax.set_title("Risk Group Distribution", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2a3550")
    return fig


def build_profile_summary(results_df):
    assessed = results_df[results_df["Assessment Status"] == "Assessed"]
    if assessed.empty:
        return pd.DataFrame(columns=["Clinical Profile", "Risk Group", "Patients"] )

    summary = (
        assessed.groupby(["Clinical Profile", "Risk Group"], as_index=False)
        .agg(Patients=("Clinical Profile", "count"))
        .sort_values("Patients", ascending=False)
    )
    return summary

def plot_latent_by_cluster(latents, clusters):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#0b1324")
    ax.set_facecolor("#0b1324")

    for cid in sorted(np.unique(clusters)):
        mask = clusters == cid
        label = CLUSTER_INFO.get(int(cid), {}).get("name", f"Cluster {cid}")
        ax.scatter(
            latents[mask, 0],
            latents[mask, 1],
            label=label,
            alpha=0.65,
            s=18,
        )

    ax.set_xlabel("z1", color="white")
    ax.set_ylabel("z2", color="white")
    ax.set_title("Latent Space Colored by Cluster", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2a3550")
    leg = ax.legend(fontsize=7)
    if leg:
        for text in leg.get_texts():
            text.set_color("white")
        leg.get_frame().set_facecolor("#0b1324")
        leg.get_frame().set_edgecolor("#2a3550")
    return fig


def plot_latent_by_pseudotime(latents, pseudotime_norm):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#0b1324")
    ax.set_facecolor("#0b1324")

    sc = ax.scatter(
        latents[:, 0],
        latents[:, 1],
        c=pseudotime_norm,
        cmap="viridis",
        alpha=0.7,
        s=18,
    )
    ax.set_xlabel("z1", color="white")
    ax.set_ylabel("z2", color="white")
    ax.set_title("Latent Space Colored by Pseudotime", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2a3550")
    cbar = plt.colorbar(sc, ax=ax, label="Normalized Pseudotime")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("#2a3550")
    return fig



def build_cluster_feature_profiles(df_clean, clusters):
    prof = df_clean.copy()
    prof["Cluster"] = clusters
    profile_means = prof.groupby("Cluster").mean(numeric_only=True).reset_index()
    profile_means["Phenotype"] = profile_means["Cluster"].map(
        lambda x: CLUSTER_INFO.get(int(x), {}).get("name", f"Cluster {x}")
    )

    ordered_cols = ["Cluster", "Phenotype"] + [
        c for c in profile_means.columns if c not in ["Cluster", "Phenotype"]
    ]
    profile_means = profile_means[ordered_cols]
    return profile_means



# ============================================================
# MULTI-HOSPITAL COLUMN MAPPING HELPERS
# ============================================================
def normalize_column_name(col_name: str) -> str:
    """
    Standardise column names for safer automatic matching.
    Example: "Age Census", "age-census" and "age_census" all become "agecensus".
    """
    return "".join(ch.lower() for ch in str(col_name).strip() if ch.isalnum())


def guess_column_mapping(uploaded_columns, expected_columns):
    """
    Automatically match hospital CSV columns to the trained model schema
    where the names are identical or very similar after normalisation.
    """
    normalised_uploaded = {
        normalize_column_name(col): col for col in uploaded_columns
    }

    mapping = {}
    for expected in expected_columns:
        key = normalize_column_name(expected)
        mapping[expected] = normalised_uploaded.get(key, "-- Not available --")

    return mapping


def apply_hospital_column_mapping(df_raw: pd.DataFrame, mapping: dict):
    """
    Convert hospital-specific CSV column names into the trained model column names.
    Unmapped trained variables are created as missing values so that the saved
    preprocessing pipeline can handle them consistently.
    """
    df_mapped = pd.DataFrame(index=df_raw.index)
    mapped_expected_cols = []

    for expected_col in ALL_COLS:
        source_col = mapping.get(expected_col, "-- Not available --")
        if source_col != "-- Not available --" and source_col in df_raw.columns:
            df_mapped[expected_col] = df_raw[source_col]
            mapped_expected_cols.append(expected_col)
        else:
            df_mapped[expected_col] = np.nan

    missing_expected_cols = [col for col in ALL_COLS if col not in mapped_expected_cols]
    return df_mapped, mapped_expected_cols, missing_expected_cols


def show_mapping_quality(mapped_expected_cols, missing_expected_cols):
    """
    Display simple validation feedback before running the model.
    Only the selected 15 deployment features are shown to users, while the
    remaining trained variables are created internally as missing values.
    """
    selected_mapped = [col for col in DISPLAY_COLS if col in mapped_expected_cols]
    selected_missing = [col for col in DISPLAY_COLS if col not in mapped_expected_cols]

    c1, c2, c3 = st.columns(3)
    c1.metric("Displayed Features", len(DISPLAY_COLS))
    c2.metric("Mapped Displayed Features", len(selected_mapped))
    c3.metric("Unmapped Displayed Features", len(selected_missing))

    if len(selected_mapped) == 0:
        st.error(
            "No displayed clinical variables have been mapped. "
            "Please map at least one hospital CSV column before analysis."
        )
    elif selected_missing:
        st.warning(
            "Some displayed variables were not mapped. They will be passed as missing values "
            "and handled by the saved preprocessing pipeline."
        )
        with st.expander("View unmapped displayed variables", expanded=False):
            st.write(selected_missing)
    else:
        st.success("All 15 displayed clinical variables have been mapped successfully.")

    with st.expander("Technical note: full trained schema preserved internally", expanded=False):
        st.write(
            f"The app displays {len(DISPLAY_COLS)} selected clinical variables, but internally preserves "
            f"the full trained schema of {len(ALL_COLS)} variables required by the saved preprocessor "
            "and TTVAE model. Non-displayed variables are created as missing values and handled by preprocessing."
        )



# ============================================================
# HERO SECTION (BETTER ALIGNED STREAMLIT VERSION)
# ============================================================
left_col, right_col = st.columns([1.6, 1], gap="large")

with left_col:
    st.markdown("### ✦ AI-POWERED")
    st.markdown("# TB Risk Profiling System")
    st.markdown(
        """
        latent tuberculosis risk sequencing and clinical profile discovery.
        """
    )

    st.markdown("### Key Capabilities")

    st.markdown(
        """
        **🛡 Privacy Preserving**  
        Generative synthetic data without exposing real patients
        """
    )

    st.markdown(
        """
        **✦ Advanced AI**  
        Latent representation learning and progression modeling
        """
    )

    st.markdown(
        """
        **📊 Actionable Insights**  
        Identify high-risk patterns and patient risk profiles
        """
    )

with right_col:
    st.markdown("<div style='height: 70px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; font-size: 120px;'>🫁</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align: center; margin-top: -10px;'>TB Risk Visualization</h3>",
        unsafe_allow_html=True,
    )

# ============================================================
# UPLOAD + ANALYSIS WITH MULTI-HOSPITAL COLUMN MAPPING
# ============================================================
st.markdown(
    """
    <div class="section-card">
        <div class="section-head">
            <div class="section-icon">☁️</div>
            <div>
                <div class="section-title">Upload Patient Cohort</div>
                <div class="section-subtitle">
                    Upload a hospital CSV file, map its columns to the trained model schema,
                    and run TB risk profiling.
                </div>
            </div>
        </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a CSV file containing TB patient records",
    type=["csv"],
    label_visibility="collapsed",
)

st.markdown("</div>", unsafe_allow_html=True)

results = None
df_clean = None
latents = None
pseudotime_norm = None
clusters = None
rec_error = None
ood_flags = None
analyze = False
df_raw = None
mapping = {}

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)

        if len(df_raw) == 0:
            st.error("The uploaded CSV file has no patient records.")
            st.stop()

        if len(df_raw) > MAX_ROWS:
            st.error(
                f"The uploaded dataset contains {len(df_raw):,} rows. "
                f"This deployment supports up to {MAX_ROWS:,} rows per run."
            )
            st.stop()

        st.success(
            f"CSV uploaded successfully: {len(df_raw):,} patient records and "
            f"{len(df_raw.columns):,} columns detected."
        )

        with st.expander("Preview uploaded hospital CSV", expanded=False):
            st.dataframe(df_raw.head(20), use_container_width=True)

        st.markdown("### Column Setup")
        st.info(
            "This system was trained on a 2014–2015 cross-sectional TB dataset. "
            "To support safe use, each uploaded record is checked against the training distribution. "
            "Records above the 95th percentile reconstruction-error threshold are marked as out-of-distribution "
            "and are not assigned a risk group, risk position, or clinical profile."
        )

        auto_mapping = guess_column_mapping(df_raw.columns, DISPLAY_COLS)
        available_options = ["-- Not available --"] + list(df_raw.columns)

        schema_mode = st.radio(
            "How are the CSV columns named?",
            options=[
                "CSV already uses the correct model column names",
                "CSV uses different hospital column names",
            ],
            horizontal=True,
        )

        if schema_mode == "CSV already uses the correct model column names":
            mapping = auto_mapping
            st.caption(
                "The app will automatically use matching column names from the uploaded file. "
                "Missing model variables will be handled by the saved preprocessing pipeline."
            )
        else:
            st.caption(
                "Map each displayed clinical variable to the matching column in the hospital CSV. "
                "The grid layout keeps the page compact and easier to present."
            )

            mapping = {}
            grid_cols_per_row = 3

            for start_idx in range(0, len(DISPLAY_COLS), grid_cols_per_row):
                row_variables = DISPLAY_COLS[start_idx:start_idx + grid_cols_per_row]
                cols = st.columns(grid_cols_per_row, gap="medium")

                for i, expected_col in enumerate(row_variables):
                    with cols[i]:
                        default_source = auto_mapping.get(expected_col, "-- Not available --")
                        default_index = (
                            available_options.index(default_source)
                            if default_source in available_options
                            else 0
                        )

                        st.markdown(
                            f"""
                            <div style="
                                background: rgba(10, 16, 30, 0.62);
                                border: 1px solid rgba(114, 137, 218, 0.18);
                                border-radius: 14px;
                                padding: 0.55rem 0.65rem;
                                margin-bottom: 0.25rem;
                                font-weight: 700;
                                font-size: 0.90rem;
                                color: #eef2ff;
                            ">
                                {expected_col}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        mapping[expected_col] = st.selectbox(
                            label=f"Select hospital column for {expected_col}",
                            options=available_options,
                            index=default_index,
                            key=f"map_grid_{expected_col}",
                            label_visibility="collapsed",
                        )

        df_mapped_preview, mapped_expected_cols, missing_expected_cols = apply_hospital_column_mapping(
            df_raw, mapping
        )
        show_mapping_quality(mapped_expected_cols, missing_expected_cols)

        with st.expander("Preview internally mapped model-ready columns", expanded=False):
            st.dataframe(df_mapped_preview.head(20), use_container_width=True)

        analyze = st.button(
            "Analyze Cohort",
            type="primary",
            disabled=(len(mapped_expected_cols) == 0),
        )

    except Exception as e:
        st.error(
            "The uploaded CSV file could not be read. Please confirm that it is a valid CSV file."
        )
        st.exception(e)
        st.stop()

if uploaded_file is not None and analyze:
    try:
        df_mapped, mapped_expected_cols, missing_expected_cols = apply_hospital_column_mapping(
            df_raw, mapping
        )

        df_clean, X = transform_uploaded_data(df_mapped)

        latents = compute_latent(model, X)
        pseudotime_norm = compute_pseudotime(latents, bounds=PT_BOUNDS)
        clusters = assign_cluster(kmeans, latents)

        rec_error = batched_reconstruction_error(model, X, batch_size=BATCH_SIZE)
        ood_flags = rec_error > OOD_THRESHOLD

        results = build_patient_results(
            latents=latents,
            pseudotime_norm=pseudotime_norm,
            clusters=clusters,
            rec_error=rec_error,
            ood_flags=ood_flags,
        )

        st.success("Cohort processed successfully. OOD records were safely abstained from risk assignment.")

    except Exception as e:
        st.error(
            "The uploaded data could not be processed after column mapping. "
            "Please confirm that mapped columns contain values compatible with the trained schema."
        )
        st.exception(e)
        st.stop()


# ============================================================
# RESULTS
# ============================================================
if results is not None:
    st.markdown("### Cohort Summary")

    assessed_count = int((results["Assessment Status"] == "Assessed").sum())
    ood_count = int((results["Assessment Status"] != "Assessed").sum())

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Records", f"{len(results):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Assessed Records", f"{assessed_count:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Not Assessed / OOD", f"{ood_count:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        assessed_profiles = results.loc[results["Assessment Status"] == "Assessed", "Clinical Profile"].nunique()
        st.metric("Profiles Detected", int(assessed_profiles))
        st.markdown("</div>", unsafe_allow_html=True)

    if ood_count > 0:
        st.warning(
            "Some records were outside the training distribution. For safety, the system did not assign "
            "a risk group, risk position, or clinical profile to those records."
        )

    st.markdown("### Patient-Level Results")
    st.dataframe(results, use_container_width=True)

    st.markdown("### Distribution Views")
    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.pyplot(plot_profile_distribution(results), use_container_width=True)

    with col4:
        st.pyplot(plot_risk_group_distribution(results), use_container_width=True)

    st.markdown("### Summary Interpretation")

    with st.expander("Risk Profile Summary", expanded=True):
        summary = build_profile_summary(results)
        st.dataframe(summary, use_container_width=True)

    with st.expander("Clinical Profile Definitions", expanded=False):
        for cid in CLUSTER_ORDER:
            info = CLUSTER_INFO[cid]
            st.markdown(
                f"**{info['name']}**  \n"
                f"- Risk group: {info['risk']}  \n"
                f"- Interpretation: {info['summary']}  \n"
                f"- Main indicators: {', '.join(info['key_features'])}"
            )

    with st.expander("Uploaded Data Preview", expanded=False):
        st.dataframe(df_clean.head(200), use_container_width=True)

    st.download_button(
        "Download Patient-Level Results",
        results.to_csv(index=False),
        file_name="tb_risk_results.csv",
        mime="text/csv",
    )





# ============================================================
# SYNTHETIC DECODER (FIXED ✅)
# ============================================================
def decode_synthetic_from_transformed(syn_df: pd.DataFrame):
    """
    Properly decode synthetic samples from transformed space into
    human-readable clinical-like values.
    """

    decoded = pd.DataFrame(index=syn_df.index)

    def rescale(series, max_val, min_val=0):
        return (series.clip(0, 1) * (max_val - min_val) + min_val).round().astype(int)

    # Explicit continuous schema (CRITICAL)
    CONT_SCHEMA = {
        "age_census": (100, 0),
        "cough_d": (30, 0),
        "fever_d": (30, 0),
        "wloss_d": (365, 0),
        "sputum_d": (30, 0),
        "tbhist_y": (35, 1990),
        "tbtreat_w": (52, 0),
    }

    # Continuous
    for col, (mx, mn) in CONT_SCHEMA.items():
        tcol = f"cont__{col}"
        if tcol in syn_df.columns:
            decoded[col] = rescale(syn_df[tcol], mx, mn)

    # Binary
    for col in BINARY_COLS:
        tcol = f"bin__{col}"
        if tcol in syn_df.columns:
            decoded[col] = (syn_df[tcol] >= 0.5).astype(int)

    # Categorical
    for col in CATEGORICAL_COLS:
        prefix = f"cat__{col}_"
        matches = [c for c in syn_df.columns if c.startswith(prefix)]
        if matches:
            vals = (
                syn_df[matches]
                .idxmax(axis=1)
                .str.replace(prefix, "", regex=False)
                .str.replace(".0", "", regex=False)
            )
            decoded[col] = vals

    return decoded




# ============================================================
# SYNTHETIC DATA GENERATION (STYLED)
# ============================================================

st.markdown(
    textwrap.dedent("""
    <div style="
        background: linear-gradient(180deg, rgba(13,20,38,0.96), rgba(8,14,28,0.98));
        border: 1px solid rgba(114, 137, 218, 0.20);
        border-radius: 24px;
        padding: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="
                width:52px; height:52px;
                border-radius:16px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:22px;
                background: linear-gradient(135deg, rgba(45,212,191,0.25), rgba(124,77,255,0.25));
                border: 1px solid rgba(124,77,255,0.25);
            ">
                👥
            </div>
            <div>
                <div style="font-size:22px; font-weight:800;">
                    Synthetic Patient Generation
                </div>
                <div style="color:#a8b2d1; font-size:14px;">
                    Generate realistic synthetic TB patient profiles for testing and analysis
                </div>
            </div>
        </div>
    """),
    unsafe_allow_html=True,
)

# Layout inside card
col1, col2 = st.columns([1.2, 1])

with col1:
    num_samples = st.slider(
        "Number of synthetic patients",
        10, 200, 50
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_clicked = st.button("✨ Generate Synthetic Patients")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# GENERATION LOGIC (UNCHANGED)
# ============================================================

if generate_clicked:
    z = torch.randn(num_samples, LATENT_DIM)

    with torch.no_grad():
        syn_array = model.decode(z).cpu().numpy()

    syn_df = pd.DataFrame(syn_array, columns=feature_names)
    decoded = decode_synthetic_from_transformed(syn_df)

    st.success(f"Generated {num_samples} synthetic patient records.")

    st.markdown("### Preview of Generated Data")
    st.dataframe(decoded.head(10), use_container_width=True)

    st.download_button(
        "⬇ Download Synthetic Dataset",
        decoded.to_csv(index=False),
        file_name="synthetic_tb_patients.csv",
        mime="text/csv",
    )

# ============================================================
# FOOTER NOTE
# ============================================================

st.markdown(
    textwrap.dedent("""
    <div style="
        margin-top: 1rem;
        padding: 0.9rem;
        border-radius: 16px;
        background: rgba(8, 13, 26, 0.6);
        border: 1px solid rgba(114, 137, 218, 0.18);
        color: #a8b2d1;
        font-size: 14px;
    ">
        ⚠️ Synthetic records are statistically plausible but not real patients and must not be used directly for clinical decision-making.
    </div>
    """),
    unsafe_allow_html=True,
)
