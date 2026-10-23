import os
import glob
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)

st.set_page_config(
    page_title="Predictive Analysis of Child Malnutrition",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
    --brand: #0D9488;
    --brand-soft: #CCFBF1;
    --accent: #EA580C;
    --accent-soft: #FFEDD5;
    --success: #10B981;
    --success-soft: #D1FAE5;
    --ink: #1E293B;
    --muted: #64748B;
    --line: #E2E8F0;
    --card: #FFFFFF;
    --bg: #F8FAFC;
}

html, body, [class*="css"] {
    font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--bg);
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3, h4 {
    color: var(--ink);
    letter-spacing: -0.01em;
}

h2 {
    font-size: 1.35rem;
    margin-top: 1.8rem;
    margin-bottom: 1rem;
    padding-left: 0.75rem;
    border-left: 4px solid var(--brand);
    border-bottom: none;
    font-weight: 600;
}

h3 {
    font-size: 1.05rem;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.app-header {
    padding: 1.6rem 2rem;
    background: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%);
    color: #ffffff;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(13, 148, 136, 0.25);
    position: relative;
    overflow: hidden;
}
.app-header::after {
    content: "";
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.1);
    border-radius: 50%;
    pointer-events: none;
}
.app-header .title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
    position: relative;
    z-index: 1;
}
.app-header .subtitle {
    font-size: 1rem;
    opacity: 0.9;
    margin-top: 0.4rem;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

.insight {
    background: var(--brand-soft);
    border-left: 4px solid var(--brand);
    padding: 0.8rem 1.2rem;
    border-radius: 4px 8px 8px 4px;
    font-size: 0.9rem;
    color: var(--ink);
    margin: 1rem 0 1.5rem 0;
}

[data-testid="stSidebar"] {
    background: var(--card);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h2 {
    border-left: none;
    padding-left: 0;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: none;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--muted);
    margin-bottom: 0;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #E2E8F0;
    color: var(--ink);
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: var(--brand) !important;
    border: 1px solid #E2E8F0;
    border-bottom: 2px solid var(--brand) !important;
    box-shadow: 0 -2px 6px rgba(0,0,0,0.02);
}

[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--line);
}

div[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 10px;
    background: var(--card);
}

.stButton>button, .stDownloadButton>button {
    background: var(--brand);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: background 0.2s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: #0F766E;
    color: #fff;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid var(--line);
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.metric-card h4 { margin: 0; color: var(--muted); font-size: 1rem; text-transform: uppercase; }
.metric-card h2 { font-size: 2.5rem; margin: 0.5rem 0 0 0; color: var(--brand); border-left: none; padding-left: 0;}

.app-footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.78rem;
    padding: 2rem 0 0.5rem 0;
    border-top: 1px solid var(--line);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": "#64748B",
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "axes.titlecolor": "#1E293B",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#64748B",
    "ytick.color": "#64748B",
    "font.size": 9.5,
    "figure.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
})

PALETTE = {
    "brand": "#0D9488",
    "brand_light": "#14B8A6",
    "accent": "#EA580C",
    "accent_soft": "#FB923C",
    "success": "#10B981",
    "success_soft": "#34D399",
    "warning": "#F59E0B",
    "neutral": "#94A3B8",
    "series": ["#0D9488", "#14B8A6", "#5EEAD4", "#99F6E4"],
}

CONFIG = {
    "file_hints": ("nutritional", "rhu", "opt"),
    "sheet_includes": ("opt", "202"),
    "search_paths": (".", "/content/data", "./data"),
    "age_min": 0,
    "age_max": 59,
    "malnourished_codes": {"UW", "SUW", "ST", "SST", "MW", "SW"},
    "missing_tokens": {"", "-", "--", "—", "N/A", "NA", "NAN", "NONE", "NULL",
                       "?", ".", "N.D.", "ND"},
    "test_size": 0.20,
    "random_state": 42,
    "age_bins": [0, 6, 12, 24, 36, 48, 60],
    "age_labels": ["0-5", "6-11", "12-23", "24-35", "36-47", "48-59"],
}

BARANGAY_COORDS = {
    "MAMBALILI": (8.06, 126.05),
    "BUNAWAN BROOK": (8.15, 126.05),
    "SAN TEODORO": (8.18, 126.00),
    "POBLACION": (8.18, 126.00),
    "CONSUELO": (8.20, 125.95),
    "SAN ANDRES": (8.10, 125.98),
    "IMELDA": (8.12, 126.08),
    "NUEVA ERA": (8.14, 126.10),
    "SAN MARCOS": (8.08, 126.02),
    "LIBERTAD": (8.05, 125.95),
    "HAGNAYA": (8.16, 125.92),
    "BABADAN": (8.22, 126.05),
}

ZSCORE_MAP = {
    "N": 0, "OW": 2, "OB": 3, "T": 2,
    "UW": -2, "SUW": -3, "ST": -2, "SST": -3, "MW": -2, "SW": -3
}

def strip_display(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = re.sub(r'[\U00010000-\U0010ffff]', '', str(value))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_columns(df):
    df = df.copy()
    cleaned = []
    for c in df.columns:
        name = strip_display(c)
        if not name: name = "col"
        cleaned.append(name)
    seen = {}
    out = []
    for name in cleaned:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    df.columns = out
    return df

def find_col(df, *candidates):
    normalized = {}
    for c in df.columns:
        key = re.sub(r'[\s_\-()]', '', str(c)).lower()
        normalized[key] = c
    for cand in candidates:
        key = re.sub(r'[\s_\-()]', '', str(cand)).lower()
        if key in normalized:
            return normalized[key]
    return None

def process_excel_source(source, is_uploaded=False):
    try:
        book = pd.ExcelFile(source)
        collected = []
        for sheet in book.sheet_names:
            if not any(k in sheet.lower() for k in CONFIG["sheet_includes"]):
                continue
            raw = pd.read_excel(book, sheet_name=sheet, header=None)
            header_row = None
            for i in range(min(15, len(raw))):
                row_values = [strip_display(v).upper() for v in raw.iloc[i].tolist()]
                if any(v in ("NO.", "NO") for v in row_values):
                    header_row = i
                    break
            if header_row is None: header_row = 3
            cols = []
            for j, c in enumerate(raw.iloc[header_row].tolist()):
                name = strip_display(c).replace("\n", " ")
                if not name: name = f"col_{j}"
                cols.append(name)
            body = raw.iloc[header_row + 1:].copy()
            body.columns = cols
            body = body.reset_index(drop=True)
            body["_source_sheet"] = strip_display(sheet)
            collected.append(body)
        if not collected: return None, None
        filename = source.name if is_uploaded else os.path.basename(source)
        return pd.concat(collected, ignore_index=True, sort=False), filename
    except Exception as exc:
        st.error("could not read workbook: " + strip_display(exc))
        return None, None

def load_data(uploaded_file, use_example):
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8-sig", low_memory=False)
                df["_source_sheet"] = "uploaded_csv"
                return df, uploaded_file.name
            except Exception as e:
                st.error(f"Error reading CSV: {e}"); return None, None
        else:
            return process_excel_source(uploaded_file, is_uploaded=True)
    elif use_example:
        files = []
        for path in CONFIG["search_paths"]:
            if os.path.isdir(path):
                files.extend(glob.glob(os.path.join(path, "*.xlsx")))
                files.extend(glob.glob(os.path.join(path, "*.xls")))
                files.extend(glob.glob(os.path.join(path, "*.csv")))
        if not files: return None, None
        preferred = [f for f in files if any(k in os.path.basename(f).lower() for k in CONFIG["file_hints"])]
        target = preferred[0] if preferred else files[0]
        if target.lower().endswith(".csv"):
            try:
                df = pd.read_csv(target, encoding="utf-8-sig", low_memory=False)
                df["_source_sheet"] = "unknown"
                return df, os.path.basename(target)
            except Exception as e:
                st.error(f"Error reading example CSV: {e}"); return None, None
        else:
            return process_excel_source(target, is_uploaded=False)
    return None, None

def normalize_text_columns(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: strip_display(v) if not pd.isna(v) else v)
            df[col] = df[col].replace(list(CONFIG["missing_tokens"]), np.nan)
    return df

def drop_non_records(df):
    no_col = find_col(df, "NO.", "NO", "No")
    if no_col is None: return df
    numeric = pd.to_numeric(df[no_col], errors="coerce")
    return df[numeric.notna()].copy()

def drop_duplicates(df):
    meta = [c for c in df.columns if str(c).startswith("_")]
    check_cols = [c for c in df.columns if c not in meta]
    return df.drop_duplicates(subset=check_cols, keep="first").copy()

def prepare_columns(df):
    df = df.copy()
    cols = {
        "sex": find_col(df, "SEX", "Sex", "gender"),
        "age": find_col(df, "AGE (Months)", "AGE (months)", "Age", "age_months"),
        "weight": find_col(df, "WEIGHT (kg)", "Weight (kg)", "weight_kg"),
        "height": find_col(df, "HEIGHT (cm)", "Height (cm)", "height_cm"),
        "barangay": find_col(df, "BARANGAY", "Barangay", "brgy"),
        "waz": find_col(df, "WEIGHT-FOR-AGE (WAZ)", "WAZ"),
        "haz": find_col(df, "HEIGHT-FOR-AGE (HAZ)", "HAZ"),
        "whz": find_col(df, "WEIGHT-FOR-HEIGHT (WHZ)", "WHZ"),
        "sheet": "_source_sheet",
    }
    for key in ("age", "weight", "height"):
        if cols[key]: df[cols[key]] = pd.to_numeric(df[cols[key]], errors="coerce")
    for key in ("sex", "barangay", "waz", "haz", "whz"):
        if cols[key]:
            df[cols[key]] = df[cols[key]].apply(lambda v: v.strip().upper() if isinstance(v, str) else v)
    return df, cols

def impute_missing(df, cols):
    df = df.copy()
    for key in ("age", "weight", "height"):
        col = cols[key]
        if col and df[col].isna().any(): df[col] = df[col].fillna(df[col].median())
    for key in ("sex", "barangay"):
        col = cols[key]
        if col and df[col].isna().any():
            mode_vals = df[col].mode()
            fill = mode_vals.iloc[0] if not mode_vals.empty else "UNKNOWN"
            df[col] = df[col].fillna(fill)
    return df

def build_target(df, cols):
    waz, haz, whz = cols["waz"], cols["haz"], cols["whz"]
    if not all([waz, haz, whz]): return None
    codes = CONFIG["malnourished_codes"]
    def row_malnourished(r):
        for c in (waz, haz, whz):
            v = r[c]
            if isinstance(v, str) and v in codes: return 1
        return 0
    return df.apply(row_malnourished, axis=1).astype(int)

def build_features(df, cols):
    features = pd.DataFrame(index=df.index)
    if cols["sex"]: features["sex"] = df[cols["sex"]].astype(str)
    if cols["age"]: features["age_months"] = pd.to_numeric(df[cols["age"]], errors="coerce")
    if cols["weight"]: features["weight_kg"] = pd.to_numeric(df[cols["weight"]], errors="coerce")
    if cols["height"]: features["height_cm"] = pd.to_numeric(df[cols["height"]], errors="coerce")
    if cols["barangay"]: features["barangay"] = df[cols["barangay"]].astype(str)
    return features

def encode_features(features):
    encoded = features.copy()
    encoders = {}
    for col in ("sex", "barangay"):
        if col in encoded.columns:
            encoder = LabelEncoder()
            encoded[col] = encoder.fit_transform(encoded[col].astype(str))
            encoders[col] = encoder
    return encoded, encoders

def train_all_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"],
        stratify=y if y.nunique() > 1 else None
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    candidates = {
        "Decision Tree": DecisionTreeClassifier(random_state=CONFIG["random_state"]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=CONFIG["random_state"]),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=CONFIG["random_state"]),
    }
    trained = {}
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
        trained[name] = {"model": model, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba, "X_test": X_test_scaled}
    return trained, scaler

def metric_row(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }

def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Data Source")
    uploaded_file = st.file_uploader("Upload OPT dataset", type=["csv", "xlsx", "xls"])
    use_example = st.checkbox("Use example dataset (if no upload)", value=False)

raw_df, loaded_file = load_data(uploaded_file, use_example)

if raw_df is None:
    st.info("Awaiting data source. Please upload an OPT dataset file or check the 'Use example dataset' box in the sidebar.")
    st.stop()

df = clean_columns(raw_df)
df = normalize_text_columns(df)
df = drop_non_records(df)
df = drop_duplicates(df)
df, col_map = prepare_columns(df)
df = df[df[col_map["age"]].notna() & (df[col_map["age"]] >= CONFIG["age_min"]) & (df[col_map["age"]] <= CONFIG["age_max"])].copy()
df = impute_missing(df, col_map)

target = build_target(df, col_map)
if target is None:
    st.error("required indicator columns (WAZ, HAZ, WHZ) were not detected"); st.stop()

df = df.loc[target.index].copy()
df["malnourished"] = target.values

features = build_features(df, col_map)
valid_mask = features.notna().all(axis=1)
features = features[valid_mask].reset_index(drop=True)
labels = df.loc[valid_mask, "malnourished"].reset_index(drop=True)
df_model = df.loc[valid_mask].reset_index(drop=True)

encoded, encoders = encode_features(features)
trained, scaler = train_all_models(encoded, labels)
metrics = pd.DataFrame({name: metric_row(r["y_test"], r["y_pred"]) for name, r in trained.items()}).T
metrics["F1+Recall"] = metrics["F1-Score"] + metrics["Recall"]
best_combined = metrics["F1+Recall"].idxmax()

st.markdown(
    '<div class="app-header">'
    '<div class="title">Predictive Analysis of Child Malnutrition in Bunawan, Agusan del Sur</div>'
    '<div class="subtitle">Data-driven insights and predictive modeling using Operation Timbang records</div>'
    '</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("## Filters")
    if col_map["barangay"]:
        brgy_options = ["All Barangays"] + sorted(df_model[col_map["barangay"]].dropna().unique().tolist())
        focus_brgy = st.selectbox("Focus Barangay", options=brgy_options)
    else:
        focus_brgy = "All Barangays"
        
    if col_map["sex"]:
        sex_options = sorted(df_model[col_map["sex"]].dropna().unique().tolist())
        selected_sex = st.multiselect("Sex", options=sex_options, default=sex_options)
    else:
        selected_sex = None

filtered = df_model.copy()
if focus_brgy != "All Barangays" and col_map["barangay"]:
    filtered = filtered[filtered[col_map["barangay"]] == focus_brgy]
if selected_sex is not None and col_map["sex"]:
    filtered = filtered[filtered[col_map["sex"]].isin(selected_sex)]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Demographic Profile",
    "2. Nutritional Status (WAZ/HAZ/WHZ)",
    "3. Patterns & Key Factors",
    "4. Model Evaluation",
    "5. Decision Support"
])

with tab1:
    st.markdown("## Analyze the demographic profile of the children in Bunawan")
    if focus_brgy != "All Barangays":
        st.info(f"Currently viewing data for **{focus_brgy}**. Select 'All Barangays' in the sidebar to view the whole municipality.")
    
    st.markdown("### Geographic Distribution (Map)")
    map_data = df_model.groupby(col_map["barangay"]).agg(
        Total=("malnourished", "count"),
        Malnourished=("malnourished", "sum")
    ).reset_index()
    map_data["Malnutrition Rate (%)"] = (map_data["Malnourished"] / map_data["Total"] * 100).round(2)
    map_data["lat"] = map_data[col_map["barangay"]].map(lambda x: BARANGAY_COORDS.get(x, (8.1, 126.0))[0])
    map_data["lon"] = map_data[col_map["barangay"]].map(lambda x: BARANGAY_COORDS.get(x, (8.1, 126.0))[1])
    
    fig_map = px.scatter_map(
        map_data, lat="lat", lon="lon", size="Total", color="Malnutrition Rate (%)",
        hover_name=col_map["barangay"], color_continuous_scale="Reds",
        size_max=30, zoom=10, height=400, title="Malnutrition Rate Across Barangays (Darker Red = Higher Rate)"
    )
    fig_map.update_layout(map_style="open-street-map", margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Malnourishment by Barangay")
        brgy_summary = filtered.groupby(col_map["barangay"])["malnourished"].agg(Total="count", Malnourished="sum").reset_index()
        brgy_summary["Rate"] = (brgy_summary["Malnourished"] / brgy_summary["Total"] * 100).round(2)
        fig_brgy = px.bar(brgy_summary.sort_values("Rate", ascending=True), x="Rate", y=col_map["barangay"], orientation="h", color="Rate", color_continuous_scale="Reds", text="Rate")
        fig_brgy.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_brgy.update_layout(showlegend=False, height=400, xaxis_title="Malnutrition Rate (%)", yaxis_title="")
        st.plotly_chart(fig_brgy, use_container_width=True)

    with col_b:
        st.markdown("### Sex Distribution by Barangay (Stacked)")
        sex_brgy = pd.crosstab(filtered[col_map["barangay"]], [filtered[col_map["sex"]], filtered["malnourished"]])
        sex_brgy.columns = [f"{c[0]} ({'Mal' if c[1]==1 else 'Not Mal'})" for c in sex_brgy.columns]
        fig_sex = go.Figure()
        for col in sex_brgy.columns:
            color = PALETTE["accent"] if "Mal)" in col else PALETTE["success"]
            if "Male" in col: color = "#0D9488" if "Not Mal" in col else "#EA580C"
            elif "Female" in col: color = "#14B8A6" if "Not Mal" in col else "#FB923C"
            fig_sex.add_trace(go.Bar(name=col, x=sex_brgy.index, y=sex_brgy[col], marker_color=color))
        fig_sex.update_layout(barmode='stack', height=400, xaxis_title="", yaxis_title="Number of Children")
        st.plotly_chart(fig_sex, use_container_width=True)
        
        if focus_brgy != "All Barangays" and col_map["sex"]:
            sex_counts = filtered.groupby(col_map["sex"])["malnourished"].sum()
            if not sex_counts.empty:
                worst_sex = sex_counts.idxmax()
                insight(f"Analysis for <b>{focus_brgy}</b>: The <b>{worst_sex}</b> demographic has the highest number of malnourished children ({sex_counts.max()} cases), indicating a need for targeted gender-sensitive interventions in this area.")

    st.markdown("### Age Group Distribution")
    work_age = filtered.copy()
    work_age["age_group"] = pd.cut(work_age[col_map["age"]], bins=CONFIG["age_bins"], labels=CONFIG["age_labels"], include_lowest=True, right=False)
    age_summary = work_age.groupby("age_group", observed=True)["malnourished"].agg(Total="count", Malnourished="sum").reset_index()
    age_summary["Rate"] = (age_summary["Malnourished"] / age_summary["Total"] * 100).round(2)
    
    fig_age = px.bar(age_summary, x="age_group", y="Malnourished", color="Rate", color_continuous_scale="Reds", text="Rate", title="Malnourished Children by Age Group (Months)")
    fig_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_age, use_container_width=True)
    
    st.markdown("#### Feeding Program Summary Report (Tabular)")
    st.caption("This table provides the exact number of malnourished children per age group for the selected barangay, useful for planning feeding programs.")
    feed_table = filtered.groupby([col_map["barangay"], "age_group" if "age_group" in filtered.columns else pd.cut(filtered[col_map["age"]], bins=CONFIG["age_bins"], labels=CONFIG["age_labels"], include_lowest=True, right=False)], observed=True)["malnourished"].sum().reset_index()
    feed_table.columns = ["Barangay", "Age Group", "Malnourished Count"]
    st.dataframe(feed_table.pivot(index="Barangay", columns="Age Group", values="Malnourished Count").fillna(0).astype(int), use_container_width=True)

    st.markdown("### Variable Relationships (Scatter Plots)")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        fig_s1 = px.scatter(filtered, x=col_map["age"], y=col_map["weight"], color=filtered["malnourished"].astype(str), 
                            labels={"color": "Malnourished", col_map["age"]: "Age (Months)", col_map["weight"]: "Weight (kg)"},
                            color_discrete_map={"0": PALETTE["success"], "1": PALETTE["accent"]}, title="Age vs Weight")
        st.plotly_chart(fig_s1, use_container_width=True)
    with sc2:
        fig_s2 = px.scatter(filtered, x=col_map["age"], y=col_map["height"], color=filtered["malnourished"].astype(str), 
                            labels={"color": "Malnourished", col_map["age"]: "Age (Months)", col_map["height"]: "Height (cm)"},
                            color_discrete_map={"0": PALETTE["success"], "1": PALETTE["accent"]}, title="Age vs Height")
        st.plotly_chart(fig_s2, use_container_width=True)
    with sc3:
        fig_s3 = px.scatter(filtered, x=col_map["height"], y=col_map["weight"], color=filtered["malnourished"].astype(str), 
                            labels={"color": "Malnourished", col_map["height"]: "Height (cm)", col_map["weight"]: "Weight (kg)"},
                            color_discrete_map={"0": PALETTE["success"], "1": PALETTE["accent"]}, title="Height vs Weight")
        st.plotly_chart(fig_s3, use_container_width=True)
        
    insight("Look closely at the Height vs Weight scatter plot. Children clustering in the bottom-left quadrant are both short and light for their age, indicating severe stunting and wasting (acute and chronic malnutrition).")

with tab2:
    st.markdown("## Determine the nutritional status based on standard indicators")
    st.write("Visualizing the distribution of WHO 2006 Child Growth Standard classifications across the selected barangays.")
    
    status_cols = [col_map["waz"], col_map["haz"], col_map["whz"]]
    status_names = ["Weight-for-Age (WAZ)", "Height-for-Age (HAZ)", "Weight-for-Height (WHZ)"]
    
    for i, col in enumerate(status_cols):
        st.markdown(f"### {status_names[i]}")
        
        if focus_brgy == "All Barangays":
            stat_df = filtered.groupby([col_map["barangay"], col]).size().reset_index(name="Count")
            fig_stat = px.bar(stat_df, x=col_map["barangay"], y="Count", color=col, barmode="stack",
                              title=f"{status_names[i]} Distribution by Barangay",
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_stat, use_container_width=True)
        else:
            stat_df = filtered[col].value_counts().reset_index()
            stat_df.columns = ["Status", "Count"]
            fig_pie = px.pie(stat_df, names="Status", values="Count", hole=0.4, 
                             title=f"{status_names[i]} Distribution in {focus_brgy}",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with st.expander(f"View raw counts for {status_names[i]}"):
            st.dataframe(filtered.groupby([col_map["barangay"], col]).size().unstack(fill_value=0), use_container_width=True)

with tab3:
    st.markdown("## Identify patterns and key factors associated with child malnutrition")
    
    st.markdown("### Correlation Heatmap (Age, Weight, Height, WAZ, HAZ, WHZ)")
    st.caption("WHO z-score categorical values (e.g., 'UW', 'St') are mapped to approximate numeric scores to calculate correlation.")
    corr_df = filtered[[col_map["age"], col_map["weight"], col_map["height"], col_map["waz"], col_map["haz"], col_map["whz"]]].copy()
    corr_df.columns = ["Age", "Weight", "Height", "WAZ", "HAZ", "WHZ"]
    
    for z_col in ["WAZ", "HAZ", "WHZ"]:
        corr_df[z_col] = corr_df[z_col].map(ZSCORE_MAP).fillna(0)
        
    fig_corr, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_df.corr(), annot=True, cmap="RdBu_r", center=0, fmt=".2f", ax=ax)
    st.pyplot(fig_corr); plt.close()

    st.markdown("### K-Means Clustering (Grouping Similar Children)")
    st.write("Identifying clusters based on Age, Weight, and Height to find distinct nutritional profiles.")
    cluster_df = filtered[[col_map["age"], col_map["weight"], col_map["height"]]].dropna().copy()
    cluster_df.columns = ["Age", "Weight", "Height"]
    
    if len(cluster_df) > 5:
        X_cluster = StandardScaler().fit_transform(cluster_df)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_cluster)
        cluster_df["Cluster"] = kmeans.labels_.astype(str)
        
        fig_cluster = px.scatter(cluster_df, x="Age", y="Weight", color="Cluster", 
                                 title="K-Means Clustering: Age vs Weight (3 Clusters)",
                                 color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        with st.expander("Cluster Characteristics (Mean values)"):
            st.dataframe(cluster_df.groupby("Cluster").mean().round(2))
    else:
        st.warning("Not enough data to perform clustering.")

    st.markdown("### Feature Importance (Random Forest)")
    st.write("Ranking variables based on how much they contribute to predicting malnutrition.")
    rf_model = trained["Random Forest"]["model"]
    imp_df = pd.DataFrame({"Feature": encoded.columns, "Importance": rf_model.feature_importances_})
    imp_df["Importance (%)"] = (imp_df["Importance"] / imp_df["Importance"].sum() * 100).round(2)
    imp_df = imp_df.sort_values("Importance (%)", ascending=True)
    
    fig_imp = px.bar(imp_df, x="Importance (%)", y="Feature", orientation="h", 
                     title="Random Forest Feature Importance (Ranked by Percentage)",
                     color="Importance (%)", color_continuous_scale="Teal")
    fig_imp.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    st.plotly_chart(fig_imp, use_container_width=True)

with tab4:
    st.markdown("## Evaluate the predictive model that classify malnutrition risk")
    st.markdown("### Test-Set Performance Metrics")
    
    cols = st.columns(4)
    metrics_to_show = ["Accuracy", "Precision", "Recall", "F1-Score"]
    for i, metric in enumerate(metrics_to_show):
        best_val = metrics[metric].max()
        best_model = metrics[metric].idxmax()
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{metric}</h4>
                <h2>{best_val:.2%}</h2>
                <p style="color: #64748B; margin: 0;">Best: {best_model}</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("### Detailed Algorithm Comparison")
    st.dataframe(metrics.drop(columns="F1+Recall").style.highlight_max(axis=0, color="#CCFBF1").format("{:.4f}"), use_container_width=True)

    st.markdown("### ROC-AUC Curves")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
    for name, r in trained.items():
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc); plt.close()

    st.markdown("### Confusion Matrices")
    pair_cols = st.columns(2)
    for i, (name, r) in enumerate(trained.items()):
        with pair_cols[i % 2]:
            cm = confusion_matrix(r["y_test"], r["y_pred"], labels=[0, 1])
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", xticklabels=["Not Mal.", "Mal."], yticklabels=["Not Mal.", "Mal."], ax=ax_cm, cbar=False)
            ax_cm.set_title(name)
            ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm); plt.close()

    st.markdown("### Why these four algorithms?")
    insight("""
    <b>Decision Tree</b> and <b>Random Forest</b> were selected for their ability to handle non-linear relationships and provide interpretable feature importance, which is crucial for public health policy. 
    <b>Naive Bayes</b> serves as a highly efficient probabilistic baseline, particularly robust on small datasets. 
    <b>Logistic Regression</b> was chosen as the standard statistical baseline for binary classification, allowing us to compare complex tree-based models against a traditional linear model.
    """)

with tab5:
    st.markdown("## Decision Support: Predicted Risk per Barangay")
    best_model = trained[best_combined]["model"]
    full_matrix = scaler.transform(encoded)
    full_proba = best_model.predict_proba(full_matrix)[:, 1] if hasattr(best_model, "predict_proba") else best_model.predict(full_matrix).astype(float)
    full_pred = (full_proba >= 0.5).astype(int)

    scored = df_model.copy()
    scored["risk_probability"] = full_proba
    scored["predicted_label"] = np.where(full_pred == 1, "At Risk", "Not At Risk")

    if col_map["barangay"]:
        risk_summary = scored.groupby(col_map["barangay"])["predicted_label"].value_counts().unstack(fill_value=0)
        if "At Risk" not in risk_summary.columns: risk_summary["At Risk"] = 0
        if "Not At Risk" not in risk_summary.columns: risk_summary["Not At Risk"] = 0
        risk_summary["Total"] = risk_summary.sum(axis=1)
        risk_summary["Mean Prob"] = scored.groupby(col_map["barangay"])["risk_probability"].mean().round(4)
        risk_summary = risk_summary.sort_values("Mean Prob", ascending=False)
        
        fig_risk = px.bar(risk_summary.reset_index(), x="Mean Prob", y=col_map["barangay"], orientation="h",
                          title="Mean Predicted Risk Probability by Barangay",
                          color="Mean Prob", color_continuous_scale="Oranges")
        st.plotly_chart(fig_risk, use_container_width=True)
        
        st.dataframe(risk_summary, use_container_width=True)

    st.markdown("### Download Predictions")
    export_cols = [col_map["barangay"], col_map["sex"], col_map["age"], "malnourished", "risk_probability", "predicted_label"]
    export = scored[export_cols].copy()
    st.download_button(
        "Download per-child predictions CSV",
        export.to_csv(index=False).encode("utf-8"),
        "per_child_predictions.csv",
        "text/csv"
    )
