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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

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

.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.2rem;
    margin-bottom: 1.8rem;
}
.kpi-card {
    background: var(--card);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border: 1px solid var(--line);
    box-shadow: 0 2px 8px rgba(15, 30, 45, 0.03);
    transition: all 0.2s ease;
}
.kpi-card:hover {
    box-shadow: 0 6px 16px rgba(15, 30, 45, 0.08);
    transform: translateY(-2px);
}
.kpi-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.kpi-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--brand);
    line-height: 1.1;
}
.kpi-card .value.accent { color: var(--accent); }
.kpi-card .value.success { color: var(--success); }
.kpi-card .caption {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.4rem;
}

.pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.pill-good { background: var(--success-soft); color: var(--success); }
.pill-bad { background: var(--accent-soft); color: var(--accent); }
.pill-neutral { background: var(--brand-soft); color: var(--brand); }

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

[data-testid="stMetricValue"] {
    color: var(--brand);
    font-weight: 700;
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
    "non_malnourished_codes": {"N", "OW", "OB", "T"},
    "missing_tokens": {"", "-", "--", "—", "N/A", "NA", "NAN", "NONE", "NULL",
                       "?", ".", "N.D.", "ND"},
    "test_size": 0.20,
    "random_state": 42,
    "cv_folds": 5,
    "age_bins": [0, 6, 12, 24, 36, 48, 60],
    "age_labels": ["0-5", "6-11", "12-23", "24-35", "36-47", "48-59"],
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
        if not name:
            name = "col"
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


def read_all_sheets():
    files = []
    for path in CONFIG["search_paths"]:
        if os.path.isdir(path):
            files.extend(glob.glob(os.path.join(path, "*.xlsx")))
            files.extend(glob.glob(os.path.join(path, "*.xls")))
            files.extend(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        return None, None
    preferred = [f for f in files
                 if any(k in os.path.basename(f).lower() for k in CONFIG["file_hints"])]
    target = preferred[0] if preferred else files[0]
    try:
        if target.lower().endswith(".csv"):
            df = pd.read_csv(target, encoding="utf-8-sig", low_memory=False)
            df["_source_sheet"] = "unknown"
            return df, os.path.basename(target)
        book = pd.ExcelFile(target)
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
            if header_row is None:
                header_row = 3
            cols = []
            for j, c in enumerate(raw.iloc[header_row].tolist()):
                name = strip_display(c).replace("\n", " ")
                if not name:
                    name = f"col_{j}"
                cols.append(name)
            body = raw.iloc[header_row + 1:].copy()
            body.columns = cols
            body = body.reset_index(drop=True)
            body["_source_sheet"] = strip_display(sheet)
            collected.append(body)
        if not collected:
            return None, None
        return pd.concat(collected, ignore_index=True, sort=False), os.path.basename(target)
    except Exception as exc:
        st.error("could not read workbook: " + strip_display(exc))
        return None, None


def normalize_text_columns(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: strip_display(v) if not pd.isna(v) else v)
            df[col] = df[col].replace(list(CONFIG["missing_tokens"]), np.nan)
            df[col] = df[col].replace({"nan": np.nan, "NaN": np.nan, "None": np.nan})
    return df


def drop_non_records(df):
    no_col = find_col(df, "NO.", "NO", "No")
    if no_col is None:
        return df
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
        "purok": find_col(df, "PUROK / ADDRESS", "PUROK", "Purok", "ADDRESS"),
        "date": find_col(df, "DATE MEASURED", "Date Measured"),
        "birthday": find_col(df, "BIRTHDAY", "Birthday"),
        "waz": find_col(df, "WEIGHT-FOR-AGE (WAZ)", "WAZ"),
        "haz": find_col(df, "HEIGHT-FOR-AGE (HAZ)", "HAZ"),
        "whz": find_col(df, "WEIGHT-FOR-HEIGHT (WHZ)", "WHZ"),
        "sheet": "_source_sheet",
    }
    for key in ("age", "weight", "height"):
        if cols[key]:
            df[cols[key]] = pd.to_numeric(df[cols[key]], errors="coerce")
    for key in ("sex", "barangay", "waz", "haz", "whz"):
        if cols[key]:
            df[cols[key]] = df[cols[key]].apply(
                lambda v: v.strip().upper() if isinstance(v, str) else v
            )
    return df, cols


def drop_missing_anthropometry(df, cols):
    w, h = cols["weight"], cols["height"]
    if w and h:
        return df[~(df[w].isna() & df[h].isna())].copy()
    return df


def drop_missing_status(df, cols):
    waz, haz, whz = cols["waz"], cols["haz"], cols["whz"]
    if waz and haz and whz:
        return df[~(df[waz].isna() & df[haz].isna() & df[whz].isna())].copy()
    return df


def drop_out_of_scope_age(df, cols):
    age = cols["age"]
    if age:
        df = df[df[age].notna()].copy()
        df = df[(df[age] >= CONFIG["age_min"]) & (df[age] <= CONFIG["age_max"])].copy()
    return df


def impute_missing(df, cols):
    df = df.copy()
    for key in ("age", "weight", "height"):
        col = cols[key]
        if col and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for key in ("sex", "barangay"):
        col = cols[key]
        if col and df[col].isna().any():
            mode_vals = df[col].mode()
            fill = mode_vals.iloc[0] if not mode_vals.empty else "UNKNOWN"
            df[col] = df[col].fillna(fill)
    return df


def build_target(df, cols):
    waz, haz, whz = cols["waz"], cols["haz"], cols["whz"]
    if not all([waz, haz, whz]):
        return None
    codes = CONFIG["malnourished_codes"]
    def row_malnourished(r):
        for c in (waz, haz, whz):
            v = r[c]
            if isinstance(v, str) and v in codes:
                return 1
        return 0
    return df.apply(row_malnourished, axis=1).astype(int)


def build_features(df, cols):
    features = pd.DataFrame(index=df.index)
    if cols["sex"]:
        features["sex"] = df[cols["sex"]].astype(str)
    if cols["age"]:
        features["age_months"] = pd.to_numeric(df[cols["age"]], errors="coerce")
    if cols["weight"]:
        features["weight_kg"] = pd.to_numeric(df[cols["weight"]], errors="coerce")
    if cols["height"]:
        features["height_cm"] = pd.to_numeric(df[cols["height"]], errors="coerce")
    if cols["barangay"]:
        features["barangay"] = df[cols["barangay"]].astype(str)
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


def decode_barangay(codes, encoder):
    if encoder is None:
        return codes
    reverse = {i: name for i, name in enumerate(encoder.classes_)}
    return pd.Series(codes).map(reverse).fillna("UNKNOWN").values


def train_all_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y if y.nunique() > 1 else None
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    candidates = {
        "Decision Tree": DecisionTreeClassifier(random_state=CONFIG["random_state"]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=CONFIG["random_state"]),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=CONFIG["random_state"]),
    }
    trained = {}
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = y_pred.astype(float)
        trained[name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "X_test": X_test_scaled,
        }
    return trained, scaler, X_train, X_test, y_train, y_test


def metric_row(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }


def share_of(part, whole):
    return round((part / whole * 100), 2) if whole else 0.0


def kpi_row(cards):
    html = '<div class="kpi-row">'
    for label, value, caption, kind in cards:
        value_class = {"default": "", "accent": "accent", "success": "success"}.get(kind, "")
        html += (
            '<div class="kpi-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value {value_class}">{value}</div>'
            f'<div class="caption">{caption}</div>'
            '</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


raw_df, loaded_file = read_all_sheets()

if raw_df is None:
    st.error("no workbook found in the project directory")
    st.stop()

df = clean_columns(raw_df)
df = normalize_text_columns(df)
df = drop_non_records(df)
df = drop_duplicates(df)
df, col_map = prepare_columns(df)
df = drop_missing_anthropometry(df, col_map)
df = drop_missing_status(df, col_map)
df = drop_out_of_scope_age(df, col_map)
df = impute_missing(df, col_map)

target = build_target(df, col_map)
if target is None:
    st.error("required indicator columns (WAZ, HAZ, WHZ) were not detected")
    st.stop()

df = df.loc[target.index].copy()
df["malnourished"] = target.values

features = build_features(df, col_map)
valid_mask = features.notna().all(axis=1)
features = features[valid_mask].reset_index(drop=True)
labels = df.loc[valid_mask, "malnourished"].reset_index(drop=True)
df_model = df.loc[valid_mask].reset_index(drop=True)

if len(features) < 10 or labels.nunique() < 2:
    st.error("not enough valid records or class variety to train models")
    st.stop()

encoded, encoders = encode_features(features)

trained, scaler, X_train, X_test, y_train, y_test = train_all_models(encoded, labels)

metrics = pd.DataFrame({
    name: metric_row(r["y_test"], r["y_pred"]) for name, r in trained.items()
}).T

metrics["F1+Recall"] = metrics["F1-Score"] + metrics["Recall"]
best_combined = metrics["F1+Recall"].idxmax()
best_f1 = metrics["F1-Score"].idxmax()
best_recall = metrics["Recall"].idxmax()

cv_folds = CONFIG["cv_folds"]
scaled_features = StandardScaler().fit_transform(encoded)
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                     random_state=CONFIG["random_state"])
cv_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=CONFIG["random_state"]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=CONFIG["random_state"]),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=CONFIG["random_state"]),
}
cv_scores = {}
for name, model in cv_models.items():
    cv_scores[name] = cross_val_score(
        model, scaled_features, labels, cv=cv, scoring="accuracy")
cv_df = pd.DataFrame(cv_scores)
cv_df.index = [f"Fold {i+1}" for i in range(cv_folds)]
cv_df.loc["Mean"] = cv_df.mean()
cv_df.loc["Std"] = cv_df.iloc[:-2].std()

metrics_with_cv = metrics.copy()
for name in metrics_with_cv.index:
    mean_val = cv_scores[name].mean()
    std_val = cv_scores[name].std()
    metrics_with_cv.loc[name, "CV Accuracy (mean ± std)"] = (
        f"{mean_val:.4f} ± {std_val:.4f}"
    )

st.markdown(
    '<div class="app-header">'
    '<div class="title">Predictive Analysis of Child Malnutrition in Bunawan, Agusan del Sur</div>'
    '<div class="subtitle">Data-driven insights and predictive modeling using Operation Timbang records</div>'
    '</div>',
    unsafe_allow_html=True
)

overview_tab, brgy_tab, demo_tab, factors_tab, evaluation_tab, decision_tab = st.tabs([
    "Overview",
    "Results by Barangay",
    "Results by Sex and Age Group",
    "Significant Factors",
    "Model Evaluation",
    "Decision Support",
])

with st.sidebar:
    st.markdown("## Filters")
    if col_map["barangay"]:
        brgy_options = sorted(df_model[col_map["barangay"]].dropna().unique().tolist())
        selected_brgy = st.multiselect(
            "Barangay", options=brgy_options, default=brgy_options)
    else:
        selected_brgy = None
    if col_map["sex"]:
        sex_options = sorted(df_model[col_map["sex"]].dropna().unique().tolist())
        selected_sex = st.multiselect(
            "Sex", options=sex_options, default=sex_options)
    else:
        selected_sex = None
    age_range = st.slider(
        "Age in months",
        min_value=int(CONFIG["age_min"]),
        max_value=int(CONFIG["age_max"]),
        value=(int(CONFIG["age_min"]), int(CONFIG["age_max"])),
    )

filtered = df_model.copy()
if selected_brgy is not None and col_map["barangay"]:
    filtered = filtered[filtered[col_map["barangay"]].isin(selected_brgy)]
if selected_sex is not None and col_map["sex"]:
    filtered = filtered[filtered[col_map["sex"]].isin(selected_sex)]
if col_map["age"]:
    filtered = filtered[(filtered[col_map["age"]] >= age_range[0]) &
                        (filtered[col_map["age"]] <= age_range[1])]

with overview_tab:
    total = len(filtered)
    mal = int(filtered["malnourished"].sum())
    not_mal = total - mal
    share = share_of(mal, total)

    kpi_row([
        ("Records", f"{total:,}", "filtered children", "default"),
        ("Malnourished", f"{mal:,}", "binary label = 1", "accent"),
        ("Not Malnourished", f"{not_mal:,}", "binary label = 0", "success"),
        ("Malnourished share", f"{share}%", "of filtered records", "accent"),
    ])

    st.markdown("## Nutritional Status Distribution")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        counts = filtered["malnourished"].value_counts().reindex([0, 1], fill_value=0)
        bars = ax.bar(
            ["Not Malnourished", "Malnourished"],
            counts.values,
            color=[PALETTE["success"], PALETTE["accent"]],
            width=0.55
        )
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(1, total * 0.015),
                    str(int(v)), ha="center", fontsize=10, fontweight="600")
        ax.set_ylabel("Children")
        ax.set_ylim(0, counts.max() * 1.18 if counts.max() else 1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col_b:
        st.markdown("<h3>Share</h3>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="label">Malnourished</div>'
            f'<div class="value accent">{share}%</div>'
            f'<div class="caption">{mal} of {total} children</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("## Records by Year and Barangay")
    if col_map["barangay"] and col_map["sheet"]:
        crosstab = pd.crosstab(
            filtered[col_map["sheet"]], filtered[col_map["barangay"]],
            margins=True, margins_name="Total"
        )
        with st.expander("records by sheet and barangay", expanded=True):
            st.dataframe(crosstab, use_container_width=True)

    st.markdown("## Year-over-Year Malnourishment")
    if col_map["sheet"]:
        year_summary = (
            filtered.groupby(col_map["sheet"])["malnourished"]
            .agg(Total="count", Malnourished="sum")
        )
        year_summary["Not Malnourished"] = (
            year_summary["Total"] - year_summary["Malnourished"])
        year_summary["Malnourished %"] = (
            year_summary["Malnourished"] /
            year_summary["Total"].replace(0, np.nan) * 100
        ).round(2)

        fig, ax = plt.subplots(figsize=(8, 3.4))
        plot_df = year_summary.reset_index().rename(columns={col_map["sheet"]: "Sheet"})
        plot_df[["Malnourished", "Not Malnourished"]].plot(
            kind="bar", stacked=True, ax=ax,
            color=[PALETTE["accent"], PALETTE["success"]],
            width=0.5, edgecolor="white"
        )
        ax.set_xticklabels(plot_df["Sheet"].astype(str), rotation=0)
        ax.set_ylabel("Children")
        ax.set_xlabel("")
        ax.legend(frameon=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("year-over-year rates", expanded=False):
            st.dataframe(year_summary, use_container_width=True)

        if len(year_summary) >= 2:
            sheets = year_summary.index.tolist()
            first, last = sheets[0], sheets[-1]
            delta = (year_summary.loc[last, "Malnourished %"] -
                     year_summary.loc[first, "Malnourished %"])
            insight(
                f"malnourished share changed from "
                f"<b>{year_summary.loc[first, 'Malnourished %']}%</b> ({first}) to "
                f"<b>{year_summary.loc[last, 'Malnourished %']}%</b> ({last}); "
                f"difference of <b>{delta:+.2f}</b> percentage points."
            )

    st.markdown("## WHO Indicator Codes")
    indicator_cols = [col_map["waz"], col_map["haz"], col_map["whz"]]
    indicator_cols = [c for c in indicator_cols if c]
    if indicator_cols:
        cols_ui = st.columns(len(indicator_cols))
        for i, col in enumerate(indicator_cols):
            with cols_ui[i]:
                st.markdown(f"<h3>{col}</h3>", unsafe_allow_html=True)
                counts = filtered[col].fillna("not recorded").value_counts()
                st.dataframe(counts.rename("records").to_frame(),
                             use_container_width=True)

with brgy_tab:
    st.markdown("## Malnourishment by Barangay")
    if col_map["barangay"]:
        brgy_summary = (
            filtered.groupby(col_map["barangay"])["malnourished"]
            .agg(Total="count", Malnourished="sum")
        )
        brgy_summary["Not Malnourished"] = (
            brgy_summary["Total"] - brgy_summary["Malnourished"])
        brgy_summary["Malnourished %"] = (
            brgy_summary["Malnourished"] /
            brgy_summary["Total"].replace(0, np.nan) * 100
        ).round(2)
        brgy_summary = brgy_summary.sort_values(
            "Malnourished %", ascending=False)

        fig, ax = plt.subplots(figsize=(10, max(3.5, len(brgy_summary) * 0.42)))
        plot_df = brgy_summary.sort_values("Malnourished %", ascending=True)
        bars = ax.barh(plot_df.index, plot_df["Malnourished %"],
                       color=PALETTE["brand"], height=0.65)
        for bar, v in zip(bars, plot_df["Malnourished %"].values):
            ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=9, color="#64748B")
        ax.set_xlabel("Malnourished share (%)")
        ax.set_xlim(0, max(plot_df["Malnourished %"].max() * 1.25, 10))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("barangay table", expanded=True):
            st.dataframe(brgy_summary, use_container_width=True)

        worst = brgy_summary.index[0]
        insight(
            "highest malnourished share recorded in <b>" + str(worst) +
            "</b> at <b>" + str(brgy_summary.loc[worst, "Malnourished %"]) +
            "%</b>."
        )

with demo_tab:
    st.markdown("## Malnourishment by Sex")
    if col_map["sex"]:
        sex_summary = (
            filtered.groupby(col_map["sex"])["malnourished"]
            .agg(Total="count", Malnourished="sum")
        )
        sex_summary["Not Malnourished"] = (
            sex_summary["Total"] - sex_summary["Malnourished"])
        sex_summary["Malnourished %"] = (
            sex_summary["Malnourished"] /
            sex_summary["Total"].replace(0, np.nan) * 100
        ).round(2)
        st.dataframe(sex_summary, use_container_width=True)

    st.markdown("## Malnourishment by Age Group")
    if col_map["age"]:
        work = filtered.copy()
        work["age_group"] = pd.cut(
            work[col_map["age"]],
            bins=CONFIG["age_bins"],
            labels=CONFIG["age_labels"],
            include_lowest=True,
            right=False,
        )
        age_summary = (
            work.groupby("age_group", observed=True)["malnourished"]
            .agg(Total="count", Malnourished="sum")
        )
        age_summary["Not Malnourished"] = (
            age_summary["Total"] - age_summary["Malnourished"])
        age_summary["Malnourished %"] = (
            age_summary["Malnourished"] /
            age_summary["Total"].replace(0, np.nan) * 100
        ).round(2)

        fig, ax = plt.subplots(figsize=(10, 3.4))
        plot_df = age_summary.reset_index()
        bars = ax.bar(
            plot_df["age_group"].astype(str),
            plot_df["Malnourished %"],
            color=PALETTE["brand_light"], width=0.6
        )
        for bar, v in zip(bars, plot_df["Malnourished %"].values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_xlabel("Age group (months)")
        ax.set_ylabel("Malnourished share (%)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("age-group table", expanded=False):
            st.dataframe(age_summary, use_container_width=True)

with factors_tab:
    st.markdown("## Variable Importance Across Models")

    importance = {}
    for name, r in trained.items():
        model = r["model"]
        if hasattr(model, "feature_importances_"):
            importance[name] = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance[name] = np.abs(model.coef_[0])

    if importance:
        imp_df = pd.DataFrame(importance, index=encoded.columns)
        imp_df["Average"] = imp_df.mean(axis=1)
        imp_df = imp_df.sort_values("Average", ascending=False)

        rank_df = imp_df.drop(columns="Average").rank(ascending=False, method="average")
        imp_df["Consensus Rank"] = rank_df.mean(axis=1).round(2)
        imp_df = imp_df.sort_values("Consensus Rank", ascending=True)

        st.caption("tree importance and absolute logistic coefficients are not "
                   "on a common scale; consensus rank is the mean of per-model ranks.")

        fig, ax = plt.subplots(figsize=(10, max(3.5, len(imp_df) * 0.45)))
        plot_df = imp_df.drop(columns=["Average", "Consensus Rank"]).iloc[::-1]
        plot_df.plot(kind="barh", ax=ax,
                     color=PALETTE["series"][:len(plot_df.columns)],
                     width=0.7, edgecolor="white")
        ax.set_xlabel("Importance or absolute coefficient")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("importance table", expanded=True):
            st.dataframe(imp_df.round(4), use_container_width=True)

        top_name = imp_df.index[0]
        insight(
            "highest consensus importance: <b>" + str(top_name) +
            "</b> with mean rank <b>" +
            f"{imp_df.loc[top_name, 'Consensus Rank']:.2f}</b>."
        )

with evaluation_tab:
    st.markdown("## Test-Set Performance")
    st.caption("point estimates come from a single stratified 80/20 split; "
               "the last column reports cross-validation accuracy mean ± std.")
    display_metrics = metrics_with_cv.drop(columns="F1+Recall").round(4)
    st.dataframe(
        display_metrics.style.highlight_max(axis=0, color="#CCFBF1"),
        use_container_width=True
    )

    st.markdown("## Best Model Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="label">Highest F1</div>'
            f'<div class="value">{metrics.loc[best_f1, "F1-Score"]:.4f}</div>'
            f'<div class="caption">{best_f1}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="label">Highest Recall</div>'
            f'<div class="value">{metrics.loc[best_recall, "Recall"]:.4f}</div>'
            f'<div class="caption">{best_recall}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="label">Highest F1 + Recall</div>'
            f'<div class="value accent">{metrics.loc[best_combined, "F1+Recall"]:.4f}</div>'
            f'<div class="caption">{best_combined}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.caption("best-model selection uses combined F1 + Recall, per the "
               "screening-oriented evaluation; a precision-first policy "
               "would reorder the candidates.")

    st.markdown("## Confusion Matrices")
    pair_cols = st.columns(2)
    for i, (name, r) in enumerate(trained.items()):
        with pair_cols[i % 2]:
            cm = confusion_matrix(r["y_test"], r["y_pred"], labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4.2, 3.4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu",
                        xticklabels=["Not Mal.", "Mal."],
                        yticklabels=["Not Mal.", "Mal."], ax=ax,
                        cbar=False, annot_kws={"fontsize": 12, "fontweight": "600"})
            ax.set_title(name, pad=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("## Per-Model Classification Report")
    for name, r in trained.items():
        with st.expander(name):
            report = classification_report(
                r["y_test"], r["y_pred"],
                target_names=["Not Malnourished", "Malnourished"],
                output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).T.round(4),
                         use_container_width=True)

    st.markdown("## Recall by Barangay (Test Set)")
    if col_map["barangay"]:
        test_rows = X_test.index.tolist()
        if "barangay" in encoded.columns:
            brgy_codes_test = encoded.iloc[test_rows]["barangay"].values
            brgy_names_test = decode_barangay(
                brgy_codes_test, encoders.get("barangay"))
        else:
            brgy_names_test = None

        if brgy_names_test is not None:
            recall_records = []
            for name, r in trained.items():
                df_te = pd.DataFrame({
                    "barangay": brgy_names_test,
                    "y_true": r["y_test"].values,
                    "y_pred": r["y_pred"],
                })
                for brgy_name, group in df_te.groupby("barangay"):
                    pos_mask = group["y_true"] == 1
                    if pos_mask.sum() == 0:
                        recall_val = np.nan
                    else:
                        recall_val = (
                            (group.loc[pos_mask, "y_pred"] == 1).sum()
                            / pos_mask.sum()
                        )
                    recall_records.append({
                        "Model": name,
                        "Barangay": brgy_name,
                        "Positives": int(pos_mask.sum()),
                        "Recall": round(float(recall_val), 4) if not pd.isna(recall_val) else "n/a",
                    })
            recall_df = pd.DataFrame(recall_records)
            with st.expander("per-barangay recall on the held-out test set"):
                st.dataframe(recall_df, use_container_width=True)

    st.markdown("## Cross-Validation Accuracy")
    with st.expander("fold-by-fold cross-validation accuracy", expanded=True):
        st.dataframe(cv_df.round(4), use_container_width=True)
        mean_lines = " | ".join(
            f"{n}: {s.mean():.4f} ± {s.std():.4f}"
            for n, s in cv_scores.items()
        )
        st.write("mean cross-validation accuracy — " + mean_lines)

with decision_tab:
    st.markdown("## Predicted Risk per Barangay")
    best_model = trained[best_combined]["model"]
    full_matrix = scaler.transform(encoded)
    if hasattr(best_model, "predict_proba"):
        full_proba = best_model.predict_proba(full_matrix)[:, 1]
    else:
        full_proba = best_model.predict(full_matrix).astype(float)
    full_pred = (full_proba >= 0.5).astype(int)

    scored = df_model.copy()
    scored["risk_probability"] = full_proba
    scored["predicted_risk"] = full_pred
    scored["predicted_label"] = scored["predicted_risk"].map(
        {1: "At Risk", 0: "Not At Risk"})

    st.caption("risk_probability is the model's estimated probability of "
               "malnourishment; the binary label uses a 0.50 threshold.")

    if col_map["barangay"]:
        risk_summary = (
            scored.groupby(col_map["barangay"])["predicted_risk"]
            .agg(Total="count", At_Risk="sum")
        )
        risk_summary["Not At Risk"] = (
            risk_summary["Total"] - risk_summary["At_Risk"])
        risk_summary["At Risk %"] = (
            risk_summary["At_Risk"] /
            risk_summary["Total"].replace(0, np.nan) * 100
        ).round(2)
        mean_prob = (
            scored.groupby(col_map["barangay"])["risk_probability"].mean().round(4)
        )
        risk_summary["Mean Probability"] = mean_prob
        risk_summary = risk_summary.sort_values(
            "Mean Probability", ascending=False)

        fig, ax = plt.subplots(figsize=(10, max(3.5, len(risk_summary) * 0.42)))
        plot_df = risk_summary.sort_values("Mean Probability", ascending=True)
        bars = ax.barh(plot_df.index, plot_df["Mean Probability"],
                       color=PALETTE["warning"], height=0.65)
        for bar, v in zip(bars, plot_df["Mean Probability"].values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9, color="#64748B")
        ax.set_xlabel("Mean predicted probability")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("risk table by barangay", expanded=True):
            st.dataframe(risk_summary, use_container_width=True)

        top = risk_summary.index[0]
        insight(
            "highest mean predicted probability in <b>" + str(top) +
            "</b> at <b>" + f"{risk_summary.loc[top, 'Mean Probability']:.4f}" +
            "</b>; best model for scoring is <b>" + best_combined + "</b>."
        )

    st.markdown("## Predicted Risk Distribution")
    dist = scored["predicted_label"].value_counts()
    col_a, col_b = st.columns([2, 3])
    with col_a:
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.pie(dist.values, labels=dist.index, autopct="%1.1f%%",
               colors=[PALETTE["accent"], PALETTE["success"]][:len(dist)],
               startangle=90,
               wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
               textprops=dict(color="#1E293B", fontsize=9))
        ax.set_title("Predicted nutritional risk")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col_b:
        st.dataframe(dist.rename("count").to_frame(),
                     use_container_width=True)

    st.markdown("## Record-Level Predictions")
    show_cols = []
    for key in ("barangay", "sex", "age", "weight", "height"):
        if col_map[key]:
            show_cols.append(col_map[key])
    if show_cols:
        view = scored[show_cols + ["malnourished", "risk_probability",
                                   "predicted_label"]].sort_values(
            "risk_probability", ascending=False
        )
        with st.expander("highest predicted risk records first", expanded=True):
            st.dataframe(view.head(20), use_container_width=True)

    st.markdown("## Downloads")
    export_cols = show_cols + ["malnourished", "risk_probability", "predicted_label"]
    export = scored[export_cols].copy()
    st.download_button(
        "Download per-child predictions CSV",
        export.to_csv(index=False).encode("utf-8"),
        "per_child_predictions.csv",
        "text/csv"
    )

st.markdown(
    '<div class="app-footer">'
    + "model scores — "
    + " | ".join(
        f"{name}: F1 {metrics.loc[name, 'F1-Score']:.4f}, "
        f"Recall {metrics.loc[name, 'Recall']:.4f}"
        for name in metrics.index
    )
    + '</div>',
    unsafe_allow_html=True
)
