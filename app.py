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
    page_title="Child Nutritional Status Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
h1 {color: #1f4e79;}
h2 {color: #1f4e79; border-bottom: 1px solid #e6e6e6; padding-bottom: 4px;}
h3 {color: #333;}
[data-testid="stMetricValue"] {color: #1f4e79;}
</style>
""", unsafe_allow_html=True)

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
    for col in ("sex", "barangay"):
        if col in encoded.columns:
            encoded[col] = LabelEncoder().fit_transform(encoded[col].astype(str))
    return encoded

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

encoded = encode_features(features)

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

st.title("Child Nutritional Status Dashboard")
st.caption("Computed results from the Operation Timbang records")

overview_tab, brgy_tab, demo_tab, factors_tab, evaluation_tab, decision_tab = st.tabs([
    "Overview",
    "Results by Barangay",
    "Results by Sex and Age Group",
    "Significant Factors",
    "Model Evaluation",
    "Decision Support",
])

with st.sidebar:
    st.subheader("Filters")
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
    st.subheader("Record Summary")
    total = len(filtered)
    mal = int(filtered["malnourished"].sum())
    not_mal = total - mal
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{total:,}")
    c2.metric("Malnourished", f"{mal:,}")
    c3.metric("Not Malnourished", f"{not_mal:,}")
    c4.metric("Malnourished share", f"{share_of(mal, total)}%")

    st.subheader("Distribution of Nutritional Status")
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = filtered["malnourished"].value_counts().reindex([0, 1], fill_value=0)
    ax.bar(["Not Malnourished", "Malnourished"], counts.values,
           color=["#2ecc71", "#e74c3c"], edgecolor="black")
    ax.set_ylabel("Children")
    ax.set_title("Binary nutritional status")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(1, total * 0.01), str(int(v)), ha="center")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Records by Year and Barangay")
    if col_map["barangay"] and col_map["sheet"]:
        crosstab = pd.crosstab(
            filtered[col_map["sheet"]], filtered[col_map["barangay"]],
            margins=True, margins_name="Total"
        )
        with st.expander("Records by sheet and barangay", expanded=True):
            st.dataframe(crosstab, use_container_width=True)
    elif col_map["sheet"]:
        sheet_counts = filtered[col_map["sheet"]].value_counts()
        st.dataframe(sheet_counts.rename("records").to_frame(),
                     use_container_width=True)

    st.subheader("Year-over-Year Malnourishment")
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
        with st.expander("Year-over-year malnourishment rates", expanded=True):
            st.dataframe(year_summary, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df = year_summary.reset_index().rename(columns={col_map["sheet"]: "Sheet"})
        plot_df[["Malnourished", "Not Malnourished"]].plot(
            kind="bar", stacked=True, ax=ax,
            color=["#e74c3c", "#2ecc71"], edgecolor="black"
        )
        ax.set_xticklabels(plot_df["Sheet"].astype(str), rotation=0)
        ax.set_ylabel("Children")
        ax.set_title("Malnourishment count by OPT sheet")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if len(year_summary) >= 2:
            sheets = year_summary.index.tolist()
            first = sheets[0]
            last = sheets[-1]
            delta = (year_summary.loc[last, "Malnourished %"] -
                     year_summary.loc[first, "Malnourished %"])
            st.write(
                "malnourished share changed from " +
                f"{year_summary.loc[first, 'Malnourished %']}% ({first}) " +
                f"to {year_summary.loc[last, 'Malnourished %']}% ({last}); " +
                f"difference of {delta:+.2f} percentage points."
            )
    else:
        st.info("no sheet column detected")

    st.subheader("WHO Indicator Codes")
    indicator_cols = [col_map["waz"], col_map["haz"], col_map["whz"]]
    indicator_cols = [c for c in indicator_cols if c]
    if indicator_cols:
        cols_ui = st.columns(len(indicator_cols))
        for i, col in enumerate(indicator_cols):
            with cols_ui[i]:
                counts = filtered[col].fillna("not recorded").value_counts()
                st.dataframe(counts.rename("records").to_frame(),
                             use_container_width=True)
    else:
        st.info("no indicator columns detected")

with brgy_tab:
    st.subheader("Malnourishment by Barangay")
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
        st.dataframe(brgy_summary, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(brgy_summary) * 0.45)))
        plot_df = brgy_summary.sort_values("Malnourished %", ascending=True)
        ax.barh(plot_df.index, plot_df["Malnourished %"], color="#e74c3c")
        ax.set_xlabel("Malnourished share (%)")
        ax.set_title("Share of malnourished children per barangay")
        for i, v in enumerate(plot_df["Malnourished %"].values):
            ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        worst = brgy_summary.index[0]
        st.write(
            "highest malnourished share recorded in " + worst +
            " at " + str(brgy_summary.loc[worst, "Malnourished %"]) + "%.")
    else:
        st.info("no barangay column detected")

with demo_tab:
    st.subheader("Malnourishment by Sex")
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

    st.subheader("Malnourishment by Age Group")
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
        st.dataframe(age_summary, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        plot_df = age_summary.reset_index()
        ax.bar(plot_df["age_group"].astype(str),
               plot_df["Malnourished %"], color="#3498db", edgecolor="black")
        ax.set_xlabel("Age group (months)")
        ax.set_ylabel("Malnourished share (%)")
        ax.set_title("Share of malnourished children per age group")
        for i, v in enumerate(plot_df["Malnourished %"].values):
            ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with factors_tab:
    st.subheader("Variable Importance Across Models")

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
        st.dataframe(imp_df.round(4), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(imp_df) * 0.45)))
        plot_df = imp_df.drop(columns=["Average", "Consensus Rank"])
        plot_df.plot(kind="barh", ax=ax)
        ax.set_xlabel("Importance or absolute coefficient")
        ax.invert_yaxis()
        ax.set_title("Per-variable importance by model")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        top_name = imp_df.index[0]
        st.write(
            "highest consensus importance: " + str(top_name) +
            " with mean rank " + f"{imp_df.loc[top_name, 'Consensus Rank']:.2f}.")
    else:
        st.info("no importance-computable model was trained")

with evaluation_tab:
    st.subheader("Test-Set Performance")
    st.caption("point estimates come from a single stratified 80/20 split; "
               "the last column reports cross-validation accuracy mean ± std.")
    display_metrics = metrics_with_cv.drop(columns="F1+Recall").round(4)
    st.dataframe(
        display_metrics.style.highlight_max(axis=0, color="#c6efce"),
        use_container_width=True
    )

    st.subheader("Best Model Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Highest F1", best_f1,
              f"{metrics.loc[best_f1, 'F1-Score']:.4f}")
    c2.metric("Highest Recall", best_recall,
              f"{metrics.loc[best_recall, 'Recall']:.4f}")
    c3.metric("Highest F1 + Recall", best_combined,
              f"{metrics.loc[best_combined, 'F1+Recall']:.4f}")

    st.caption("best-model selection uses combined F1 + Recall, per the "
               "screening-oriented evaluation; a precision-first policy "
               "would reorder the candidates.")

    st.subheader("Confusion Matrices")
    pair_cols = st.columns(2)
    for i, (name, r) in enumerate(trained.items()):
        with pair_cols[i % 2]:
            cm = confusion_matrix(r["y_test"], r["y_pred"], labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4, 3.4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Not Mal.", "Mal."],
                        yticklabels=["Not Mal.", "Mal."], ax=ax)
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.subheader("Per-Model Classification Report")
    for name, r in trained.items():
        with st.expander(name):
            report = classification_report(
                r["y_test"], r["y_pred"],
                target_names=["Not Malnourished", "Malnourished"],
                output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).T.round(4),
                         use_container_width=True)

    st.subheader("Recall by Barangay (Test Set)")
    if col_map["barangay"]:
        test_rows = X_test.index.tolist()
        test_barangays = encoded.iloc[test_rows]["barangay"].values \
            if "barangay" in encoded.columns else None
        if test_barangays is not None:
            recall_records = []
            for name, r in trained.items():
                df_te = pd.DataFrame({
                    "barangay_code": test_barangays,
                    "y_true": r["y_test"].values,
                    "y_pred": r["y_pred"],
                })
                for brgy_code, group in df_te.groupby("barangay_code"):
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
                        "Barangay Code": brgy_code,
                        "Positives": int(pos_mask.sum()),
                        "Recall": round(float(recall_val), 4) if not pd.isna(recall_val) else "n/a",
                    })
            recall_df = pd.DataFrame(recall_records)
            with st.expander("per-barangay recall on the held-out test set"):
                st.dataframe(recall_df, use_container_width=True)
    else:
        st.info("barangay column not detected; per-barangay recall not available")

    st.subheader("Cross-Validation Accuracy")
    with st.expander("fold-by-fold cross-validation accuracy", expanded=True):
        st.dataframe(cv_df.round(4), use_container_width=True)
        mean_lines = " | ".join(
            f"{n}: {s.mean():.4f} ± {s.std():.4f}"
            for n, s in cv_scores.items()
        )
        st.write("mean cross-validation accuracy - " + mean_lines)

with decision_tab:
    st.subheader("Predicted Risk per Barangay")
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
        with st.expander("predicted at-risk children per barangay"):
            st.dataframe(risk_summary, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(risk_summary) * 0.45)))
        plot_df = risk_summary.sort_values("Mean Probability", ascending=True)
        ax.barh(plot_df.index, plot_df["Mean Probability"], color="#e67e22")
        ax.set_xlabel("Mean predicted probability of malnourishment")
        ax.set_title("Predicted probability of malnourishment per barangay (" +
                     best_combined + ")")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if len(risk_summary) > 0:
            top = risk_summary.index[0]
            st.write(
                "highest mean predicted probability in " + str(top) +
                " at " + f"{risk_summary.loc[top, 'Mean Probability']:.4f}; " +
                "best model for scoring is " + best_combined + ".")

    st.subheader("Predicted Risk Distribution")
    dist = scored["predicted_label"].value_counts()
    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.pie(dist.values, labels=dist.index, autopct="%1.1f%%",
               colors=["#e74c3c", "#2ecc71"][:len(dist)], startangle=90)
        ax.set_title("Predicted nutritional risk")
        st.pyplot(fig)
        plt.close()
    with c2:
        st.dataframe(dist.rename("count").to_frame(),
                     use_container_width=True)

    st.subheader("Record-Level Predictions")
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

    st.subheader("Downloads")
    export_cols = show_cols + ["malnourished", "risk_probability", "predicted_label"]
    export = scored[export_cols].copy()
    st.download_button(
        "Download per-child predictions CSV",
        export.to_csv(index=False).encode("utf-8"),
        "per_child_predictions.csv",
        "text/csv"
    )

st.caption("model scores - " + " | ".join(
    f"{name}: F1 {metrics.loc[name, 'F1-Score']:.4f}, "
    f"Recall {metrics.loc[name, 'Recall']:.4f}"
    for name in metrics.index
))
