import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Child Malnutrition Prediction Dashboard",
    page_icon="🍼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f4e79;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h2 { color: white; margin: 0; font-size: 2rem; }
    .metric-card p { color: white; margin: 0; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS — WHO classification codes
# ============================================================
MALNOURISHED_CODES = {'UW', 'SUW', 'St', 'SSt', 'MW', 'SW'}
NORMAL_CODES = {'N', 'OW', 'Ob', 'T'}


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================
@st.cache_data
def load_data():
    """Load both OPT 2024 and OPT 2025 sheets from the Excel file.
    Works on Streamlit Cloud (current folder) and Colab (/content/data)."""
    search_paths = ['.', '/content/data']
    files = []
    for path in search_paths:
        if os.path.isdir(path):
            files += glob.glob(os.path.join(path, '*.xlsx'))
            files += glob.glob(os.path.join(path, '*.xls'))
            files += glob.glob(os.path.join(path, '*.csv'))

    if not files:
        return None, None

    preferred = [f for f in files if 'nutritional' in os.path.basename(f).lower()
                 or 'rhu' in os.path.basename(f).lower()
                 or 'opt' in os.path.basename(f).lower()]
    target_file = preferred[0] if preferred else files[0]

    try:
        if target_file.lower().endswith('.csv'):
            df = pd.read_csv(target_file, encoding='utf-8-sig', low_memory=False)
            df['_source_year'] = 'Unknown'
        else:
            xls = pd.ExcelFile(target_file)
            sheet_names = xls.sheet_names
            all_sheets = []
            for s in sheet_names:
                s_lower = s.lower()
                if 'opt' in s_lower or '202' in s_lower:
                    try:
                        sheet_df = pd.read_excel(xls, sheet_name=s, header=None)
                        header_idx = None
                        for i in range(min(10, len(sheet_df))):
                            row_vals = [str(v).strip().upper() for v in sheet_df.iloc[i].tolist()]
                            if any(v == 'NO.' or v == 'NO' for v in row_vals):
                                header_idx = i
                                break
                        if header_idx is None:
                            header_idx = 3
                        header_row = sheet_df.iloc[header_idx].tolist()
                        new_cols = []
                        for j, c in enumerate(header_row):
                            c_str = str(c).replace('\n', ' ').strip()
                            if pd.isna(c) or c_str in ('nan', ''):
                                c_str = f"col_{j}"
                            new_cols.append(c_str)
                        body = sheet_df.iloc[header_idx + 1:].copy()
                        body.columns = new_cols
                        body = body.reset_index(drop=True)
                        body['_source_year'] = s
                        all_sheets.append(body)
                    except Exception as e:
                        st.warning(f"Could not read sheet '{s}': {e}")
                        continue
            if not all_sheets:
                return None, None
            df = pd.concat(all_sheets, ignore_index=True, sort=False)
    except Exception as e:
        st.error(f"Error reading {target_file}: {e}")
        return None, None

    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]

    no_col = None
    for c in df.columns:
        if str(c).strip().upper() in ('NO.', 'NO'):
            no_col = c
            break
    if no_col:
        df = df[pd.to_numeric(df[no_col], errors='coerce').notna()].copy()

    df = df.dropna(how='all').dropna(axis=1, how='all')

    return df, os.path.basename(target_file)


def find_col(df, *candidates):
    """Find a column by fuzzy name match."""
    cols_norm = {str(c).lower().replace(' ', '').replace('_', '').replace('-', ''): c
                 for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(' ', '').replace('_', '').replace('-', '')
        if key in cols_norm:
            return cols_norm[key]
    return None


def derive_malnutrition_label(df):
    """Build a binary 'malnourished' label from WAZ, HAZ, WHZ codes."""
    waz_col = find_col(df, 'WEIGHT-FOR-AGE (WAZ)', 'WAZ', 'weight-for-age')
    haz_col = find_col(df, 'HEIGHT-FOR-AGE (HAZ)', 'HAZ', 'height-for-age')
    whz_col = find_col(df, 'WEIGHT-FOR-HEIGHT (WHZ)', 'WHZ', 'weight-for-height')

    if not all([waz_col, haz_col, whz_col]):
        return None, None, None, None

    def is_malnourished(row):
        for c in (waz_col, haz_col, whz_col):
            v = str(row[c]).strip().upper()
            if v in MALNOURISHED_CODES:
                return 1
        return 0

    y = df.apply(is_malnourished, axis=1)
    return y, waz_col, haz_col, whz_col


def preprocess_data(df):
    """Encode features, build target label, return X, y, df_clean."""
    df = df.copy()

    y_series, waz_col, haz_col, whz_col = derive_malnutrition_label(df)
    if y_series is None:
        return None, None, None, None, None, None

    df['_malnourished'] = y_series.values

    sex_col = find_col(df, 'SEX')
    age_col = find_col(df, 'AGE (Months)', 'AGE', 'age (months)')
    weight_col = find_col(df, 'WEIGHT (kg)', 'WEIGHT', 'weight (kg)')
    height_col = find_col(df, 'HEIGHT (cm)', 'HEIGHT', 'height (cm)')
    brgy_col = find_col(df, 'BARANGAY')

    feature_frame = pd.DataFrame(index=df.index)
    if sex_col:    feature_frame['sex'] = df[sex_col].astype(str).str.strip().str.upper()
    if age_col:    feature_frame['age_months'] = pd.to_numeric(df[age_col], errors='coerce')
    if weight_col: feature_frame['weight_kg'] = pd.to_numeric(df[weight_col], errors='coerce')
    if height_col: feature_frame['height_cm'] = pd.to_numeric(df[height_col], errors='coerce')
    if brgy_col:   feature_frame['barangay'] = df[brgy_col].astype(str).str.strip().str.upper()

    valid_mask = feature_frame.notna().all(axis=1)
    feature_frame = feature_frame[valid_mask].copy()
    y = df.loc[valid_mask, '_malnourished'].astype(int)
    df_clean = df.loc[valid_mask].copy()

    for col in ['sex', 'barangay']:
        if col in feature_frame.columns:
            le = LabelEncoder()
            feature_frame[col] = le.fit_transform(feature_frame[col].astype(str))

    X = feature_frame.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df_clean = df_clean.reset_index(drop=True)

    return X, y, df_clean, waz_col, haz_col, whz_col


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_models(X, y, test_size=0.2, random_state=42):
    """Train 4 classifiers per Chapter 3 methodology."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Naïve Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
            'X_test': X_test_scaled
        }

    return results, scaler, X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred):
    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, zero_division=0)
    }


# ============================================================
# LOAD DATA
# ============================================================
df_raw, filename = load_data()

st.markdown('<div class="main-header">🍼 Child Malnutrition Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predictive Analysis of Child Malnutrition in Bunawan, Agusan del Sur<br>'
            '<i>Using Data Mining Techniques (Decision Tree • Random Forest • Naïve Bayes • Logistic Regression)</i></div>',
            unsafe_allow_html=True)

# ============================================================
# 🔍 DEBUG PANEL
# ============================================================
with st.expander("🔍 Debug: Detected Data", expanded=False):
    if df_raw is None:
        st.error("❌ No dataset found. Please upload `nutritional-status-RHU-1.xlsx` to the repo root.")
    else:
        st.success(f"✅ Loaded file: `{filename}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df_raw.shape[0]:,}")
        c2.metric("Columns", df_raw.shape[1])
        c3.metric("Sheets", df_raw['_source_year'].nunique()
                  if '_source_year' in df_raw.columns else 1)

        st.write("**All detected columns:**")
        st.code(list(df_raw.columns))

        st.write("**First 5 rows:**")
        st.dataframe(df_raw.head(), use_container_width=True)

if df_raw is None:
    st.stop()

# Preprocess
X, y, df_clean, waz_col, haz_col, whz_col = preprocess_data(df_raw)

if X is None or len(X) < 10:
    st.error("⚠️ Could not build a valid dataset. Check that the Excel file has "
             "columns: SEX, AGE (Months), WEIGHT (kg), HEIGHT (cm), BARANGAY, "
             "and WAZ/HAZ/WHZ indicator columns.")
    st.stop()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Model Configuration")
st.sidebar.info(f"📊 Total records after cleaning: **{len(X)}**")

available_features = list(X.columns)
selected_features = st.sidebar.multiselect(
    "Select predictor variables:",
    options=available_features,
    default=available_features
)

if len(selected_features) < 2:
    st.warning("⚠️ Please select at least 2 predictor variables from the sidebar.")
    st.stop()

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

X_sel = X[selected_features]

# Train
results, scaler, X_train, X_test, y_train, y_test = train_models(
    X_sel, y, test_size=test_size, random_state=int(random_state)
)

metrics_table = pd.DataFrame({
    name: compute_metrics(r['y_test'], r['y_pred'])
    for name, r in results.items()
}).T

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "📈 EDA",
    "🌳 Per-Algorithm Details", "📋 Cross-Validation",
    "🔑 Feature Importance", "💡 Recommendations"
])


# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab1:
    st.header("📊 Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><p>Total Records</p><h2>{len(X):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><p>Predictors Used</p><h2>{len(selected_features)}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><p>Malnourished</p><h2>{(y==1).sum():,}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><p>Not Malnourished</p><h2>{(y==0).sum():,}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🏆 Model Performance Summary")
    st.dataframe(
        metrics_table.style.format("{:.4f}").highlight_max(axis=0, color='#c6efce'),
        use_container_width=True
    )

    st.subheader("📋 Dataset Sample (Cleaned)")
    display_cols = [c for c in df_clean.columns if not c.startswith('_')]
    st.dataframe(df_clean[display_cols].head(10), use_container_width=True)


# ------------------------------------------------------------
# TAB 2: EDA
# ------------------------------------------------------------
with tab2:
    st.header("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = y.map({0: 'Not Malnourished', 1: 'Malnourished'}).value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=colors[:len(counts)], startangle=90)
        ax.set_title("Nutritional Status Distribution")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Class Counts")
        st.dataframe(
            counts.rename('Count').to_frame().assign(
                Percentage=lambda d: (d['Count']/d['Count'].sum()*100).round(2)
            ), use_container_width=True
        )

    st.markdown("---")
    st.subheader("WHO Indicator Distributions (WAZ / HAZ / WHZ)")
    indicator_cols = [c for c in [waz_col, haz_col, whz_col] if c and c in df_clean.columns]
    if indicator_cols:
        cols = st.columns(len(indicator_cols))
        for i, col in enumerate(indicator_cols):
            with cols[i]:
                vc = df_clean[col].astype(str).str.strip().value_counts()
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(vc.index, vc.values, color='#3498db', edgecolor='black')
                ax.set_title(str(col))
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    st.markdown("---")
    st.subheader("Numeric Feature Distributions")
    numeric_cols = [c for c in selected_features if X_sel[c].nunique() > 5][:6]
    if numeric_cols:
        n = len(numeric_cols)
        fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(15, 4 * ((n + 2) // 3)))
        axes = axes.flatten() if n > 1 else [axes]
        for i, col in enumerate(numeric_cols):
            axes[i].hist(X_sel[col].dropna().astype(float), bins=30,
                         color='#3498db', edgecolor='black')
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Correlation Heatmap (Features + Target)")
    corr_data = X_sel.copy()
    corr_data['_target'] = y.values
    corr = corr_data.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax,
                cbar_kws={'shrink': 0.8})
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ------------------------------------------------------------
# TAB 3: PER-ALGORITHM DETAILS
# ------------------------------------------------------------
with tab3:
    st.header("🌳 Per-Algorithm Performance Details")

    selected_alg = st.selectbox("Choose an algorithm to inspect:",
                                list(results.keys()))

    result = results[selected_alg]
    m = compute_metrics(result['y_test'], result['y_pred'])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{m['Accuracy']:.4f}")
    c2.metric("Precision", f"{m['Precision']:.4f}")
    c3.metric("Recall",    f"{m['Recall']:.4f}")
    c4.metric("F1-Score",  f"{m['F1-Score']:.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Malnourished', 'Malnourished'],
                    yticklabels=['Not Malnourished', 'Malnourished'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Classification Report")
        rep = classification_report(result['y_test'], result['y_pred'],
                                    target_names=['Not Malnourished', 'Malnourished'],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

    st.markdown("---")
    model = result['model']
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importances")
        imp = pd.Series(model.feature_importances_, index=selected_features).sort_values()
        fig, ax = plt.subplots(figsize=(9, max(3.5, len(imp) * 0.5)))
        imp.plot(kind='barh', color='#2ecc71', ax=ax)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    elif hasattr(model, 'coef_'):
        st.subheader("Coefficient Analysis")
        coefs = pd.Series(model.coef_[0], index=selected_features).sort_values()
        fig, ax = plt.subplots(figsize=(9, max(3.5, len(coefs) * 0.5)))
        colors_bar = ['#e74c3c' if v < 0 else '#2ecc71' for v in coefs.values]
        coefs.plot(kind='barh', color=colors_bar, ax=ax)
        ax.axvline(0, color='black', lw=1)
        ax.set_xlabel("Coefficient")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Positive → increases likelihood of malnutrition; "
                   "Negative → decreases likelihood.")


# ------------------------------------------------------------
# TAB 4: CROSS-VALIDATION
# ------------------------------------------------------------
with tab4:
    st.header("📋 Stratified K-Fold Cross-Validation")

    k = st.slider("Number of folds", 3, 10, 5)

    X_scaled = StandardScaler().fit_transform(X_sel)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    cv_models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naïve Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    cv_scores = {}
    for name, model in cv_models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        cv_scores[name] = scores

    cv_df = pd.DataFrame(cv_scores)
    cv_df.index = [f"Fold {i+1}" for i in range(k)]
    cv_df.loc['Mean'] = cv_df.mean()
    cv_df.loc['Std']  = cv_df.iloc[:-2].std()

    st.dataframe(cv_df.round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    cv_df.iloc[:-2].plot(kind='bar', ax=ax,
                         color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{k}-Fold Cross-Validation Accuracy per Fold")
    ax.legend(loc='lower right', ncol=4)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("**Mean CV Accuracy:** " +
            " | ".join([f"{n}: {s.mean():.4f} ± {s.std():.4f}"
                        for n, s in cv_scores.items()]))


# ------------------------------------------------------------
# TAB 5: FEATURE IMPORTANCE (Aggregate)
# ------------------------------------------------------------
with tab5:
    st.header("🔑 Aggregate Feature Importance")

    importance_data = {}
    for name, r in results.items():
        model = r['model']
        if hasattr(model, 'feature_importances_'):
            importance_data[name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_data[name] = np.abs(model.coef_[0])

    if importance_data:
        imp_df = pd.DataFrame(importance_data, index=selected_features)
        imp_df['Average'] = imp_df.mean(axis=1)
        imp_df = imp_df.sort_values('Average', ascending=False)

        st.dataframe(imp_df.round(4), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(imp_df) * 0.5)))
        imp_df.drop(columns='Average').plot(kind='barh', ax=ax)
        ax.set_xlabel("Importance / |Coefficient|")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.success(f"🎯 Top predictor across all models: **{imp_df.index[0]}**")
    else:
        st.warning("No feature-importance-capable model was trained.")


# ------------------------------------------------------------
# TAB 6: RECOMMENDATIONS
# ------------------------------------------------------------
with tab6:
    st.header("💡 Recommendations & Interpretation")

    best_model_f1 = metrics_table['F1-Score'].idxmax()
    best_f1 = metrics_table['F1-Score'].max()

    st.success(f"### 🏆 Best Performing Model: **{best_model_f1}** "
               f"(F1-Score = {best_f1:.4f})")

    st.markdown("---")
    st.subheader("📌 Full Model Performance Comparison")
    st.dataframe(metrics_table.style.format("{:.4f}"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_table.plot(kind='bar', ax=ax,
                       color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Across All Metrics")
    ax.legend(loc='lower right', ncol=4)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("🔑 Key Risk Factors (from Logistic Regression)")

    lr_model = results['Logistic Regression']['model']
    coefs = pd.Series(lr_model.coef_[0], index=selected_features).sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ⚠️ Factors INCREASING Malnutrition Risk")
        pos = coefs[coefs > 0]
        if len(pos) == 0:
            st.write("_None_")
        for f, v in pos.items():
            st.markdown(f"- **{f}**: +{v:.4f}")
    with col2:
        st.markdown("#### ✅ Factors DECREASING Malnutrition Risk")
        neg = coefs[coefs < 0]
        if len(neg) == 0:
            st.write("_None_")
        for f, v in neg.items():
            st.markdown(f"- **{f}**: {v:.4f}")

    st.markdown("---")
    st.subheader("🎯 Recommendations")

    st.markdown(f"""
    Based on the analysis of **{len(X):,}** child records from Bunawan RHU
    (OPT 2024 + OPT 2025) using **{len(selected_features)}** predictors:

    1. **For RHU Health Workers & BNS:**
       - Deploy the **{best_model_f1}** model as a preliminary screening tool.
       - Prioritize follow-up on children flagged as high-risk for proactive
         nutrition counselling and supplemental feeding enrollment.
       - Use the feature-importance ranking to focus on the most influential
         anthropometric and demographic signals.

    2. **For Local Government Units (LGUs):**
       - Allocate supplemental feeding budgets to barangays with the highest
         predicted malnutrition prevalence.
       - Align interventions with the **Philippine Plan of Action for Nutrition
         (PPAN) 2023–2028** and Executive Order No. 70 (Whole-of-Nation Approach).
       - Use the dashboard's barangay-level breakdown for evidence-based
         programming.

    3. **For Parents & Guardians:**
       - Participate in scheduled OPT and BNS counselling sessions.
       - Maintain consistent weight and height monitoring, especially for
         children aged 0–59 months.
       - Seek early intervention at the first sign of stunting or wasting.

    4. **For Future Researchers:**
       - Extend with **XGBoost, Gradient Boosting, or Neural Networks**.
       - Apply **SMOTE** to address class imbalance if present.
       - Include **SHAP values** for deeper interpretability.
       - Integrate **household, maternal, and WASH variables** not present
         in the OPT dataset for richer modelling.

    5. **Limitations:**
       - Cross-sectional design — individual change over time not modelled.
       - Only OPT-recorded variables are used (no socioeconomic data).
       - Findings are specific to Bunawan and require re-validation elsewhere.
       - Dashboard is static and requires manual re-execution on new data.
    """)

    st.markdown("---")
    st.subheader("📥 Download Results")

    csv = metrics_table.to_csv().encode('utf-8')
    st.download_button("📊 Download Metrics CSV", csv,
                       "malnutrition_model_metrics.csv", "text/csv")

    preds = pd.DataFrame({'Actual': y_test.map({0: 'Not Malnourished', 1: 'Malnourished'})})
    for name, r in results.items():
        preds[f'{name} Predicted'] = pd.Series(r['y_pred']).map(
            {0: 'Not Malnourished', 1: 'Malnourished'}
        )
    st.download_button("📄 Download Predictions CSV",
                       preds.to_csv(index=False).encode('utf-8'),
                       "malnutrition_predictions.csv", "text/csv")


# Footer
st.markdown("---")
st.caption("🎓 Capstone Dashboard • Predictive Analysis of Child Malnutrition in Bunawan, "
           "Agusan del Sur Using Data Mining Techniques • "
           "Simbajon, E.J.A. & Lumsad, J.B.O. • Agusan del Sur State University • 2026")