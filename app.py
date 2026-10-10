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

# Page configuration
st.set_page_config(
    page_title="Child Malnutrition Prediction Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        padding: 0.8rem 0;
        border-bottom: 3px solid #1f4e79;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h2 { color: white; margin: 0; font-size: 1.8rem; }
    .metric-card p { color: white; margin: 0; opacity: 0.9; font-size: 0.9rem; }
    .phase-header {
        background-color: #1f4e79;
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# WHO classification constants
MALNOURISHED_CODES = {'UW', 'SUW', 'ST', 'SST', 'MW', 'SW'}
NON_MALNOURISHED_CODES = {'N', 'OW', 'OB', 'T'}
MISSING_TOKENS = {'', '-', '--', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '?', '.', 'N.D.', 'ND'}
EXPECTED_BARANGAYS = {
    'MAMBALILI', 'BUNAWAN BROOK', 'SAN TEODORO', 'POBLACION', 'CONSUELO',
    'SAN ANDRES', 'IMELDA', 'NUEVA ERA', 'SAN MARCOS', 'LIBERTAD',
    'HAGNAYA', 'BABADAN'
}
TARGET_RECORD_COUNT = 317

# Phase 1 - Load the OPT workbook
@st.cache_data
def load_data():
    search_paths = ['.', '/content/data']
    files = []
    for path in search_paths:
        if os.path.isdir(path):
            files += glob.glob(os.path.join(path, '*.xlsx'))
            files += glob.glob(os.path.join(path, '*.xls'))
            files += glob.glob(os.path.join(path, '*.csv'))

    if not files:
        return None, None

    preferred = [f for f in files
                 if any(k in os.path.basename(f).lower()
                        for k in ('nutritional', 'rhu', 'opt'))]
    target_file = preferred[0] if preferred else files[0]

    try:
        if target_file.lower().endswith('.csv'):
            df = pd.read_csv(target_file, encoding='utf-8-sig', low_memory=False)
            df['_source_sheet'] = 'Unknown'
        else:
            xls = pd.ExcelFile(target_file)
            all_sheets = []
            for s in xls.sheet_names:
                s_lower = s.lower()
                if 'opt' in s_lower or '202' in s_lower:
                    try:
                        sheet_df = pd.read_excel(xls, sheet_name=s, header=None)
                        header_idx = None
                        for i in range(min(10, len(sheet_df))):
                            row_vals = [str(v).strip().upper()
                                        for v in sheet_df.iloc[i].tolist()]
                            if any(v in ('NO.', 'NO') for v in row_vals):
                                header_idx = i
                                break
                        if header_idx is None:
                            header_idx = 3
                        header_row = sheet_df.iloc[header_idx].tolist()
                        new_cols = []
                        for j, c in enumerate(header_row):
                            c_str = str(c).replace('\n', ' ').strip()
                            if pd.isna(c) or c_str.lower() in ('nan', 'none', ''):
                                c_str = "col_" + str(j)
                            new_cols.append(c_str)
                        body = sheet_df.iloc[header_idx + 1:].copy()
                        body.columns = new_cols
                        body = body.reset_index(drop=True)
                        body['_source_sheet'] = s.strip()
                        all_sheets.append(body)
                    except Exception as e:
                        st.warning("Could not read sheet '" + str(s) + "': " + str(e))
                        continue
            if not all_sheets:
                return None, None
            df = pd.concat(all_sheets, ignore_index=True, sort=False)
    except Exception as e:
        st.error("Error reading " + str(target_file) + ": " + str(e))
        return None, None

    return df, os.path.basename(target_file)

# Cleaning pipeline
def clean_data(df):
    report = {}
    df = df.copy()
    report['rows_initial'] = len(df)
    report['cols_initial'] = len(df.columns)

    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(list(MISSING_TOKENS), np.nan)
            df[col] = df[col].replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan})

    no_col = None
    for c in df.columns:
        if str(c).strip().upper() in ('NO.', 'NO'):
            no_col = c
            break
    if no_col:
        df['_no_numeric'] = pd.to_numeric(df[no_col], errors='coerce')
        dropped_non_records = int(df['_no_numeric'].isna().sum())
        df = df[df['_no_numeric'].notna()].copy()
        df = df.drop(columns=['_no_numeric'])
        report['rows_dropped_non_records'] = dropped_non_records
    else:
        report['rows_dropped_non_records'] = 0

    df = df.dropna(how='all').dropna(axis=1, how='all')

    meta_cols = [c for c in df.columns if c.startswith('_')]
    check_cols = [c for c in df.columns if c not in meta_cols]
    before_dupes = len(df)
    df = df.drop_duplicates(subset=check_cols, keep='first').copy()
    report['rows_dropped_duplicates'] = int(before_dupes - len(df))

    report['rows_after_cleaning'] = len(df)
    report['cols_after_cleaning'] = len(df.columns)

    return df, report

# Fuzzy column lookup
def find_col(df, *candidates):
    cols_norm = {}
    for c in df.columns:
        key = str(c).lower().replace(' ', '').replace('_', '').replace('-', '')
        cols_norm[key] = c
    for cand in candidates:
        key = cand.lower().replace(' ', '').replace('_', '').replace('-', '')
        if key in cols_norm:
            return cols_norm[key]
    return None

# Imputation and normalization
def impute_and_prepare(df):
    df = df.copy()
    rep = {}

    sex_col = find_col(df, 'SEX')
    age_col = find_col(df, 'AGE (Months)', 'AGE', 'age (months)')
    weight_col = find_col(df, 'WEIGHT (kg)', 'WEIGHT', 'weight (kg)')
    height_col = find_col(df, 'HEIGHT (cm)', 'HEIGHT', 'height (cm)')
    brgy_col = find_col(df, 'BARANGAY')
    waz_col = find_col(df, 'WEIGHT-FOR-AGE (WAZ)', 'WAZ')
    haz_col = find_col(df, 'HEIGHT-FOR-AGE (HAZ)', 'HAZ')
    whz_col = find_col(df, 'WEIGHT-FOR-HEIGHT (WHZ)', 'WHZ')

    for col in [age_col, weight_col, height_col]:
        if col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if weight_col and height_col:
        before = len(df)
        df = df[~(df[weight_col].isna() & df[height_col].isna())].copy()
        rep['rows_dropped_no_anthropometry'] = int(before - len(df))
    else:
        rep['rows_dropped_no_anthropometry'] = 0

    if all([waz_col, haz_col, whz_col]):
        before = len(df)
        df = df[~(df[waz_col].isna() & df[haz_col].isna() & df[whz_col].isna())].copy()
        rep['rows_dropped_no_status'] = int(before - len(df))
    else:
        rep['rows_dropped_no_status'] = 0

    rep['median_imputed'] = {}
    for col, name in [(age_col, 'age_months'),
                      (weight_col, 'weight_kg'),
                      (height_col, 'height_cm')]:
        if col:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            rep['median_imputed'][name] = n_missing

    rep['mode_imputed'] = {}
    for col, name in [(sex_col, 'sex'), (brgy_col, 'barangay')]:
        if col:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'UNKNOWN'
                df[col] = df[col].fillna(mode_val)
            rep['mode_imputed'][name] = n_missing

    if sex_col:
        df[sex_col] = df[sex_col].astype(str).str.strip().str.upper()
    if brgy_col:
        df[brgy_col] = df[brgy_col].astype(str).str.strip().str.upper()

    return df, rep

# Build binary target from WHO indicator codes
def derive_malnutrition_label(df):
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

# Build feature matrix and target vector
def build_features(df):
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
    if sex_col:
        feature_frame['sex'] = df[sex_col].astype(str).str.strip().str.upper()
    if age_col:
        feature_frame['age_months'] = pd.to_numeric(df[age_col], errors='coerce')
    if weight_col:
        feature_frame['weight_kg'] = pd.to_numeric(df[weight_col], errors='coerce')
    if height_col:
        feature_frame['height_cm'] = pd.to_numeric(df[height_col], errors='coerce')
    if brgy_col:
        feature_frame['barangay'] = df[brgy_col].astype(str).str.strip().str.upper()

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

# Train the four classifiers
def train_models(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Naive Bayes': GaussianNB(),
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

# Compute the four evaluation metrics
def compute_metrics(y_true, y_pred):
    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, zero_division=0)
    }

# Load and clean
df_raw, filename = load_data()

st.markdown('<div class="main-header">Child Malnutrition Prediction Dashboard</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predictive Analysis of Child Malnutrition in Bunawan, '
            'Agusan del Sur Using Data Mining Techniques<br>'
            '<i>Decision Tree &middot; Random Forest &middot; Naive Bayes &middot; Logistic Regression</i></div>',
            unsafe_allow_html=True)

if df_raw is None:
    st.error("No dataset found. Please upload nutritional-status-RHU-1.xlsx to the repo root.")
    st.stop()

df_cleaned, cleaning_report = clean_data(df_raw)
df_prepared, imputation_report = impute_and_prepare(df_cleaned)

X, y, df_clean, waz_col, haz_col, whz_col = build_features(df_prepared)

if X is None or len(X) < 10:
    st.error("Could not build a valid dataset. Check that the Excel file has columns: "
             "SEX, AGE (Months), WEIGHT (kg), HEIGHT (cm), BARANGAY, and WAZ/HAZ/WHZ.")
    st.stop()

# Sidebar controls
st.sidebar.header("Model Configuration")
st.sidebar.info("Total records after cleaning: " + str(len(X)))

available_features = list(X.columns)
selected_features = st.sidebar.multiselect(
    "Select predictor variables:",
    options=available_features,
    default=available_features
)

if len(selected_features) < 2:
    st.warning("Please select at least 2 predictor variables from the sidebar.")
    st.stop()

test_size = st.sidebar.slider("Test size (per Chapter 3: 80/20 split)",
                              0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

X_sel = X[selected_features]

results, scaler, X_train, X_test, y_train, y_test = train_models(
    X_sel, y, test_size=test_size, random_state=int(random_state)
)

metrics_table = pd.DataFrame({
    name: compute_metrics(r['y_test'], r['y_pred'])
    for name, r in results.items()
}).T

best_by_f1 = metrics_table['F1-Score'].idxmax()
best_by_recall = metrics_table['Recall'].idxmax()
metrics_table['F1+Recall'] = metrics_table['F1-Score'] + metrics_table['Recall']
best_combined = metrics_table['F1+Recall'].idxmax()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Phase 1 - Data Collection",
    "Phase 2-3 - Preprocessing & Features",
    "Phase 4-5 - Model Building & Evaluation",
    "Phase 6 - Prediction & Analysis",
    "Downloads"
])

# Tab 1 - Phase 1
with tab1:
    st.markdown('<div class="phase-header">Phase 1: Data Collection</div>',
                unsafe_allow_html=True)
    st.write("Source: Bunawan Rural Health Unit (RHU) - Operation Timbang (OPT) "
             "2024 and OPT 2025 records. Secondary data only; no primary collection.")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="metric-card"><p>Loaded File</p>'
                '<h2 style="font-size:1rem;">' + str(filename) + '</h2></div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><p>Raw Records</p>'
                '<h2>' + str(cleaning_report["rows_initial"]) + '</h2></div>',
                unsafe_allow_html=True)
    c3.markdown('<div class="metric-card"><p>Records After Cleaning</p>'
                '<h2>' + str(len(X)) + '</h2></div>',
                unsafe_allow_html=True)
    c4.markdown('<div class="metric-card"><p>Target (Chapter 3)</p>'
                '<h2>' + str(TARGET_RECORD_COUNT) + '</h2></div>',
                unsafe_allow_html=True)

    st.divider()
    st.subheader("Source Sheets")
    if '_source_sheet' in df_raw.columns:
        sheet_counts = df_raw['_source_sheet'].value_counts()
        st.dataframe(sheet_counts.rename('Records').to_frame(), use_container_width=True)

    st.divider()
    st.subheader("Barangays Covered")
    brgy_col = find_col(df_prepared, 'BARANGAY')
    if brgy_col:
        brgy_counts = df_prepared[brgy_col].value_counts()
        st.write("Distinct barangays detected: " + str(brgy_counts.nunique()))
        st.dataframe(brgy_counts.rename('Records').to_frame(), use_container_width=True)
        missing_brgys = EXPECTED_BARANGAYS - set(brgy_counts.index)
        if missing_brgys:
            st.warning("Barangays listed in Chapter 3 but not present in data: "
                       + str(sorted(missing_brgys)))

    st.divider()
    st.subheader("Raw Data Preview (First 5 Records)")
    st.dataframe(df_raw.head(), use_container_width=True)

# Tab 2 - Phase 2 and 3
with tab2:
    st.markdown('<div class="phase-header">Phase 2: Data Preprocessing</div>',
                unsafe_allow_html=True)
    st.write("Cleaning operations applied: whitespace stripping, missing-token normalization, "
             "duplicate removal, median imputation (continuous), mode imputation (categorical).")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows Removed (non-records)", cleaning_report['rows_dropped_non_records'])
    c2.metric("Duplicate Rows Removed", cleaning_report['rows_dropped_duplicates'])
    c3.metric("Rows Kept After Cleaning", cleaning_report['rows_after_cleaning'])

    st.divider()
    st.subheader("Missing Value Handling")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Continuous variables - median imputation:")
        st.dataframe(pd.DataFrame([imputation_report['median_imputed']]).T.rename(
            columns={0: 'Missing Values Imputed'}), use_container_width=True)
    with col2:
        st.write("Categorical variables - mode imputation:")
        st.dataframe(pd.DataFrame([imputation_report['mode_imputed']]).T.rename(
            columns={0: 'Missing Values Imputed'}), use_container_width=True)

    st.divider()
    st.subheader("Anthropometric Records Dropped")
    st.write("Records missing BOTH weight and height: "
             + str(imputation_report['rows_dropped_no_anthropometry']))
    st.write("Records missing ALL three nutritional status codes: "
             + str(imputation_report['rows_dropped_no_status']))

    st.markdown('<div class="phase-header">Phase 3: Feature Selection</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Predictors", len(selected_features))
    c2.metric("Training Set (80%)", len(X_train))
    c3.metric("Testing Set (20%)", len(X_test))

    st.write("Predictor variables used:")
    st.code(selected_features)

    st.write("Target variable: _malnourished (1 = Malnourished, 0 = Not Malnourished)")
    st.write("Class distribution:")
    dist = y.value_counts().rename({0: 'Not Malnourished', 1: 'Malnourished'})
    st.dataframe(dist.rename('Count').to_frame().assign(
        Percentage=lambda d: (d['Count'] / d['Count'].sum() * 100).round(2)
    ), use_container_width=True)

    st.divider()
    st.subheader("Encoded Feature Sample (First 5 Rows)")
    st.dataframe(X_sel.head(), use_container_width=True)

    st.divider()
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

# Tab 3 - Phase 4 and 5
with tab3:
    st.markdown('<div class="phase-header">Phase 4: Model Building</div>',
                unsafe_allow_html=True)
    st.write("Four supervised classification algorithms trained on the 80% training split:")
    st.markdown("- Decision Tree - interpretable rule-based classifier\n"
                "- Random Forest - ensemble method\n"
                "- Naive Bayes - probabilistic classifier\n"
                "- Logistic Regression - binary baseline classifier")

    st.markdown('<div class="phase-header">Phase 5: Model Evaluation</div>',
                unsafe_allow_html=True)

    st.subheader("Performance Metrics (Test Set)")
    st.dataframe(
        metrics_table.drop(columns='F1+Recall').style.format("{:.4f}")
        .highlight_max(axis=0, color='#c6efce'),
        use_container_width=True
    )

    st.divider()
    st.subheader("Best Model Selection (per Chapter 3)")
    st.info("Best by F1-Score: " + str(best_by_f1) + "  |  "
            "Best by Recall: " + str(best_by_recall) + "  |  "
            "Best by combined F1 + Recall: " + str(best_combined))

    st.divider()
    st.subheader("Confusion Matrices - All Algorithms")
    cols = st.columns(2)
    for i, (name, r) in enumerate(results.items()):
        with cols[i % 2]:
            cm = confusion_matrix(r['y_test'], r['y_pred'])
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Mal.', 'Mal.'],
                        yticklabels=['Not Mal.', 'Mal.'], ax=ax)
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.divider()
    st.subheader("Detailed Classification Reports")
    for name, r in results.items():
        with st.expander(name + " - Classification Report"):
            rep = classification_report(r['y_test'], r['y_pred'],
                                        target_names=['Not Malnourished', 'Malnourished'],
                                        output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

    st.divider()
    st.subheader("Cross-Validation (Stratified K-Fold)")
    k = st.slider("Number of folds", 3, 10, 5)
    X_scaled = StandardScaler().fit_transform(X_sel)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    cv_models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    cv_scores = {name: cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
                 for name, model in cv_models.items()}
    cv_df = pd.DataFrame(cv_scores)
    cv_df.index = ["Fold " + str(i + 1) for i in range(k)]
    cv_df.loc['Mean'] = cv_df.mean()
    cv_df.loc['Std'] = cv_df.iloc[:-2].std()
    st.dataframe(cv_df.round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    cv_df.iloc[:-2].plot(kind='bar', ax=ax,
                         color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
    ax.set_ylabel("Accuracy")
    ax.set_title(str(k) + "-Fold Cross-Validation Accuracy")
    ax.legend(loc='lower right', ncol=4)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Tab 4 - Phase 6
with tab4:
    st.markdown('<div class="phase-header">Phase 6: Prediction and Analysis</div>',
                unsafe_allow_html=True)
    st.write("Apply the best model to classify malnutrition risk, identify at-risk children "
             "per barangay, and interpret significant predictor variables.")

    st.divider()
    st.subheader("Significant Predictor Variables (Feature Importance)")

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
        st.success("Top predictor across all models: " + str(imp_df.index[0]))

    st.divider()
    st.subheader("At-Risk Children per Barangay")
    st.write("Applying the best-performing model to the full cleaned dataset.")

    best_model = results[best_combined]['model']
    X_all_scaled = scaler.transform(X_sel)
    full_predictions = best_model.predict(X_all_scaled)

    pred_df = df_clean.copy()
    pred_df['_predicted_risk'] = full_predictions
    pred_df['_predicted_label'] = pred_df['_predicted_risk'].map(
        {1: 'At Risk', 0: 'Not At Risk'}
    )

    brgy_col = find_col(pred_df, 'BARANGAY')
    brgy_summary = None
    if brgy_col:
        brgy_summary = pred_df.groupby(brgy_col)['_predicted_risk'].agg(
            Total='count', At_Risk='sum'
        )
        brgy_summary['Not_At_Risk'] = brgy_summary['Total'] - brgy_summary['At_Risk']
        brgy_summary['At_Risk_Percent'] = (
            brgy_summary['At_Risk'] / brgy_summary['Total'] * 100
        ).round(2)
        brgy_summary = brgy_summary.sort_values('At_Risk', ascending=False)
        st.dataframe(brgy_summary, use_container_width=True)

        fig, ax = plt.subplots(figsize=(11, 6))
        brgy_summary[['At_Risk', 'Not_At_Risk']].plot(
            kind='barh', stacked=True, ax=ax,
            color=['#e74c3c', '#2ecc71']
        )
        ax.set_xlabel("Number of Children")
        ax.set_title("Predicted Malnutrition Risk per Barangay (" + str(best_combined) + ")")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("Predicted Labels - Sample Records")
    display_cols = [c for c in pred_df.columns if not c.startswith('_')]
    st.dataframe(
        pred_df[display_cols + ['_predicted_label']].head(15),
        use_container_width=True
    )

    st.divider()
    st.subheader("Predicted Risk Distribution")
    risk_counts = pred_df['_predicted_label'].value_counts()
    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
               colors=['#e74c3c', '#2ecc71'][:len(risk_counts)], startangle=90)
        ax.set_title("Predicted Nutritional Risk")
        st.pyplot(fig)
        plt.close()
    with c2:
        st.dataframe(risk_counts.rename('Count').to_frame(), use_container_width=True)

# Tab 5 - Downloads
with tab5:
    st.header("Download Results")

    csv = metrics_table.drop(columns='F1+Recall').to_csv().encode('utf-8')
    st.download_button("Download Model Metrics CSV", csv,
                       "malnutrition_model_metrics.csv", "text/csv")

    preds = pd.DataFrame({'Actual': y_test.map({0: 'Not Malnourished', 1: 'Malnourished'})})
    for name, r in results.items():
        preds[name + ' Predicted'] = pd.Series(r['y_pred']).map(
            {0: 'Not Malnourished', 1: 'Malnourished'}
        )
    st.download_button("Download Test Set Predictions CSV",
                       preds.to_csv(index=False).encode('utf-8'),
                       "malnutrition_predictions.csv", "text/csv")

    if brgy_summary is not None:
        st.download_button("Download Barangay Risk Summary CSV",
                           brgy_summary.to_csv().encode('utf-8'),
                           "barangay_risk_summary.csv", "text/csv")

# Footer
st.divider()
st.caption("Capstone Dashboard | Predictive Analysis of Child Malnutrition in Bunawan, "
           "Agusan del Sur Using Data Mining Techniques | "
           "Simbajon, E.J.A. & Lumsad, J.B.O. | Agusan del Sur State University | 2026")
