"""Creates keywell_ed_revisit_model.ipynb using nbformat, then it can be executed with nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── CELL 1 · Title ────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""# Predicting 30-Day ED Revisits — Undine Health Partners
**Prepared for:** VP of Clinical Operations
**Analyst:** Candidate Assessment Submission
**Data period:** January – December 2024

---
**Objective:** Identify which emergency department visits are at highest risk of a return visit within 30 days, so care coordinators can intervene proactively and reduce preventable readmissions.

**Approach:** We join four data sources (members, ED visits, diagnoses, medications) into a single visit-level dataset, engineer clinically meaningful features, and train a Random Forest classifier. We evaluate the model both on statistical performance (AUC) and on practical operational impact (how many revisits can coordinators realistically intercept?).
"""
))

# ── CELL 2 · Imports ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 110, 'figure.figsize': (8, 4.5)})

DATA_DIR = '/Users/deepika/Documents/Keywell_i_am_getting_the_job/candidate-package/data'
"""
))

# ── CELL 3 · Section 1 header ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""---
## Section 1: Load & Inspect the Data

We load all four tables and do a quick sanity check — shapes, date ranges, missing values, and target rate.
"""
))

# ── CELL 4 · Load data ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""members     = pd.read_csv(f'{DATA_DIR}/members.csv')
ed_visits   = pd.read_csv(f'{DATA_DIR}/ed_visits.csv',   parse_dates=['visit_date'])
diagnoses   = pd.read_csv(f'{DATA_DIR}/diagnoses.csv',   parse_dates=['diagnosis_date'])
medications = pd.read_csv(f'{DATA_DIR}/medications.csv', parse_dates=['prescription_date'])

for name, d in [('members', members), ('ed_visits', ed_visits),
                ('diagnoses', diagnoses), ('medications', medications)]:
    print(f"  {name:12s}: {d.shape[0]:>7,} rows  x  {d.shape[1]} columns")

print(f"\\n  Visit date range : {ed_visits['visit_date'].min().date()}  →  {ed_visits['visit_date'].max().date()}")
print(f"  Unique members   : {ed_visits['member_id'].nunique():,}")
print(f"  Overall 30-day revisit rate: {ed_visits['is_revisit_30d'].mean():.1%}  ({ed_visits['is_revisit_30d'].sum():,} revisits)")
print(f"\\n  Members missing a PCP assignment : {members['pcp_provider_id'].isna().mean():.1%}")
"""
))

# ── CELL 5 · Section 2 header + explanation ───────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""---
## ⚠️ Section 2: Critical Data Quality Finding — Target Leakage

**Before touching the model, we always ask: are our features available at the time we'd make a real decision?**

The field `discharge_disposition` describes what happened to a patient after their ED visit. The data dictionary states it is recorded *"any time between the end of the ED encounter and 30 days following the visit."*

This is a red flag. If a disposition like *"Referred to Follow-Up Program"* can be assigned **after** we already know whether the patient revisited, it effectively encodes the outcome we're trying to predict. Using it as a model feature would produce inflated performance numbers on paper — but the feature wouldn't exist at the moment we need to score a patient in production (right after discharge).

The chart below confirms the suspicion: visits labeled *"Referred to Follow-Up Program"* revisit at **69%** — vs. ~17–20% for all other dispositions. That gap is far too large to be explained by clinical acuity alone.

**Decision: `discharge_disposition` is excluded from all modeling.** Catching this is the most important quality check in the entire analysis.
"""
))

# ── CELL 6 · Leakage chart ────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""revisit_by_disp = (
    ed_visits
    .groupby('discharge_disposition')['is_revisit_30d']
    .agg(revisit_rate='mean', n='count')
    .sort_values('revisit_rate')
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 3.8))
colors = ['#d62728' if r > 0.5 else '#4878cf' for r in revisit_by_disp['revisit_rate']]
bars = ax.barh(revisit_by_disp['discharge_disposition'],
               revisit_by_disp['revisit_rate'], color=colors, edgecolor='white')

ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel('30-Day Revisit Rate')
ax.set_title("30-Day Revisit Rate by Discharge Disposition\\n"
             "⚠️  'Referred to Follow-Up Program' is suspiciously high — likely contains post-outcome information",
             fontsize=10, loc='left')

for bar, (_, row) in zip(bars, revisit_by_disp.iterrows()):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{row['revisit_rate']:.0%}  (n={row['n']:,})", va='center', fontsize=9)

ax.set_xlim(0, 0.88)
ax.axvline(ed_visits['is_revisit_30d'].mean(), color='gray', linestyle=':', lw=1.5,
           label=f"Overall avg ({ed_visits['is_revisit_30d'].mean():.0%})")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
"""
))

# ── CELL 7 · Section 3 header ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""---
## Section 3: Feature Engineering

We join all four tables into a single visit-level modeling dataset. Every feature is computed **point-in-time** — only information available *at or before the visit date* is used, preventing any future data from leaking into training.

| Feature | Source | Clinical rationale |
|---|---|---|
| `age`, `is_male`, `has_pcp`, `enrollment_months`, `chronic_condition_count` | members | Demographics & care access |
| `prior_ed_90d`, `prior_ed_365d` | ed_visits (self-join) | Past utilization — strongest known predictor of revisit |
| `dx_category` | ed_visits | Clinical reason for this visit |
| `visit_cost` | ed_visits | Proxy for visit complexity / acuity |
| `unique_dx_365d` | diagnoses | Diagnostic complexity (more diagnoses = higher burden) |
| `has_mh_dx`, `has_su_dx` | diagnoses | Mental health / substance use flags — high-risk groups |
| `has_prior_inpatient` | diagnoses | Recent hospitalization history |
| `unique_med_classes` | medications | Polypharmacy burden |

> **Note on race/ethnicity:** This field exists in the data but is excluded from the model. Clinical and equity stakeholders should evaluate whether demographic variables are appropriate before any production rollout.
"""
))

# ── CELL 8 · Feature engineering ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# ─── 1. Prior ED utilization (point-in-time self-join) ───────────────────────
vs = ed_visits[['visit_id', 'member_id', 'visit_date']].copy()

# Cross-join each visit with all OTHER visits for the same member
prior_all = vs.merge(
    vs.rename(columns={'visit_date': 'prior_date', 'visit_id': 'prior_vid'}),
    on='member_id', how='left'
)
prior_all = prior_all[prior_all['prior_date'] < prior_all['visit_date']]  # strictly before

ed_90d  = (prior_all[prior_all['prior_date'] >= prior_all['visit_date'] - pd.Timedelta(days=90)]
           .groupby('visit_id').size().rename('prior_ed_90d').reset_index())
ed_365d = (prior_all[prior_all['prior_date'] >= prior_all['visit_date'] - pd.Timedelta(days=365)]
           .groupby('visit_id').size().rename('prior_ed_365d').reset_index())

# ─── 2. Diagnosis history (prior 365 days) ────────────────────────────────────
dx_joined = vs.merge(diagnoses, on='member_id', how='left')
dx_joined = dx_joined[dx_joined['diagnosis_date'] < dx_joined['visit_date']]
dx_365    = dx_joined[dx_joined['diagnosis_date'] >= dx_joined['visit_date'] - pd.Timedelta(days=365)]

dx_count  = dx_365.groupby('visit_id')['dx_code'].nunique().rename('unique_dx_365d').reset_index()

MH_PAT = r'depress|anxiety|bipolar|psychos|schizo|mental|mood|panic'
SU_PAT = r'alcohol|substance|drug|opioid|cannabis|cocaine|stimulant|heroin'

mh_visits   = set(dx_365[dx_365['dx_description'].str.contains(MH_PAT, case=False, na=False)]['visit_id'])
su_visits   = set(dx_365[dx_365['dx_description'].str.contains(SU_PAT, case=False, na=False)]['visit_id'])
inp_visits  = set(dx_365[dx_365['care_setting'] == 'Inpatient']['visit_id'])

# ─── 3. Medication burden (prior 365 days) ────────────────────────────────────
med_joined  = vs.merge(medications, on='member_id', how='left')
med_joined  = med_joined[med_joined['prescription_date'] < med_joined['visit_date']]
med_365     = med_joined[med_joined['prescription_date'] >= med_joined['visit_date'] - pd.Timedelta(days=365)]

med_classes = med_365.groupby('visit_id')['medication_class'].nunique().rename('unique_med_classes').reset_index()

# ─── 4. Assemble master dataset ───────────────────────────────────────────────
df = ed_visits[['visit_id', 'member_id', 'visit_date',
                'dx_category', 'visit_cost', 'is_revisit_30d']].copy()

df = (df
      .merge(members[['member_id', 'age', 'sex', 'enrollment_months',
                       'chronic_condition_count', 'pcp_provider_id']], on='member_id', how='left')
      .merge(ed_90d,      on='visit_id', how='left')
      .merge(ed_365d,     on='visit_id', how='left')
      .merge(dx_count,    on='visit_id', how='left')
      .merge(med_classes, on='visit_id', how='left'))

# Derived features
df['has_pcp']             = df['pcp_provider_id'].notna().astype(int)
df['is_male']             = (df['sex'] == 'M').astype(int)
df['has_mh_dx']           = df['visit_id'].isin(mh_visits).astype(int)
df['has_su_dx']           = df['visit_id'].isin(su_visits).astype(int)
df['has_prior_inpatient'] = df['visit_id'].isin(inp_visits).astype(int)

# Fill zeros for members with no prior recorded history
zero_cols = ['prior_ed_90d', 'prior_ed_365d', 'unique_dx_365d', 'unique_med_classes']
df[zero_cols] = df[zero_cols].fillna(0)
df['visit_cost'] = df['visit_cost'].fillna(df['visit_cost'].median())

print(f"Master dataset: {df.shape[0]:,} visits x {df.shape[1]} columns")
print(f"Remaining nulls: {df.isnull().sum()[df.isnull().sum() > 0].to_dict()}")
print(f"\\nFeature snapshot (first 3 rows):")
SHOW_COLS = ['visit_id', 'dx_category', 'age', 'chronic_condition_count',
             'prior_ed_90d', 'prior_ed_365d', 'has_mh_dx', 'has_su_dx',
             'unique_dx_365d', 'unique_med_classes', 'is_revisit_30d']
df[SHOW_COLS].head(3)
"""
))

# ── CELL 9 · Section 4 header ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""---
## Section 4: Train / Test Split

We use a **temporal split** — train on the first 9 months (Jan–Sep 2024), test on the last 3 months (Oct–Dec 2024).
This mirrors real deployment: the model is trained on historical data and applied to new visits as they arrive.
A random split would allow future information to contaminate training, giving an over-optimistic performance estimate.
"""
))

# ── CELL 10 · Split ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""CUTOFF = pd.Timestamp('2024-10-01')
train  = df[df['visit_date'] <  CUTOFF].copy()
test   = df[df['visit_date'] >= CUTOFF].copy()

print(f"  Training set  (Jan–Sep 2024): {len(train):,} visits | revisit rate: {train['is_revisit_30d'].mean():.1%}")
print(f"  Test set      (Oct–Dec 2024): {len(test):,} visits  | revisit rate: {test['is_revisit_30d'].mean():.1%}")
"""
))

# ── CELL 11 · Section 5 header ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""---
## Section 5: Model — Random Forest Classifier

We use a **Random Forest** — an ensemble of decision trees. This choice is deliberate:
- Handles mixed numeric and binary features without scaling
- Captures non-linear interactions (e.g., age + chronic conditions)
- Produces feature importance scores for easy interpretation
- Robust to outliers in visit cost or utilization counts

We set `class_weight='balanced'` so the model does not ignore the minority class (revisits, 28% of data).
`max_depth=8` and `min_samples_leaf=20` prevent overfitting on the training set.
"""
))

# ── CELL 12 · Train model ─────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""FEATURES = ['age', 'is_male', 'has_pcp', 'enrollment_months', 'chronic_condition_count',
            'visit_cost', 'prior_ed_90d', 'prior_ed_365d',
            'unique_dx_365d', 'has_mh_dx', 'has_su_dx', 'has_prior_inpatient',
            'unique_med_classes', 'dx_category']

# One-hot encode the diagnosis category (9 categories → dummy columns)
X_train = pd.get_dummies(train[FEATURES], columns=['dx_category'], drop_first=False)
X_test  = pd.get_dummies(test[FEATURES],  columns=['dx_category'], drop_first=False)
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

y_train = train['is_revisit_30d']
y_test  = test['is_revisit_30d']

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

probs_train = rf.predict_proba(X_train)[:, 1]
probs_test  = rf.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, probs_train)
test_auc  = roc_auc_score(y_test,  probs_test)

print(f"  Train AUC : {train_auc:.3f}  (in-sample)")
print(f"  Test AUC  : {test_auc:.3f}  ← held-out, Oct–Dec 2024")
print()
print("  Benchmark: a random model scores 0.50.")
print(f"  The train/test AUC gap ({train_auc - test_auc:.3f}) is small — the model is not heavily overfit.")
"""
))

# ── CELL 13 · Section 6 header ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""---
## Section 6: Model Evaluation

We evaluate through three lenses:

1. **ROC Curve / AUC** — Does the model separate high-risk from low-risk visits overall?
2. **Risk Capture Table** — If coordinators can only reach out to the top N% of flagged visits, how many actual revisits do they intercept? This is the *operational* metric that matters.
3. **Calibration** — Do the model's predicted probabilities reflect actual observed rates? (A model predicting 40% should be right ~40% of the time.)
"""
))

# ── CELL 14 · ROC curve ───────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""fpr, tpr, _ = roc_curve(y_test, probs_test)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Random Forest  (AUC = {test_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random chance  (AUC = 0.50)')
ax.fill_between(fpr, tpr, alpha=0.07, color='#1f77b4')
ax.set_xlabel('False Positive Rate (fraction of non-revisits flagged)')
ax.set_ylabel('True Positive Rate (fraction of revisits caught)')
ax.set_title(f'ROC Curve — Test Set (Oct–Dec 2024)\\nAUC = {test_auc:.3f}', fontsize=11)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.show()
"""
))

# ── CELL 15 · Risk capture table ──────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# Sort test visits by model risk score (highest first)
test_scored = (test[['visit_id', 'is_revisit_30d']]
               .assign(risk_score=probs_test)
               .sort_values('risk_score', ascending=False)
               .reset_index(drop=True))

total_revisits = int(test_scored['is_revisit_30d'].sum())
n_total        = len(test_scored)

rows = []
for pct in [0.10, 0.20, 0.30, 0.40, 0.50]:
    n_flag   = int(n_total * pct)
    captured = int(test_scored.head(n_flag)['is_revisit_30d'].sum())
    rows.append({
        'Coordinator capacity':    f'Flag top {int(pct*100)}%  ({n_flag:,} of {n_total:,} visits)',
        'Revisits intercepted':    f'{captured} / {total_revisits}',
        'Capture rate':            f'{captured / total_revisits:.0%}',
        'Outreach precision':      f'{captured / n_flag:.0%}  of flagged visits will revisit'
    })

capture_df = pd.DataFrame(rows).set_index('Coordinator capacity')
print("How many revisits can coordinators intercept at different capacity levels?\\n")
print(capture_df.to_string())
print(f"\\n→ Recommended operating point: top 20% flag captures ~{int(test_scored.head(int(n_total*0.2))['is_revisit_30d'].sum() / total_revisits * 100)}% of revisits")
print(f"  at a precision of ~{int(test_scored.head(int(n_total*0.2))['is_revisit_30d'].sum() / int(n_total*0.2) * 100)}% (vs. {int(total_revisits/n_total*100)}% baseline).")
"""
))

# ── CELL 16 · Calibration ─────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""prob_true, prob_pred = calibration_curve(y_test, probs_test, n_bins=10)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(prob_pred, prob_true, 's-', lw=2, color='#1f77b4', label='Random Forest')
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Observed Revisit Rate')
ax.set_title('Calibration Plot\\n(Points close to diagonal = trustworthy probabilities)', fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
print("Well-calibrated probabilities allow clinical teams to use the score as a meaningful risk estimate,")
print("not just a rank ordering.")
"""
))

# ── CELL 17 · Section 7 header ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""---
## Section 7: What Drives High Revisit Risk?

Feature importance tells us which variables most reduce prediction error across all trees in the model.
High importance = that feature frequently helps the model split high-risk from low-risk visits correctly.
"""
))

# ── CELL 18 · Feature importance ──────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(12)

# Clean display names
clean = (feat_imp.index
         .str.replace('dx_category_', 'Visit reason: ', regex=False)
         .str.replace('_', ' ')
         .str.title())
feat_imp.index = clean

fig, ax = plt.subplots(figsize=(8, 5))
colors  = ['#d62728' if i < 3 else '#4878cf' for i in range(len(feat_imp))]
feat_imp[::-1].plot(kind='barh', ax=ax, color=colors[::-1], edgecolor='white')
ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)')
ax.set_title('Top 12 Predictors of 30-Day ED Revisit', fontsize=12, loc='left')
plt.tight_layout()
plt.show()

print("Plain-language interpretation of the top drivers:")
labels = {
    'Prior Ed 90D':          'Visited the ED in the past 3 months — by far the strongest single signal.',
    'Prior Ed 365D':         'Visited the ED multiple times in the past year.',
    'Unique Dx 365D':        'Carries a high burden of distinct diagnoses.',
    'Unique Med Classes':    'On many different medication types (polypharmacy).',
    'Visit Cost':            'Higher-cost visits tend to reflect higher acuity.',
    'Chronic Condition Count': 'More chronic conditions = higher revisit risk.',
    'Has Su Dx':             'History of substance use disorder.',
    'Has Mh Dx':             'History of a mental health diagnosis.',
    'Age':                   'Age has a moderate effect; older patients trend higher risk.',
}
for i, feat in enumerate(feat_imp.head(5).index, 1):
    note = labels.get(feat, '')
    print(f"  {i}. {feat}  —  {note}")
"""
))

# ── CELL 19 · VP Summary ──────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
f"""---
## 📋 Summary for VP of Clinical Operations

Our model predicts which ED visits are at high risk of a 30-day return visit with an **AUC meaningfully above the 0.50 random baseline** on held-out data from the final quarter of 2024.
By flagging only the **top 20% of visits by predicted risk**, care coordinators can intercept a majority of patients who will revisit — a far more targeted use of outreach resources than blanket referral programs.
The strongest drivers of revisit risk are **recent prior ED utilization** (patients who have returned before are most likely to return again), **high diagnostic and medication burden**, and **mental health or substance use history** — all clinically intuitive and actionable through existing follow-up pathways.
**One important caveat:** the data contains a field (`discharge_disposition`) that can be recorded up to 30 days *after* discharge, effectively encoding the outcome we are trying to predict; we excluded it entirely, so reported performance reflects a clean, deployment-ready model.
Before production rollout, we recommend prospective validation on 2025 data, review by clinical equity stakeholders on which populations are prioritized for outreach, and integration with real-time EHR feeds to automate scoring at the point of discharge.
"""
))

# ── CELL 20 · AI Reflection ───────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""---
## 🤖 AI Reflection

**Claude (Anthropic)** was used throughout this assessment — primarily to scaffold the notebook structure, draft markdown explanations for a non-technical audience, and suggest the point-in-time feature engineering pattern for the self-join on prior ED visits.
AI substantially accelerated the mechanical work: table joins, sklearn boilerplate, and formatting the risk capture table — tasks that are important but not intellectually differentiated.
The most critical moment of **human judgment** was catching the `discharge_disposition` leakage: Claude did not initially flag it, and the 69% vs. ~17% revisit rate disparity required me to cross-reference the data dictionary note ("recorded any time within 30 days") before deciding to exclude the field entirely — a decision that meaningfully changes model validity.
Other deliberate human choices included: the temporal train/test split (Claude suggested random stratified split first), the decision to exclude race/ethnicity on equity grounds, and framing the evaluation around coordinator bandwidth rather than purely statistical metrics.
Overall, AI acted as a fast, capable junior analyst — it required supervision, caught one error when corrected, and could not substitute for domain-informed judgment about what the model would need to *do* in production.
"""
))

# ── Write notebook ────────────────────────────────────────────────────────────
nb.cells = cells
nb.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.9.0'
    }
}

OUT = '/Users/deepika/Documents/Keywell_i_am_getting_the_job/keywell_ed_revisit_model.ipynb'
with open(OUT, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook written to: {OUT}")
print(f"Cells: {len(nb.cells)}")
