# ED Revisit Risk Prediction Model

Predictive ML model to identify Medicaid members at high risk of **30-day Emergency Department revisits**, enabling proactive care coordinator outreach within 24-48 hours of discharge.

## Problem

Approximately 1 in 6 ED discharges results in a return visit within 30 days, costing ~$3,200 per revisit. Care coordination teams need to know **who to prioritize** for follow-up outreach — phone calls, appointment scheduling, medication checks.

## Approach

- Analyzed **8,000+ member records** across demographics, diagnoses (ICD-10), ED visits, and pharmacy claims
- Engineered features from chronic condition flags, visit frequency patterns, medication adherence, and care gaps
- Built classification models to flag high-risk members at time of discharge
- Optimized for **recall** — better to over-flag than miss a high-risk patient

## Tech Stack

- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **Scikit-learn** — Logistic Regression, Random Forest, XGBoost
- **Feature Engineering** — ICD-10 code grouping, temporal visit patterns, pharmacy gaps
- **Healthcare Analytics** — Medicaid claims data, ED utilization, chronic disease modeling

## Data

Analysis covers a 150,000-member Medicaid population:
- `members.csv` — Demographics and enrollment
- `ed_visits.csv` — Emergency department encounters
- `diagnoses.csv` — ICD-10 diagnosis codes
- `medications.csv` — Pharmacy claims and fills

## Impact

Each prevented revisit saves ~**$3,200** in direct costs while improving patient outcomes through timely intervention.

## Author

**Deepika Ghotra** — MS Data Science, University of Maryland  
[Portfolio](https://deepikag8.github.io) · [LinkedIn](https://linkedin.com/in/deepika-ghotra-0126)
