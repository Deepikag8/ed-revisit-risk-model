# Data Dictionary

This document describes the four data files provided with the assessment.

---

## Table Relationships

All tables share `member_id` as the common join key.

- **`members.csv`** contains one row per member.
- **`ed_visits.csv`** contains one row per ED encounter. A member may have multiple visits.
- **`diagnoses.csv`** contains one row per diagnosis record across all care settings. A member may have many diagnoses over time.
- **`medications.csv`** contains one row per pharmacy fill. A member may have multiple fills of the same medication over time.

---

## `members.csv` — Member Demographics

| Field | Type | Description |
|-------|------|-------------|
| member_id | string | Unique member identifier. Primary key. |
| age | integer | Member age in years at the start of the study period. |
| sex | string | Member sex. |
| race_ethnicity | string | Self-reported race/ethnicity. |
| zip_code | string | 5-digit residential ZIP code. |
| pcp_provider_id | string | Assigned primary care provider identifier. May be null. |
| enrollment_months | integer | Consecutive months enrolled in the health plan as of study end. |
| chronic_condition_count | integer | Count of distinct chronic conditions on record. |

---

## `ed_visits.csv` — ED Encounter Records

| Field | Type | Description |
|-------|------|-------------|
| visit_id | string | Unique visit identifier. Primary key. |
| member_id | string | Member identifier. Foreign key to `members.csv`. |
| visit_date | date | Date of the ED encounter (YYYY-MM-DD). |
| primary_dx_code | string | ICD-10-CM code for the primary visit diagnosis. |
| dx_category | string | High-level diagnostic category. Values: Cardiovascular, Endocrine, Genitourinary, GI, Injury, Mental Health, Neurological, Respiratory, Substance Use. |
| discharge_disposition | string | Disposition recorded any time between the end of the ED encounter and 30 days following the visit. **Note:** some disposition values reflect actions taken after the clinical decision to discharge (e.g., referral routing) rather than the clinical state at discharge. Values: Discharged Home, Discharged Home with Services, Left Against Medical Advice, Referred to Follow-Up Program (this is the current non-ML rules-based outreach approach), Transferred to Another Facility. |
| ed_facility_id | string | Identifier for the ED facility where the visit occurred. |
| visit_cost | float | Total billed cost of the visit in USD. |
| is_revisit_30d | integer | **Target variable.** 1 if the member had another ED visit within 30 days, 0 otherwise. |

---

## `diagnoses.csv` — Historical Diagnosis Records

| Field | Type | Description |
|-------|------|-------------|
| member_id | string | Member identifier. Foreign key to `members.csv`. |
| dx_code | string | ICD-10-CM diagnosis code. |
| dx_description | string | Human-readable diagnosis description. |
| diagnosis_date | date | Date the diagnosis was recorded (YYYY-MM-DD). |
| care_setting | string | Setting where the diagnosis was recorded. Values: ED, Inpatient, Outpatient, Telehealth. |

---

## `medications.csv` — Pharmacy Fill Records

| Field | Type | Description |
|-------|------|-------------|
| member_id | string | Member identifier. Foreign key to `members.csv`. |
| medication_name | string | Generic drug name. |
| medication_class | string | Therapeutic class. Values: Anticoagulant, Anticonvulsant, Antidepressant, Antidiabetic, Antihypertensive, Antipsychotic, Anxiolytic, Bronchodilator, Muscle Relaxant, NSAID, Opioid, PPI, Statin, Thyroid. |
| prescription_date | date | Date the prescription was filled (YYYY-MM-DD). |
| days_supply | integer | Number of days supplied per fill. |
