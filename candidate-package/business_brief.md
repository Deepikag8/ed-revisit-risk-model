# Undine Health Partners — Analytics Request

**From:** Dana Cortez, VP of Clinical Operations
**To:** Analytics Team
**Date:** January 6, 2025
**Re:** Predicting 30-Day ED Revisits for Care Coordinator Outreach

---

Team,

I'm reaching out because we need your help tackling one of our most persistent and expensive problems: members who visit the emergency department, get discharged, and then end up back in the ED within 30 days.

## The Problem

Undine Health Partners serves approximately 150,000 Medicaid members across our network. Over the past year we've seen a consistent pattern: roughly 1 in 6 ED discharges results in a return visit to the emergency department within 30 days. Each of those revisits costs an average of **$3,200**, and more importantly, it signals that something fell through the cracks in that member's care. Maybe they didn't have a follow-up appointment, maybe a prescription wasn't filled, maybe they didn't have a primary care provider to begin with. Whatever the reason, our client is failing those members and paying for it.

The care coordination team is ready to do proactive outreach — phone calls, follow-up scheduling, medication checks — but they can't call every single member who walks out of the ED. They need to know **who to prioritize**.

## What We Need

We'd like a **predictive model** that can flag members at high risk of returning to the ED within 30 days. Ideally, this would run at the time of discharge so the care coordinators can begin outreach within 24–48 hours for the highest-risk members.

We're not looking for a perfect model. We know this is a hard problem. But even a model that helps us focus our limited coordinator bandwidth on the right members would be a major win.

## The Data

We've pulled the following datasets covering our ED population for calendar year 2024:

- **`members.csv`** — Demographics and enrollment information for members who had at least one ED visit during the study period (~8,000 members).
- **`ed_visits.csv`** — Individual ED encounter records, including visit date, diagnosis, discharge status, cost, and whether the member returned within 30 days (this is the outcome we want to predict).
- **`diagnoses.csv`** — Historical diagnosis records across all care settings (ED, inpatient, outpatient, telehealth) for these members.
- **`medications.csv`** — Active medication records including drug name, therapeutic class, and days supplied.

## What Success Looks Like

If this works, we'd integrate the risk scores into our care coordination workflow. Coordinators would get a daily list of recently discharged members ranked by risk, and they'd start at the top. We estimate that even a modest reduction in avoidable revisits — say 10–15% — could save the plan $1.5M–$2M annually while improving outcomes for our most vulnerable members.

Looking forward to seeing what you find.

Best,
Dana Cortez
VP of Clinical Operations, Undine Health Partners
