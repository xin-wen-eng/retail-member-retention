# Retail Member Retention Predictor

Churn prediction and member segmentation for a simulated retail membership dataset (10,000 records).  
**Stack:** Python · scikit-learn · Pandas · Power BI

---

## Key Results

| Metric | Value |
|---|---|
| Model Accuracy | **82%** |
| Members Analyzed | 10,000 |
| At-Risk High-Value Members Identified | ~2,300 |
| Estimated Revenue Recovery Opportunity | **$450K** |

**Business Insights:**
- Credit card holders show **30% higher retention** vs non-holders
- High-frequency visitors have **50% higher renewal rates** vs low-frequency

---

## Project Structure

```
├── generate_data.py       # Simulates 10,000 member records
├── model.py               # RFM scoring + K-Means segmentation + Random Forest churn model
├── dashboard_export.py    # Exports 4 CSVs for Power BI dashboard pages
├── requirements.txt
├── data/
│   ├── members_raw.csv        (generated)
│   └── members_scored.csv     (generated)
└── outputs/
    ├── model_report.txt
    ├── segment_summary.csv
    ├── dash_overview.csv
    ├── dash_segments.csv
    ├── dash_churn_drivers.csv
    └── dash_recommendations.csv
```

---

## How to Run

```bash
pip install -r requirements.txt
mkdir data outputs

# Step 1: Generate simulated data
python generate_data.py

# Step 2: Train model + segment members
python model.py

# Step 3: Export dashboard tables
python dashboard_export.py
```

---

## Methodology

### 1. Data Simulation
10,000 member records with realistic churn probability driven by credit card ownership, visit frequency, and spending tier.

### 2. RFM Scoring
Each member scored on Recency (days since last visit), Frequency (annual visits), and Monetary (annual spend) — quintile-ranked 1–5.

### 3. K-Means Segmentation (k=5)
Members clustered into: **Champions · Loyal · Potential · At Risk · Hibernating**

### 4. Churn Prediction
Random Forest (200 trees, class-balanced) trained on RFM features + behavioral attributes.  
82% accuracy, evaluated on 20% holdout set.

### 5. Power BI Dashboard (4 pages)
| Page | Content |
|---|---|
| Executive Overview | KPI summary, churn rate, at-risk count |
| Member Segmentation | Segment profiles, revenue at risk by cluster |
| Churn Drivers | Impact of credit card, visit frequency, tier |
| Recommendations | Targeted actions with estimated revenue impact |

---

## Business Recommendations

1. **Credit Card Promotion** → Target at-risk Gold/Silver members without cards; card adoption reduces churn ~30%  
2. **Engagement Campaign** → Target at-risk low-frequency visitors; increasing visit cadence correlates with +50% renewal  
3. **VIP Retention Program** → Priority outreach to top 15% spenders facing churn risk; highest ROI intervention
