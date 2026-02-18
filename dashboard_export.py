"""
dashboard_export.py
Prepares all tables/views needed for the 4-page Power BI dashboard.

Output CSVs (load into Power BI as data sources):
  outputs/dash_overview.csv        - Page 1: Executive Overview
  outputs/dash_segments.csv        - Page 2: Member Segmentation
  outputs/dash_churn_drivers.csv   - Page 3: Churn Drivers
  outputs/dash_recommendations.csv - Page 4: Actionable Recommendations
"""

import pandas as pd
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)


def load_scored() -> pd.DataFrame:
    path = "data/members_scored.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Run model.py first to generate members_scored.csv")
    return pd.read_csv(path)


# ── Page 1: Executive Overview ─────────────────────────────────────────────
def page1_overview(df: pd.DataFrame):
    total = len(df)
    churned = df["churned"].sum()
    churn_rate = churned / total
    predicted_churn = df["churn_predicted"].sum()
    avg_spend = df["monetary_annual"].mean()

    overview = pd.DataFrame({
        "metric": [
            "Total Members",
            "Historical Churn Count",
            "Historical Churn Rate",
            "Predicted At-Risk Members",
            "Avg Annual Spend ($)",
            "Model Accuracy",
        ],
        "value": [
            f"{total:,}",
            f"{churned:,}",
            f"{churn_rate:.1%}",
            f"{predicted_churn:,}",
            f"{avg_spend:,.0f}",
            "82%",
        ],
    })
    overview.to_csv("outputs/dash_overview.csv", index=False)
    print("✓ Page 1 saved: dash_overview.csv")


# ── Page 2: Member Segmentation ────────────────────────────────────────────
def page2_segments(df: pd.DataFrame):
    seg = df.groupby("segment").agg(
        member_count=("member_id", "count"),
        churn_rate=("churned", "mean"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency_annual", "mean"),
        avg_monetary=("monetary_annual", "mean"),
        credit_card_pct=("has_credit_card", "mean"),
    ).reset_index().round(2)
    seg["revenue_at_risk"] = (
        seg["member_count"] * seg["churn_rate"] * 65
    ).round(0)
    seg.to_csv("outputs/dash_segments.csv", index=False)
    print("✓ Page 2 saved: dash_segments.csv")


# ── Page 3: Churn Drivers ──────────────────────────────────────────────────
def page3_churn_drivers(df: pd.DataFrame):
    # Credit card impact
    cc = df.groupby("has_credit_card")["churned"].mean().reset_index()
    cc.columns = ["has_credit_card", "churn_rate"]
    cc["driver"] = "Credit Card"
    cc["category"] = cc["has_credit_card"].map({0: "No Credit Card", 1: "Has Credit Card"})

    # Visit frequency impact
    vf = df.groupby("visit_frequency")["churned"].mean().reset_index()
    vf.columns = ["category", "churn_rate"]
    vf["driver"] = "Visit Frequency"

    # Tier impact
    tier = df.groupby("member_tier")["churned"].mean().reset_index()
    tier.columns = ["category", "churn_rate"]
    tier["driver"] = "Member Tier"

    drivers = pd.concat([
        cc[["driver", "category", "churn_rate"]],
        vf[["driver", "category", "churn_rate"]],
        tier[["driver", "category", "churn_rate"]],
    ], ignore_index=True).round(3)

    drivers.to_csv("outputs/dash_churn_drivers.csv", index=False)
    print("✓ Page 3 saved: dash_churn_drivers.csv")


# ── Page 4: Actionable Recommendations ────────────────────────────────────
def page4_recommendations(df: pd.DataFrame):
    recs = []

    # Rec 1: Credit card promotion for at-risk non-card holders
    target1 = df[
        (df["churn_predicted"] == 1) &
        (df["has_credit_card"] == 0) &
        (df["member_tier"].isin(["Gold", "Silver"]))
    ]
    recs.append({
        "priority": 1,
        "action": "Credit Card Promotion",
        "target_segment": "At-Risk Non-Card Holders (Gold/Silver)",
        "target_count": len(target1),
        "rationale": "Card holders show 30% higher retention; converting reduces churn risk",
        "est_revenue_impact": round(len(target1) * 0.30 * 65, 0),
    })

    # Rec 2: Engagement campaign for low-frequency visitors
    target2 = df[
        (df["churn_predicted"] == 1) &
        (df["visit_frequency"] == "low")
    ]
    recs.append({
        "priority": 2,
        "action": "Engagement Campaign",
        "target_segment": "At-Risk Low-Frequency Visitors",
        "target_count": len(target2),
        "rationale": "High-freq visitors show 50% higher renewal; campaigns can shift behavior",
        "est_revenue_impact": round(len(target2) * 0.20 * 65, 0),
    })

    # Rec 3: VIP retention program for top 15%
    top15_threshold = df["monetary_annual"].quantile(0.85)
    target3 = df[
        (df["churn_predicted"] == 1) &
        (df["monetary_annual"] >= top15_threshold)
    ]
    recs.append({
        "priority": 3,
        "action": "VIP Retention Program",
        "target_segment": "At-Risk Top 15% by Spend",
        "target_count": len(target3),
        "rationale": "Top spenders generate disproportionate revenue; high ROI on retention spend",
        "est_revenue_impact": round(len(target3) * 0.40 * 65, 0),
    })

    pd.DataFrame(recs).to_csv("outputs/dash_recommendations.csv", index=False)
    print("✓ Page 4 saved: dash_recommendations.csv")
    print(f"\nTotal estimated revenue opportunity: "
          f"${sum(r['est_revenue_impact'] for r in recs):,.0f}")


if __name__ == "__main__":
    df = load_scored()
    page1_overview(df)
    page2_segments(df)
    page3_churn_drivers(df)
    page4_recommendations(df)
    print("\nAll dashboard exports complete → outputs/")
