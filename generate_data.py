"""
generate_data.py
Generates simulated retail membership data for churn prediction analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 10000

def generate_members():
    member_ids = [f"M{str(i).zfill(6)}" for i in range(1, N + 1)]
    
    join_dates = [
        datetime(2020, 1, 1) + timedelta(days=random.randint(0, 365 * 4))
        for _ in range(N)
    ]
    
    has_credit_card = np.random.choice([0, 1], size=N, p=[0.45, 0.55])
    visit_frequency = np.random.choice(
        ["low", "medium", "high"], size=N, p=[0.35, 0.40, 0.25]
    )
    
    # Spending patterns
    base_spend = np.random.lognormal(mean=7.0, sigma=0.8, size=N)
    avg_transaction_value = base_spend / np.random.randint(1, 20, size=N)
    
    # RFM components
    recency_days = np.where(
        visit_frequency == "high",
        np.random.randint(1, 30, size=N),
        np.where(
            visit_frequency == "medium",
            np.random.randint(15, 90, size=N),
            np.random.randint(60, 365, size=N),
        ),
    )
    
    frequency_score = np.where(
        visit_frequency == "high",
        np.random.randint(20, 52, size=N),
        np.where(
            visit_frequency == "medium",
            np.random.randint(6, 20, size=N),
            np.random.randint(1, 6, size=N),
        ),
    )
    
    monetary_score = base_spend
    
    # Churn probability - strong deterministic signal for realistic ML demo
    churn_prob = (
        0.50
        - has_credit_card * 0.30
        + (visit_frequency == "low").astype(float) * 0.35
        - (visit_frequency == "high").astype(float) * 0.32
        - np.clip((monetary_score - monetary_score.mean()) / monetary_score.std(), -2, 2) * 0.10
        + np.random.normal(0, 0.03, size=N)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.98)
    churned = (np.random.uniform(size=N) < churn_prob).astype(int)
    
    member_tier = np.where(
        monetary_score > np.percentile(monetary_score, 85), "Gold",
        np.where(monetary_score > np.percentile(monetary_score, 50), "Silver", "Bronze")
    )
    
    df = pd.DataFrame({
        "member_id": member_ids,
        "join_date": join_dates,
        "has_credit_card": has_credit_card,
        "visit_frequency": visit_frequency,
        "recency_days": recency_days,
        "frequency_annual": frequency_score,
        "monetary_annual": monetary_score.round(2),
        "avg_transaction_value": avg_transaction_value.round(2),
        "member_tier": member_tier,
        "churned": churned,
    })
    
    return df


if __name__ == "__main__":
    df = generate_members()
    df.to_csv("data/members_raw.csv", index=False)
    print(f"Generated {len(df)} member records.")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(df.head())
