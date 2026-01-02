from __future__ import annotations

import os
import numpy as np
import pandas as pd


def main() -> None:
    os.makedirs("data/raw", exist_ok=True)

    rng = np.random.default_rng(42)

    # One year of daily data
    dates = pd.date_range("2025-01-01", periods=365, freq="D")

    df = pd.DataFrame({"date": dates})
    df["dow"] = df["date"].dt.dayofweek  # 0=Mon..6=Sun
    df["month"] = df["date"].dt.month

    # Promo days: random 10% of days
    df["promo"] = (rng.random(len(df)) < 0.10).astype(int)

    # Seasonality + weekly pattern + promo uplift + noise
    base = 120
    weekly = 12 * np.sin(2 * np.pi * (df["dow"] / 7.0))
    yearly = 8 * np.sin(2 * np.pi * (df["date"].dt.dayofyear / 365.0))
    promo_uplift = df["promo"] * 25
    noise = rng.normal(0, 8, size=len(df))

    df["sales"] = (base + weekly + yearly + promo_uplift + noise).round(0).astype(int)

    out_path = "data/raw/coffee_sales_daily.csv"
    df[["date", "sales", "promo"]].to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()