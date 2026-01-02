from __future__ import annotations

import os
import pandas as pd


def main() -> None:
    os.makedirs("data/processed", exist_ok=True)

    raw_path = "data/raw/coffee_sales_daily.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found. Run: python -m src.make_sample_data"
        )

    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Time features
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Lag features
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)

    # Rolling means (use prior values only)
    df["roll_7"] = df["sales"].shift(1).rolling(window=7).mean()
    df["roll_14"] = df["sales"].shift(1).rolling(window=14).mean()

    # Drop rows where features are NaN
    df_features = df.dropna().reset_index(drop=True)

    out_path = "data/processed/features.csv"
    df_features.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df_features)} rows)")


if __name__ == "__main__":
    main()