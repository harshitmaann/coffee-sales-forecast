from __future__ import annotations

import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def main() -> None:
    # Ensure folders exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Build features if not present
    features_path = "data/processed/features.csv"
    if not os.path.exists(features_path):
        from .preprocess import main as preprocess_main
        preprocess_main()

    df = pd.read_csv(features_path)

    # --- Robustness: if features file exists but columns are missing, rebuild it ---
    required_cols = {
        "sales",
        "promo",
        "dow",
        "month",
        "day",
        "lag_1",
        "lag_7",
        "roll_7",
        "roll_14",
    }
    if not required_cols.issubset(set(df.columns)):
        from .preprocess import main as preprocess_main

        preprocess_main()
        df = pd.read_csv(features_path)

    # Features / target
    target = "sales"
    feature_cols = ["promo", "dow", "month", "day", "lag_1", "lag_7", "roll_7", "roll_14"]

    X = df[feature_cols]
    y = df[target]

    # Time-based split (no shuffling)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # Save predictions
    out_pred = pd.DataFrame(
        {
            "sales": y_test.values,
            "prediction": preds,
        }
    )
    pred_path = "reports/predictions.csv"
    out_pred.to_csv(pred_path, index=False)

    # Save model
    model_path = "models/random_forest.pkl"
    joblib.dump(model, model_path)

    print("Model: RandomForestRegressor")
    print(f"MAE: {mae:.2f}")
    print(f"Saved: {pred_path}")
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()