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

    # Always train from processed features (not raw)
    features_path = "data/processed/features.csv"
    if not os.path.exists(features_path):
        from .preprocess import main as preprocess_main
        preprocess_main()

    df = pd.read_csv(features_path)

    target = "sales"
    feature_cols = ["promo", "dow", "month", "day", "lag_1", "lag_7", "roll_7", "roll_14"]

    # Safety check (prevents the KeyError you hit earlier)
    missing = [c for c in feature_cols + [target] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {features_path}: {missing}. "
            "Run: python -m src.preprocess"
        )

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
    out_pred = pd.DataFrame({"sales": y_test.values, "prediction": preds})
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