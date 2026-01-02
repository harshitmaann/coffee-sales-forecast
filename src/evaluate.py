from __future__ import annotations

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error


def main() -> None:
    os.makedirs("reports/figures", exist_ok=True)

    pred_path = "reports/predictions.csv"
    model_path = "models/random_forest.pkl"

    if not os.path.exists(pred_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Missing predictions/model. Run:\n"
            "  python -m src.train\n"
            "then:\n"
            "  python -m src.evaluate"
        )

    preds_df = pd.read_csv(pred_path)
    y_true = preds_df["sales"]
    y_pred = preds_df["prediction"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.values, label="Actual")
    plt.plot(y_pred.values, label="Predicted")
    plt.title("Coffee Sales: Predicted vs Actual")
    plt.xlabel("Test-set day index")
    plt.ylabel("Sales")
    plt.legend()

    fig_path = "reports/figures/pred_vs_actual.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()