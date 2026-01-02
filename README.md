# Coffee Sales Forecast ☕📈

An end-to-end **time series forecasting** project that predicts daily coffee sales using Python.

This repo focuses on a clean forecasting workflow:
- data generation / loading
- preprocessing + feature engineering (lags, rolling stats)
- baseline model training
- evaluation with clear metrics + a forecast plot

---

## ✅ Project Structure

- `src/` – pipeline scripts (data → features → model → evaluation)
- `data/raw/` – raw CSV input (generated locally)
- `data/processed/` – cleaned + feature dataset
- `models/` – trained model artifact
- `reports/` – predictions + metrics output
- `reports/figures/` – exported chart(s)

---

## ⚙️ Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt