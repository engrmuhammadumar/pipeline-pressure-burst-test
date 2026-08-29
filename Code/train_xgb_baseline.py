import json
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

DATA_DIR = r"D:\Pipeline RUL Data\windows_parquet_20k"
MODEL_OUT = r"D:\Pipeline RUL Data\xgb_baseline_model.joblib"

def load_split(name):
    p = Path(DATA_DIR, f"{name}.parquet")
    df = pd.read_parquet(p)
    y = df["RUL"].astype(np.float32).values
    X = df.drop(columns=["RUL", "t_end"])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    feats = X.columns.tolist()
    return X.values, y, feats

def report(lbl, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    # Compatibility: some old sklearns don't support 'squared=False'
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[{lbl}] MAE={mae:.3f} | RMSE={rmse:.3f}")


def main():
    Xtr, ytr, feats = load_split("train")
    Xva, yva, _    = load_split("val")
    Xte, yte, _    = load_split("test")

    # Optional scaling
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(Xtr); Xva = scaler.transform(Xva); Xte = scaler.transform(Xte)

    # Keep it compatible: no early stopping, no callbacks, no eval_set
    model = XGBRegressor(
        n_estimators=1200,     # modest to avoid overfitting since no early stopping
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
        random_state=42
    )

    model.fit(Xtr, ytr)

    ytr_p = model.predict(Xtr); yva_p = model.predict(Xva); yte_p = model.predict(Xte)
    print("\nResults:")
    report("Train", ytr, ytr_p)
    report("Val  ", yva, yva_p)
    report("Test ", yte, yte_p)

    joblib.dump({"model": model, "scaler": scaler, "features": feats}, MODEL_OUT)
    print(f"\n[OK] Saved model to: {MODEL_OUT}")

    try:
        imps = model.feature_importances_
        top = np.argsort(imps)[::-1][:20]
        print("\nTop-20 features:")
        for i in top:
            print(f"  {feats[i]:35s}  {imps[i]:.4f}")
    except Exception as e:
        print(f"[WARN] Feature importances unavailable: {e}")

if __name__ == "__main__":
    main()
