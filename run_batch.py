#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_batch.py
-------------
Argümansız toplu churn tahmini çalıştırıcısı.

Nasıl çalışır?
- Proje kökünde giriş dosyası olarak şu isimlerden ilk bulduğunu kullanır:
  ["batch_input.xlsx","batch_input.csv","new_customers.xlsx","new_customers.csv"]
- Model dosyasını artifacts_* klasörlerinde veya kök dizinde (*.keras / *.h5 / *.hdf5) arar.
- Ön-işleme artefaktları (le_gender.pkl, ct.pkl, sc.pkl) varsa kullanır, yoksa
  eğitim verisinden (Churn_Modelling.xlsx/csv) yeniden kurar.
- Tahminleri batch_predictions.csv dosyasına yazar.

Gereken Python paketleri:
  pip install tensorflow pandas scikit-learn openpyxl
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# -------------------------
# Kullanıcıya açık sabitler (gerekirse düzenleyin)
# -------------------------
INPUT_CANDIDATES = [
    "batch_input.xlsx", "batch_input.csv",
    "new_customers.xlsx", "new_customers.csv",
]
TRAIN_DATA_CANDIDATES = [
    "Churn_Modelling.xlsx", "Churn_Modelling.csv",
]
ARTIFACTS_DIR_CANDIDATE = None   # örn. "artifacts_2025-10-01"; None => otomatik bul
OUTPUT_PATH = "batch_predictions.csv"
THRESHOLD = 0.65
ID_COLS = ["RowNumber", "CustomerId", "Surname"]

REQUIRED_FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

# -------------------------
# Yardımcılar
# -------------------------
def read_table_auto(path: Path) -> pd.DataFrame:
    """Önce Excel, sonra CSV dene; son çare tekrar Excel dene."""
    if path.suffix.lower() in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            pass
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception as e2:
            raise RuntimeError(
                f"Dosya okunamıyor: {path} (Excel ve CSV denemeleri başarısız). Hata: {e2}"
            ) from e2


def pick_first_existing(cands) -> Path | None:
    for c in cands:
        p = Path(c)
        if p.exists():
            return p
    return None


def find_latest_artifacts_dir(base: Path) -> Path | None:
    cands = sorted(base.glob("artifacts_*"))
    return cands[-1] if cands else None


def find_model_file(search_dir: Path | None) -> Path | None:
    if not search_dir:
        return None
    for pat in ["*.keras", "*.h5", "*.hdf5", "*.savedmodel"]:
        files = sorted(search_dir.glob(pat))
        if files:
            return files[-1]
    return None


def find_model_anywhere() -> Path | None:
    # Önce artifacts_* (en güncel), sonra kök dizin
    latest_art = find_latest_artifacts_dir(Path("."))
    model = find_model_file(latest_art)
    if model:
        return model
    for pat in ["*.keras", "*.h5", "*.hdf5", "*.savedmodel"]:
        files = sorted(Path(".").glob(pat))
        if files:
            return files[-1]
    return None


def try_load_pickle(p: Path | None):
    if p and p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def rebuild_preprocessing_from_training(train_path: Path):
    """
    Eğitim verisinden ön-işleme adımlarını yeniden kur:
      - X = dataset.iloc[:, 3:-1]  (CreditScore ... EstimatedSalary)
      - y = dataset.iloc[:, -1]
      - Gender: LabelEncoder
      - Geography: OneHotEncoder
      - StandardScaler: train split üzerinde fit
    """
    dataset = read_table_auto(train_path)
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values

    le_gender = LabelEncoder()
    X[:, 2] = le_gender.fit_transform(X[:, 2])

    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(), [1])],
        remainder="passthrough",
    )
    X = np.array(ct.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    sc = StandardScaler()
    _ = sc.fit_transform(X_train)
    _ = sc.transform(X_test)

    return le_gender, ct, sc


def load_or_build_preprocessing(art_dir: Path | None, train_path: Path):
    """
    Artefakt varsa yükle; yoksa eğitim verisinden yeniden kur.
    Beklenen artefakt dosya adları:
      - le_gender.pkl / labelencoder_gender.pkl
      - ct.pkl / column_transformer.pkl
      - sc.pkl / scaler.pkl
    """
    le = ct = sc = None
    if art_dir and art_dir.exists():
        for cand in ["le_gender.pkl", "labelencoder_gender.pkl"]:
            le = le or try_load_pickle(art_dir / cand)
        for cand in ["ct.pkl", "column_transformer.pkl"]:
            ct = ct or try_load_pickle(art_dir / cand)
        for cand in ["sc.pkl", "scaler.pkl"]:
            sc = sc or try_load_pickle(art_dir / cand)
    if le is None or ct is None or sc is None:
        if not train_path or not train_path.exists():
            raise FileNotFoundError(
                "Ön-işleme artefaktları bulunamadı ve eğitim verisi yok. "
                "Lütfen eğitim dosyasını (Churn_Modelling.csv/xlsx) proje köküne koyun."
            )
        le, ct, sc = rebuild_preprocessing_from_training(train_path)
    return le, ct, sc


def normalize_gender_series(gender_s: pd.Series, le: LabelEncoder) -> np.ndarray:
    out = []
    lc = [c.lower() for c in le.classes_]
    for val in gender_s.astype(str):
        lv = val.strip().lower()
        if lv in lc:
            v = le.classes_[lc.index(lv)]
        else:
            v = "Male" if lv.startswith("m") else "Female"
        out.append(v)
    return le.transform(np.array(out, dtype=object))


def preprocess_batch(
    df_in: pd.DataFrame, le: LabelEncoder, ct: ColumnTransformer, sc: StandardScaler
) -> np.ndarray:
    missing = [c for c in REQUIRED_FEATURES if c not in df_in.columns]
    if missing:
        raise ValueError(f"Giriş veri setinde zorunlu sütunlar eksik: {missing}")

    X = df_in[REQUIRED_FEATURES].copy()

    for col in ["HasCrCard", "IsActiveMember", "NumOfProducts", "Tenure", "Age", "CreditScore"]:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)

    for col in ["Balance", "EstimatedSalary"]:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)

    X.loc[:, "Gender"] = normalize_gender_series(X["Gender"], le)

    x_arr = X.values.astype(object)
    x_ct = ct.transform(x_arr)
    x_scaled = sc.transform(x_ct)
    return x_scaled


# -------------------------
# Ana akış
# -------------------------
def main():
    # 1) Giriş dosyası
    input_path = pick_first_existing(INPUT_CANDIDATES)
    if not input_path:
        raise FileNotFoundError(
            "Giriş dosyası bulunamadı. Aşağıdakilerden birini proje köküne koyun: "
            + ", ".join(INPUT_CANDIDATES)
        )

    # 2) Model dosyası
    artifacts_dir = Path(ARTIFACTS_DIR_CANDIDATE) if ARTIFACTS_DIR_CANDIDATE else find_latest_artifacts_dir(Path("."))
    model_path = find_model_file(artifacts_dir) or find_model_anywhere()
    if not model_path or not model_path.exists():
        raise FileNotFoundError(
            "Model dosyası bulunamadı. Kök dizinde veya artifacts_* klasöründe .keras/.h5/.hdf5 dosyası arandı."
        )

    # 3) Ön-işleme artefaktları / eğitim verisi
    train_data = pick_first_existing(TRAIN_DATA_CANDIDATES)
    le, ct, sc = load_or_build_preprocessing(artifacts_dir, train_data)

    # 4) İnferens
    df_in = read_table_auto(input_path)
    id_cols = [c for c in ID_COLS if c in df_in.columns]

    X_scaled = preprocess_batch(df_in, le, ct, sc)
    model = tf.keras.models.load_model(model_path)
    proba = model.predict(X_scaled, verbose=0).reshape(-1)
    will = (proba > THRESHOLD).astype(int)

    out = pd.DataFrame({"churn_proba": proba, "will_churn": will, "threshold": THRESHOLD})
    for c in id_cols:
        out[c] = df_in[c].values
    if "Exited" in df_in.columns:
        out["Exited"] = df_in["Exited"].values

    out_cols = id_cols + [c for c in out.columns if c not in id_cols]
    out = out[out_cols]
    out_path = Path(OUTPUT_PATH)
    out.to_csv(out_path, index=False, encoding="utf-8")

    n = len(out)
    n_pos = int(out["will_churn"].sum())
    print(f"input:   {input_path}")
    print(f"Model:   {model_path}")
    print(f"output:   {out_path}")
    print(f"threshold:    {THRESHOLD}")
    print(f"will churn: {n_pos}/{n}  ({n_pos/n*100:.2f}%)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[HATA] {e}", file=sys.stderr)
        sys.exit(1)
