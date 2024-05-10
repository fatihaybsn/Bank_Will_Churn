import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import pickle
import glob
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import subprocess, sys


st.set_page_config(page_title="Churn Prediction (ANN)", page_icon="💳", layout="centered")
# st.set_page_config(page_title="Churn Prediction (ANN)", page_icon="💳", layout="wide")


# i18n sözlüğü + yardımcılar
# ---------------------------------------------------
STRINGS = {
    "en": {
        "app_title": "💳 Bank Customer Churn Prediction",
        "app_caption": "Use your own model and dataset to predict customer churn in your bank.\n\n MLP (Multi-Layer Perceptron) / Feed-Forward ANN (binary classification)",
        "expander_title": "⚙️ Model & Preprocessing Sources",
        "expander_desc": "Select your model file and dataset path:",
        "model_path_label": "Model file path (.keras, .h5, .hdf5):",
        "dataset_path_label": "Dataset used in the notebook (`Churn_Modelling.csv`):",
        "excel_note": "Note: `Churn_Modelling.csv` is actually an Excel file. (read via openpyxl)",
        "rebuild_info": "Preprocessing objects not found. Rebuilding from the dataset...",
        "err_model_not_found": "Model file not found. Please provide a valid path.",
        "bootstrap_error_prefix": "Error while loading model/preprocessing:",
        "form_title": "🧾 Customer Information",
        "geography": "Geography",
        "gender": "Gender",
        "credit_score": "CreditScore",
        "age": "Age",
        "tenure": "Tenure (years)",
        "balance": "Balance",
        "num_products": "NumOfProducts",
        "has_cr_card": "HasCrCard",
        "is_active": "IsActiveMember",
        "estimated_salary": "EstimatedSalary",
        "predict_btn": "Predict",
        "result_title": "🔮 Prediction Result",
        "churn_prob": "Churn probability (Exited=1)",
        "stay_prob": "Stay probability (Exited=0)",
        "result_churn": "**Prediction:** Customer **WILL CHURN** (Take action).",
        "result_stay": "**Prediction:** Customer **WILL STAY** (Looks fine).",
        "threshold_note": "Note: You can change the threshold (0.65) in the code.",
        "footer": "System applies preprocessing (LabelEncoder + OneHotEncoder + StandardScaler) exactly as in the training file.",
        "lang_toggle": "Türkçe arayüz",
        "batch_scoring": "Batch Scoring",
        "batch_context_text": "Load CSV or XLSX, save the file to the project root, and run run_batch.py.",
        "batch_file_load": "Bulk file upload (CSV/XLSX)"
    },
    "tr": {
        "app_title": "💳 Banka Müşteri Churn Tahmini",
        "app_caption": "Kendi modelinizi ve veri setinizi kullanarak bankanızda müşteri ayrılma tahmini yapın.\n\n MLP (Multi-Layer Perceptron) / Feed-Forward ANN (ikili sınıflandırma)",
        "expander_title": "⚙️ Model & Ön-işleme Kaynakları",
        "expander_desc": "Model dosyanızı ve veri seti yolunu seçin:",
        "model_path_label": "Model dosya yolu (.keras, .h5, .hdf5):",
        "dataset_path_label": "Notebook'ta kullandığınız veri seti (`Churn_Modelling.csv`):",
        "excel_note": "Not: `Churn_Modelling.csv` aslında bir Excel dosyasıdır. (openpyxl ile okunur)",
        "rebuild_info": "Ön-işleme nesneleri bulunamadı. Veri setinden yeniden oluşturuluyor...",
        "err_model_not_found": "Model dosyası bulunamadı. Lütfen geçerli bir yol girin.",
        "bootstrap_error_prefix": "Model/ön-işleme yüklenirken hata:",
        "form_title": "🧾 Müşteri Bilgileri",
        "geography": "Ülke",
        "gender": "Cinsiyet",
        "credit_score": "Kredi Notu",
        "age": "Yaş",
        "tenure": "Kıdem (yıl)",
        "balance": "Bakiye",
        "num_products": "Ürün Sayısı",
        "has_cr_card": "Kredi Kartı Var",
        "is_active": "Aktif Üye",
        "estimated_salary": "Tahmini Maaş",
        "predict_btn": "Hesapla",
        "result_title": "🔮 Tahmin Sonucu",
        "churn_prob": "Ayrılma Olasılığı (Exited=1)",
        "stay_prob": "Kalma Olasılığı (Exited=0)",
        "result_churn": "**Tahmin:** Müşteri **AYRILACAK** (Kampanya yapılmalı).",
        "result_stay": "**Tahmin:** Müşteri **KALACAK** (Uygun görünüyor).",
        "threshold_note": "Not: Eşik değerini (0.65) kod içinde değiştirebilirsiniz.",
        "footer": "Sistem, eğitim dosyasındaki ön-işlemeyi (LabelEncoder + OneHotEncoder + StandardScaler) bire bir uygular.",
        "lang_toggle": "Türkçe arayüz",
        "batch_scoring": "Toplu Analiz",
        "batch_context_text": "CSV veya XLSX yükleyin; dosya proje köküne kaydedilir ve run_batch.py çalıştırılır.",
        "batch_file_load": "Toplu dosya yükle (CSV/XLSX)"
    }
}

def t(lang, key):
    return STRINGS.get(lang, STRINGS["en"]).get(key, key)

def fmt_pct(x, lang):
    s = f"{x*100:.2f}%"
    return s if lang == "en" else s.replace(".", ",")

# Seçenek etiketleri (iç değerler sabit kalır; ekranda çeviri gösterilir)
GEO_LABELS = {
    "en": {"France": "France", "Germany": "Germany", "Spain": "Spain"},
    "tr": {"France": "Fransa", "Germany": "Almanya", "Spain": "İspanya"},
}
GENDER_LABELS = {
    "en": {"Female": "Female", "Male": "Male"},
    "tr": {"Female": "Kadın", "Male": "Erkek"},
}

# ---------------------------------------------------
# Dil seçimi (varsayılan: İngilizce)
# ---------------------------------------------------
ui_tr = st.sidebar.toggle(t("en", "lang_toggle"), value=False)  # label sabit TR: anlaşılır
lang = "tr" if ui_tr else "en"

# Başlık ve giriş metni
st.title(t(lang, "app_title"))
st.caption(t(lang, "app_caption"))

# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
def find_latest_artifacts_dir(base: Path) -> Path | None:
    candidates = sorted(base.glob("artifacts_*"))
    return candidates[-1] if candidates else None

def find_model_file(art_dir: Path | None) -> Path | None:
    exts = ["*.keras", "*.h5", "*.hdf5", "*.savedmodel"]
    search_dirs = [art_dir] if art_dir else []
    search_dirs.append(Path("."))
    for d in search_dirs:
        if not d:
            continue
        for pat in exts:
            files = sorted(d.glob(pat))
            if files:
                return files[-1]
    return None

def try_load_pickle(p: Path | None):
    if p and p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def rebuild_preprocessing_from_dataset(csv_like_path: Path):
    """
    Rebuild encoders + scaler to match the notebook:
      - Read 'Churn_Modelling.csv' as Excel
      - X = dataset.iloc[:, 3:-1].values
      - y = dataset.iloc[:, -1].values
      - LabelEncode Gender (col index 2 in X)
      - OneHotEncode Geography (col index 1 in X)
      - Train/test split (test_size=0.2, random_state=0)
      - StandardScaler fit on X_train
    """
    dataset = pd.read_excel(csv_like_path, engine="openpyxl")
    X = dataset.iloc[:, 3:-1].values  # CreditScore ... EstimatedSalary
    y = dataset.iloc[:, -1].values    # Exited

    le_gender = LabelEncoder()
    X[:, 2] = le_gender.fit_transform(X[:, 2])

    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(), [1])],
        remainder="passthrough"
    )
    X = np.array(ct.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    _ = sc.transform(X_test)

    return le_gender, ct, sc

def load_or_build_preprocessing(art_dir: Path | None, dataset_path: Path, lang: str = "en"):
    le = None
    ct = None
    sc = None

    if art_dir and art_dir.exists():
        le = try_load_pickle(art_dir / "le_gender.pkl")
        ct = try_load_pickle(art_dir / "ct.pkl")
        sc = try_load_pickle(art_dir / "sc.pkl")
        if not le:
            le = try_load_pickle(art_dir / "labelencoder_gender.pkl")
        if not ct:
            ct = try_load_pickle(art_dir / "column_transformer.pkl")
        if not sc:
            sc = try_load_pickle(art_dir / "scaler.pkl")

    if le is None or ct is None or sc is None:
        st.info(t(lang, "rebuild_info"))
        le, ct, sc = rebuild_preprocessing_from_dataset(dataset_path)

    return le, ct, sc

@st.cache_resource(show_spinner=False)
def bootstrap(model_path: Path | None, dataset_path: Path, lang: str) -> tuple[tf.keras.Model, LabelEncoder, ColumnTransformer, StandardScaler]:
    art_dir = model_path.parent if model_path else find_latest_artifacts_dir(Path("."))
    le, ct, sc = load_or_build_preprocessing(art_dir, dataset_path)

    if model_path is None or not model_path.exists():
        raise FileNotFoundError(t(lang, "err_model_not_found"))
    model = tf.keras.models.load_model(model_path)
    return model, le, ct, sc

def make_feature_row(
    geography: str, gender: str, credit_score: int, age: int, tenure: int,
    balance: float, num_products: int, has_cr_card: bool, is_active: bool, estimated_salary: float
):
    return np.array([[
        credit_score,
        geography,
        gender,
        age,
        tenure,
        balance,
        num_products,
        int(has_cr_card),
        int(is_active),
        estimated_salary
    ]], dtype=object)

def preprocess_single(x_raw, le_gender: LabelEncoder, ct: ColumnTransformer, sc: StandardScaler):
    x_enc = x_raw.copy()
    g = np.array([x_enc[0, 2]], dtype=object)
    if g[0] not in le_gender.classes_:
        lc = [c.lower() for c in le_gender.classes_]
        try:
            idx = lc.index(str(g[0]).lower())
            g[0] = le_gender.classes_[idx]
        except Exception:
            g[0] = "Male" if str(g[0]).strip().lower().startswith("m") else "Female"
    x_enc[0, 2] = le_gender.transform(g)[0]

    x_ct = ct.transform(x_enc)
    x_scaled = sc.transform(x_ct)
    return x_scaled

# ---------------------------------------------------
# Model + preprocessing girişleri (expander)
# ---------------------------------------------------
default_art_dir = find_latest_artifacts_dir(Path("."))
default_model_file = find_model_file(default_art_dir)

with st.expander(t(lang, "expander_title"), expanded=False):
    st.write(t(lang, "expander_desc"))
    model_file_input = st.text_input(
        t(lang, "model_path_label"),
        value=str(default_model_file) if default_model_file else ""
    )
    dataset_file_input = st.text_input(
        t(lang, "dataset_path_label"),
        value="Churn_Modelling.csv"
    )
    st.caption(t(lang, "excel_note"))

model_path = Path(model_file_input) if model_file_input else None
dataset_path = Path(dataset_file_input) if dataset_file_input else Path("Churn_Modelling.csv")

bootstrap_ok = True
try:
    model, le_gender, ct, sc = bootstrap(model_path, dataset_path, lang)
except Exception as e:
    bootstrap_ok = False
    st.error(f"{t(lang, 'bootstrap_error_prefix')} {e}")

# ---------------------------------------------------
# Input Form
# ---------------------------------------------------
with st.form("churn_form"):
    st.subheader(t(lang, "form_title"))

    col1, col2 = st.columns(2)
    with col1:
        # İç değerler sabit; ekranda dil etiketleri
        geo_options = ["France", "Germany", "Spain"]
        geography = st.selectbox(
            t(lang, "geography"),
            geo_options,
            format_func=lambda v: GEO_LABELS[lang][v]
        )

        gender_options = ["Female", "Male"]
        gender = st.selectbox(
            t(lang, "gender"),
            gender_options,
            format_func=lambda v: GENDER_LABELS[lang][v]
        )

        credit_score = st.number_input(t(lang, "credit_score"), min_value=300, max_value=900, value=600, step=1)
        age = st.number_input(t(lang, "age"), min_value=18, max_value=100, value=40, step=1)
        tenure = st.number_input(t(lang, "tenure"), min_value=0, max_value=10, value=3, step=1)
    with col2:
        balance = st.number_input(t(lang, "balance"), min_value=0.0, value=60000.0, step=100.0, format="%.2f")
        num_products = st.number_input(t(lang, "num_products"), min_value=1, max_value=4, value=2, step=1)
        has_cr_card = st.checkbox(t(lang, "has_cr_card"), value=True)
        is_active = st.checkbox(t(lang, "is_active"), value=True)
        estimated_salary = st.number_input(t(lang, "estimated_salary"), min_value=0.0, value=50000.0, step=100.0, format="%.2f")

    hesapla = st.form_submit_button(t(lang, "predict_btn"))

if hesapla:
    if not bootstrap_ok:
        st.stop()
    x_raw = make_feature_row(
        geography=geography, gender=gender, credit_score=int(credit_score),
        age=int(age), tenure=int(tenure), balance=float(balance),
        num_products=int(num_products), has_cr_card=bool(has_cr_card),
        is_active=bool(is_active), estimated_salary=float(estimated_salary)
    )

    try:
        x_scaled = preprocess_single(x_raw, le_gender, ct, sc)
        prob = float(model.predict(x_scaled, verbose=0)[0][0])
        will_churn = prob > 0.65
        st.markdown("---")
        st.subheader(t(lang, "result_title"))

        colA, colB = st.columns(2)
        with colA:
            st.metric(t(lang, "churn_prob"), fmt_pct(prob, lang))
        with colB:
            st.metric(t(lang, "stay_prob"), fmt_pct(1 - prob, lang))

        if will_churn:
            st.error(t(lang, "result_churn"))
        else:
            st.success(t(lang, "result_stay"))

        st.caption(t(lang, "threshold_note"))
    except Exception as e:
        st.exception(e)
        st.stop()



# ===========================
# TOPLU ANALİZ (Batch Scoring)
# ===========================
st.header(t(lang, "batch_scoring"))
st.write(t(lang, "batch_context_text"))

uploaded = st.file_uploader(t(lang, "batch_file_load"), type=["csv", "xlsx"])

col_run, col_clear = st.columns([1,1])
run_clicked   = col_run.button("Çalıştır", type="primary", use_container_width=True)
clear_clicked = col_clear.button("Temizle", use_container_width=True)

log_box = st.empty()

if clear_clicked:
    for p in ["batch_input.csv", "batch_input.xlsx", "batch_predictions.csv"]:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
    st.success("Geçici dosyalar temizlendi.")

if run_clicked:
    if not uploaded:
        st.error("Lütfen bir dosya yükleyin.")
    else:
        # Yüklenen dosyayı proje köküne yaz (run_batch.py'nin beklediği isimle)
        suffix = Path(uploaded.name).suffix.lower()
        in_name = "batch_input.xlsx" if suffix in [".xlsx", ".xls"] else "batch_input.csv"
        input_path = Path(in_name)
        with open(input_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.info(f"Girdi kaydedildi: {input_path}. run_batch.py çalıştırılıyor...")

        # run_batch.py'yi aynı klasörden çağır
        script_dir = Path(__file__).parent
        run_batch_path = script_dir / "run_batch.py"

        try:
            result = subprocess.run(
                [sys.executable, str(run_batch_path)],
                capture_output=True,
                text=True,
                timeout=1800  # 30 dk
            )
        except Exception as e:
            st.error(f"Komut çalıştırılamadı: {e}")
        else:
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            log_box.code((stdout + "\n" + stderr).strip(), language="bash")

            out_path = Path("batch_predictions.csv")
            if result.returncode == 0 and out_path.exists():
                st.success(f"Tamamlandı. Çıktı: {out_path}")
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="Sonuçları indir (CSV)",
                        data=f.read(),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                # İlk satırlar önizleme (opsiyonel)
                try:
                    df_prev = pd.read_csv(out_path)
                    st.dataframe(df_prev.head(20))
                except Exception:
                    pass
            else:
                st.error("İşlem başarısız oldu. Yukarıdaki günlük çıktısını kontrol edin.")

st.write("")
st.write("—")
st.caption(t(lang, "footer"))