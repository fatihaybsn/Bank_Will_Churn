# Bank Customer Churn Prediction (Keras + Streamlit)

Predict whether a bank customer will churn in the next 6 months using a Keras ANN and a clean bilingual interface (English/Türkçe). The app supports **single‑customer analysis** and **batch analysis** by uploading a CSV/XLSX; the UI triggers everything—no terminal needed.

---

## 1) Project Summary


🎥 **Demo Video:** [Watch on YouTube](https://youtu.be/Da46G0PJlD0?si=FVk7Q4svmVSqxLn3)  
A short demo showcasing the Streamlit interface and model predictions.

* **Scope:** Binary churn prediction on a retail‑bank dataset (~10k customers). Each record includes customer attributes and whether they churned in the last 6 months.
* **Interface:** Streamlit UI with **English/Turkish** language toggle.
* **Modes:**

  * **Customer analysis:** enter one customer’s features, get probability and decision.
  * **Batch analysis:** upload a CSV/XLSX and download predictions.
* **Decision policy:** If churn probability **> 0.65**, the customer is flagged as likely to churn.
* **Artifacts:** Trained Keras model (`.keras/.h5/.hdf5`). Optional preprocessing pickles can be used if available.

---

## 2) Functionality (Details)

### Customer analysis

* Input the 10 required features: `CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary`.
* The app outputs:

  * **Churn probability** (Exited=1)
  * **Decision** using threshold **0.65** (adjustable in code)

### Batch analysis

* In the **Batch Scoring** section, upload **CSV/XLSX** containing the same required feature columns. Optional IDs (`RowNumber, CustomerId, Surname`) are preserved in the output.
* Click **Run / Çalıştır**. The UI runs the batch pipeline and presents a **Download** button for `batch_predictions.csv`.

---

## 3) Purpose

Help banks proactively identify at‑risk customers and prioritize retention offers. Customers above the **0.65** threshold are segmented for targeted campaigns.

---

## 4) Installation

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

**Model placement:** Put your trained Keras model file in the project root (or an `artifacts_*` folder).
**Training data (fallback for preprocessing):** If you did not export preprocessing artifacts, place `Churn_Modelling.csv` in the project root so the app can rebuild encoders/scaler consistent with training.

---

## 5) Usage

```bash
streamlit run churn_ui_streamlit.py
```

1. Open the app in your browser.
2. Select **English** or **Türkçe** from the sidebar.
3. **Customer analysis:** Fill the form and click **Predict** to see probability and decision.
4. **Batch analysis:** Go to **Batch Scoring**, upload your CSV/XLSX, press **Run / Çalıştır**, then **Download** the results (`batch_predictions.csv`).

> The UI handles batch execution internally; end‑users do **not** need to run any console commands.

---

## 6) Technologies Used

* **TensorFlow / Keras** (classification model)
* **pandas, NumPy** (data handling)
* **scikit‑learn** (LabelEncoder, OneHotEncoder, StandardScaler, train/test split)
* **Streamlit** (bilingual web UI)
* **openpyxl** (Excel I/O)

---

## 7) Keras Model — Training Technique

* **Preprocessing:**

  * `Gender` → `LabelEncoder`
  * `Geography` → `OneHotEncoder`
  * Numerical features → `StandardScaler`
  * `train_test_split(test_size=0.2, random_state=0)`
* **Architecture:** Feed‑forward ANN (MLP) with several `Dense` layers using **ReLU** activations and a **sigmoid** output neuron (`units=1`) for binary classification.
* **Compile:** `optimizer='adam'`, `loss='binary_crossentropy'`, `metrics=['accuracy']`.
* **Output:** Churn probability in `[0, 1]` per customer.

---

## 8) UI at a Glance

* **Language toggle** (EN/TR) in the sidebar.
* **Clean forms** for single‑customer input; clear probability + decision presentation.
* **Batch Scoring**: file upload → run → **download predictions**.
* **Threshold note**: default decision threshold is **0.65** (can be changed in code).

---
