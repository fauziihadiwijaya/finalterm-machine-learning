# 🧠 Final Term — Deep Learning End-to-End Pipeline

> **Hands-on End-to-End Deep Learning Models** untuk Fraud Detection & Song Year Prediction, dilengkapi Optuna Hyperparameter Tuning, MLflow Experiment Tracking, dan LIME Model Interpretation.

---

## 👤 Identitas

| | |
|---|---|
| **Nama** | Mohammad Fauzi Hadiwijaya |
| **NIM** | 101032300044 |
| **Kelas** | TK-47-04 |

---

## 📌 Tujuan Repository

Repository ini dibuat sebagai pemenuhan tugas **UAS (Final Exam) — Machine Learning Class**, dengan tujuan menerapkan pipeline machine learning end-to-end menggunakan **Neural Network (TensorFlow/Keras)** pada dua jenis permasalahan yang berbeda: **klasifikasi** (fraud detection) dan **regresi** (prediksi tahun rilis lagu).

Setiap notebook mendemonstrasikan siklus penuh data science: mulai dari eksplorasi data, preprocessing, desain arsitektur neural network, hyperparameter tuning otomatis, hingga evaluasi dan interpretasi model.

---

## 🗂️ Struktur Repository

```
finalterm-machine-learning/
│
├── fraud_detection_dl.ipynb           # Task 1: Fraud Detection (Klasifikasi)
├── regression_song_year_dl.ipynb      # Task 2: Song Year Prediction (Regresi)
├── submission_dl.csv                  # Output prediksi fraud detection
└── README.md                          # Dokumentasi ini
```

---

## 📖 Overview Project

### 🔍 Task 1 — Fraud Detection (Klasifikasi)

Membangun model **Neural Network** untuk memprediksi probabilitas sebuah transaksi online bersifat **fraudulent**, menggunakan dataset **IEEE-CIS Fraud Detection**.

**Dataset**: `train_transaction.csv`, `test_transaction.csv`
**Target**: `isFraud` (0 = normal, 1 = fraud)

**Pipeline:**
1. EDA — analisis distribusi kelas (class imbalance ~3.5% fraud) dan missing values
2. Preprocessing — drop kolom missing >80%, label encoding, imputasi median, standard scaling
3. Feature engineering — log transform amount, ekstraksi jam & hari transaksi
4. Handle imbalance — **SMOTE** (oversampling kelas minoritas)
5. Modeling — Neural Network dengan Dense layers + BatchNormalization + Dropout
6. Hyperparameter tuning — **Optuna** (15 trials): jumlah layer, neuron, dropout, learning rate, batch size
7. Evaluasi — AUC-ROC, F1-Score, Accuracy, Precision, Recall, dengan **threshold optimal** (bukan default 0.5)
8. Tracking — **MLflow** mencatat seluruh eksperimen otomatis

---

### 🎵 Task 2 — Song Year Prediction (Regresi)

Membangun model **Neural Network** untuk memprediksi **tahun rilis lagu** berdasarkan fitur audio (timbre dan karakteristik sinyal musik).

**Dataset**: `midterm-regresi-dataset.csv`
**Target**: `year` (tahun rilis, kontinu)

**Pipeline:**
1. EDA — distribusi tahun rilis, analisis korelasi fitur terhadap target
2. Preprocessing — outlier clipping (IQR x3), imputasi median, standard scaling (fitur **dan** target)
3. Modeling — Neural Network dengan Dense layers + BatchNormalization + Dropout
4. Hyperparameter tuning — **Optuna** (15 trials)
5. Evaluasi — MSE, RMSE, MAE, R²
6. Interpretasi — **LIME** untuk menjelaskan prediksi model secara lokal (sampel rendah, tengah, tinggi)
7. Tracking — **MLflow** mencatat seluruh eksperimen otomatis

---

## 🤖 Model & Hasil Evaluasi

### Task 1 — Fraud Detection

| Model | Arsitektur | AUC-ROC | F1-Score* | Accuracy* |
|---|---|---|---|---|
| Simple MLP | 2 hidden layers | 0.8739 | 0.104548	 | 0.61115 |
| Deep MLP | 3+ hidden layers | 0.8754 | 0.059179 | 0.19875 |
| **Optuna MLP** | Tuned (15 trials) | 0.8547 | 0.071960 | 0.37580 |

> *F1-Score dan Accuracy dievaluasi ulang menggunakan **threshold optimal** (hasil `precision_recall_curve`), bukan threshold default 0.5, karena dataset sangat imbalanced.

**Insight:**
- AUC-ROC tinggi pada ketiga model menunjukkan kemampuan diskriminasi yang baik antara fraud vs non-fraud
- Pada threshold default, Precision rendah karena imbalance ekstrem — threshold optimal membantu menyeimbangkan Precision-Recall
- SMOTE efektif membantu model mengenali pola minoritas (fraud) yang sangat jarang muncul

---

### Task 2 — Song Year Prediction

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| NN Baseline | 73.1767 | 8.5543 | 5.9228 | 0.3357 |
| **NN Optuna** | **72.6895** | **8.5258** | **5.8677** | **0.3401** |

**Insight:**
- NN_Optuna unggul tipis di semua metrik dibanding baseline
- RMSE ±8.5 tahun — wajar untuk dataset audio-to-year yang secara inheren memiliki batas prediktabilitas rendah (fitur timbre tidak sepenuhnya menentukan tahun rilis)
- R² = 0.34 sejalan dengan benchmark umum dataset sejenis (Million Song Dataset Year Prediction)

**Interpretasi LIME (contoh temuan):**
- Prediksi tahun **rendah** (tua) didorong kuat oleh `feature_1`
- Prediksi tahun **tinggi** (baru) didorong kuat oleh `feature_23`
- Prediksi di rentang **tengah** dipengaruhi banyak fitur kecil secara terdistribusi

---

## ⚙️ Tools & Framework

| Tool | Kegunaan |
|---|---|
| `TensorFlow / Keras` | Membangun & melatih Neural Network |
| `scikit-learn` | Preprocessing, evaluasi, train-test split |
| `imbalanced-learn` | SMOTE untuk class imbalance (Task 1) |
| `optuna` | Hyperparameter tuning otomatis (TPE algorithm) |
| `mlflow` | Experiment tracking — log parameter, metrik, model |
| `lime` | Interpretasi prediksi model secara lokal (Task 2) |

---

## 🧭 Cara Navigasi Repository

1. **`fraud_detection_dl.ipynb`** — buka notebook ini untuk melihat pipeline klasifikasi fraud detection lengkap dari EDA hingga submission file
2. **`regression_song_year_dl.ipynb`** — buka notebook ini untuk melihat pipeline regresi prediksi tahun rilis lagu, termasuk interpretasi LIME
3. **`submission_dl.csv`** — hasil prediksi probabilitas fraud untuk test set (format: `TransactionID`, `isFraud`)

Setiap notebook disusun dengan struktur konsisten:
```
Setup → Load Data → EDA → Preprocessing → Modeling →
Hyperparameter Tuning (Optuna) → Evaluasi → MLflow Summary
```
Setiap section dilengkapi sel markdown berisi insight dan interpretasi hasil.

---

## 🚀 Cara Menjalankan

1. **Clone repository**
   ```bash
   git clone https://github.com/fauziihadiwijaya/finalterm-machine-learning.git
   ```

2. **Siapkan dataset** di Google Drive (dataset sama dengan UTS):
   - `train_transaction.csv`, `test_transaction.csv` (Task 1)
   - `midterm-regresi-dataset.csv` (Task 2)

3. **Buka notebook di Google Colab**
   - Upload notebook yang ingin dijalankan
   - Aktifkan GPU: `Runtime → Change runtime type → T4 GPU`
   - Sesuaikan path dataset pada cell **Load Dataset**
   - `Runtime → Run all`

---

## 📦 Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn imbalanced-learn optuna mlflow lime
```

---

## 📅 Informasi Tugas

| | |
|---|---|
| **Mata Kuliah** | Machine Learning |
| **Jenis Tugas** | UAS (Final Exam) — Individual Task |
