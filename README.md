# StrokeRisk-Catcher

StrokeRisk-Catcher adalah aplikasi web berbasis Deep Learning untuk melakukan prediksi awal risiko stroke berdasarkan faktor gaya hidup dan kondisi klinis pengguna.

Project ini dikembangkan sebagai bagian dari Ujian Akhir Semester (UAS) Praktikum Machine Learning dengan studi kasus klasifikasi pada dataset Cerebral Stroke Prediction.

Dalam project ini dilakukan implementasi, evaluasi, dan perbandingan dua metode Deep Learning, yaitu Multi-Layer Perceptron (MLP) dan Convolutional Neural Network 1-Dimensi (CNN 1D).

Model terbaik berdasarkan hasil evaluasi kemudian diimplementasikan ke dalam aplikasi web menggunakan Flask.

> **Disclaimer:** StrokeRisk-Catcher merupakan aplikasi prediksi awal atau skrining berbasis model Deep Learning dan bukan alat diagnosis medis.

---

## Dataset

Dataset yang digunakan adalah **Cerebral Stroke Prediction - Imbalanced Dataset**.

**Source:** Kaggle — Cerebral Stroke Prediction Imbalanced Dataset

Dataset disimpan pada:

```text
dataset/stroke.csv
```

Dataset memiliki target klasifikasi:

```text
stroke
```

dengan keterangan:

```text
0 = Tidak Stroke
1 = Stroke
```

---

## Metadata

| Field | Description |
|---|---|
| `id` | ID unik setiap data |
| `gender` | Jenis kelamin |
| `age` | Usia |
| `hypertension` | Riwayat hipertensi |
| `heart_disease` | Riwayat penyakit jantung |
| `ever_married` | Status pernikahan |
| `work_type` | Jenis pekerjaan |
| `Residence_type` | Tipe tempat tinggal |
| `avg_glucose_level` | Rata-rata kadar glukosa |
| `bmi` | Body Mass Index |
| `smoking_status` | Status merokok |
| `stroke` | Target klasifikasi: 0 = Tidak Stroke, 1 = Stroke |

---

## Aim

Project ini bertujuan untuk:

1. Melakukan preprocessing dan transformasi pada dataset stroke.
2. Membagi dataset menjadi data training, validation, dan testing.
3. Mengatasi ketidakseimbangan kelas pada data training menggunakan SMOTE.
4. Membangun model Deep Learning menggunakan MLP.
5. Membangun model Deep Learning menggunakan CNN 1D.
6. Mengevaluasi model menggunakan Accuracy, Precision, Recall, dan F1-Score.
7. Membandingkan performa MLP dan CNN 1D.
8. Menentukan model terbaik berdasarkan hasil evaluasi.
9. Mengimplementasikan model terbaik ke dalam aplikasi web.
10. Menampilkan hasil prediksi awal risiko stroke beserta probabilitas prediksi.

---

## Deep Learning Methods

### 1. Multi-Layer Perceptron (MLP)

Model pertama menggunakan arsitektur Multi-Layer Perceptron.

Arsitektur utama model:

```text
Input (10 Features)
        ↓
Dense (128, ReLU)
        ↓
Batch Normalization
        ↓
Dropout (0.30)
        ↓
Dense (64, ReLU)
        ↓
Batch Normalization
        ↓
Dropout (0.20)
        ↓
Dense (32, ReLU)
        ↓
Dense (1, Sigmoid)
```

Konfigurasi training:

```text
Optimizer  : Adam
Learning Rate : 0.001
Loss       : Binary Crossentropy
Epochs     : 50
Batch Size : 32
Callback   : EarlyStopping
```

---

### 2. Convolutional Neural Network 1D (CNN 1D)

Model kedua menggunakan Convolutional Neural Network 1-Dimensi.

Data input diubah menjadi format:

```text
(samples, features, channels)
```

Arsitektur utama model:

```text
Input (10 Features, 1 Channel)
        ↓
Conv1D (64 Filters)
        ↓
Batch Normalization
        ↓
MaxPooling1D
        ↓
Dropout (0.30)
        ↓
Conv1D (32 Filters)
        ↓
Batch Normalization
        ↓
Flatten
        ↓
Dense (64, ReLU)
        ↓
Dropout (0.20)
        ↓
Dense (32, ReLU)
        ↓
Dense (1, Sigmoid)
```

Konfigurasi training:

```text
Optimizer  : Adam
Learning Rate : 0.001
Loss       : Binary Crossentropy
Epochs     : 50
Batch Size : 32
Callback   : EarlyStopping
```

---

## Preprocessing

Tahapan preprocessing yang dilakukan adalah:

1. Membaca dataset.
2. Menghapus kolom `id`.
3. Menghapus data dengan nilai `gender = Other`.
4. Menangani missing value pada `bmi` menggunakan median.
5. Menghapus data duplikat.
6. Melakukan Label Encoding pada fitur kategorikal.
7. Membagi dataset menjadi training, validation, dan testing.
8. Melakukan Standard Scaling.
9. Melakukan SMOTE hanya pada data training.
10. Menyimpan hasil preprocessing dan objek preprocessing untuk digunakan kembali.

Objek hasil preprocessing:

```text
models/scaler.pkl
models/label_encoders.pkl
models/preprocessed_data.npz
```

---

## Evaluation Metrics

Karena studi kasus merupakan klasifikasi, model dievaluasi menggunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Accuracy tidak digunakan sebagai satu-satunya dasar pemilihan model karena dataset stroke memiliki distribusi kelas yang tidak seimbang.

Recall dan F1-Score menjadi pertimbangan penting karena aplikasi digunakan sebagai sistem prediksi awal atau skrining risiko stroke.

---

## Model Evaluation Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| MLP | 0.8396 | 0.0582 | 0.5169 | 0.1046 |
| CNN 1D | 0.8014 | 0.0606 | 0.6864 | 0.1113 |

### Best Model

Berdasarkan hasil evaluasi dan komparasi, model terbaik yang dipilih adalah:

```text
CNN 1D
```

CNN 1D dipilih karena memperoleh nilai Recall dan F1-Score yang lebih tinggi dibandingkan MLP.

Dalam konteks prediksi awal risiko stroke, kemampuan model dalam mendeteksi kelas positif menjadi pertimbangan penting.

Meskipun MLP memperoleh Accuracy yang lebih tinggi, CNN 1D memiliki kemampuan yang lebih baik dalam mendeteksi data pasien yang termasuk kelas stroke.

---

## Web Application

Model CNN 1D sebagai model terbaik diimplementasikan pada aplikasi web menggunakan Flask.

Pengguna memasukkan data:

- Jenis kelamin
- Usia
- Riwayat hipertensi
- Riwayat penyakit jantung
- Status pernikahan
- Jenis pekerjaan
- Tipe tempat tinggal
- Rata-rata kadar glukosa
- BMI
- Status merokok

Alur prediksi aplikasi:

```text
User Input
    ↓
Label Encoding
    ↓
Standard Scaling
    ↓
Reshape Input for CNN 1D
    ↓
CNN 1D Model
    ↓
Prediction Probability
    ↓
Classification Result
```

Threshold klasifikasi yang digunakan adalah:

```text
Probability >= 0.5 → Terindikasi Risiko Stroke

Probability < 0.5 → Tidak Terindikasi Risiko Stroke
```

Output aplikasi terdiri dari:

- Hasil klasifikasi risiko stroke.
- Probabilitas prediksi kelas stroke.
- Disclaimer bahwa hasil prediksi bukan diagnosis medis.

---

## Tech Stack

### Machine Learning & Deep Learning

- Python
- TensorFlow
- Keras
- Scikit-learn
- Imbalanced-learn
- NumPy
- Pandas

### Data Visualization

- Matplotlib

### Model Persistence

- Joblib
- Keras Model Format

### Web Application

- Flask
- HTML
- CSS

### Development Tools

- Visual Studio Code
- Git
- GitHub

---

## Folder Structure

```text
StrokeRisk-Catcher/
│
├── dataset/
│   └── stroke.csv
│
├── models/
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   ├── preprocessed_data.npz
│   ├── mlp_model.keras
│   └── cnn_model.keras
│
├── results/
│   ├── mlp_accuracy.png
│   ├── mlp_loss.png
│   ├── mlp_confusion_matrix.png
│   ├── mlp_classification_report.txt
│   │
│   ├── cnn_accuracy.png
│   ├── cnn_loss.png
│   ├── cnn_confusion_matrix.png
│   ├── cnn_classification_report.txt
│   ├── cnn_metrics.npz
│   │
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── model_comparison_report.txt
│   ├── evaluation_mlp_confusion_matrix.png
│   └── evaluation_cnn_confusion_matrix.png
│
├── training/
│   ├── preprocessing.py
│   ├── train_mlp.py
│   ├── train_cnn.py
│   └── evaluate.py
│
├── website/
│   ├── app.py
│   │
│   ├── static/
│   │
│   └── templates/
│       └── index.html
│
├── .gitignore
├── requirements.txt
└── README.md
```

> File model dan hasil preprocessing pada folder `models/` dapat dikecualikan dari repository melalui `.gitignore` karena file tersebut merupakan hasil generate dari proses preprocessing dan training.

---

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
```

Masuk ke folder project:

```bash
cd StrokeRisk-Catcher
```

### 2. Create Virtual Environment

Disarankan menggunakan Python 3.11 karena kompatibilitas TensorFlow.

```bash
py -3.11 -m venv venv
```

### 3. Activate Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Project dijalankan secara berurutan.

### 1. Preprocessing Dataset

```bash
python training/preprocessing.py
```

File yang dihasilkan:

```text
models/scaler.pkl
models/label_encoders.pkl
models/preprocessed_data.npz
```

### 2. Train MLP Model

```bash
python training/train_mlp.py
```

File utama yang dihasilkan:

```text
models/mlp_model.keras
```

Hasil evaluasi MLP disimpan pada folder:

```text
results/
```

### 3. Train CNN 1D Model

```bash
python training/train_cnn.py
```

File utama yang dihasilkan:

```text
models/cnn_model.keras
```

Hasil evaluasi CNN 1D disimpan pada folder:

```text
results/
```

### 4. Evaluate and Compare Models

```bash
python training/evaluate.py
```

File utama yang dihasilkan:

```text
results/model_comparison.csv
results/model_comparison.png
results/model_comparison_report.txt
```

### 5. Run Web Application

```bash
python website/app.py
```

Secara default aplikasi dapat diakses melalui:

```text
http://127.0.0.1:5000
```

---

## Example Prediction Result

Contoh hasil prediksi aplikasi:

```text
Hasil Prediksi

Terindikasi Risiko Stroke

Probabilitas prediksi kelas stroke: 83.27%
```

Hasil tersebut merupakan probabilitas prediksi model terhadap kelas stroke dan tidak dapat digunakan sebagai diagnosis medis.

---

## Current Development Status

```text
[✓] Dataset Preparation
[✓] Data Preprocessing
[✓] Train / Validation / Test Split
[✓] SMOTE
[✓] MLP Model Training
[✓] CNN 1D Model Training
[✓] Model Evaluation
[✓] Model Comparison
[✓] Best Model Selection
[✓] Flask Backend Integration
[✓] Basic Web Prediction Interface
[ ] Web Interface Styling
[ ] Web Application Testing
[ ] Final Documentation
```

---

## Disclaimer

StrokeRisk-Catcher dibuat untuk keperluan akademik dan pembelajaran Machine Learning serta Deep Learning.

Prediksi yang dihasilkan aplikasi bukan diagnosis medis dan tidak boleh digunakan sebagai pengganti pemeriksaan, konsultasi, atau keputusan dari tenaga kesehatan profesional.
