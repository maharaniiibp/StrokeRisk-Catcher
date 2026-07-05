import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# =====================================================
# PATH
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models")
RESULT_PATH = os.path.join(BASE_DIR, "results")

# =====================================================
# LOAD DATA HASIL PREPROCESSING
# =====================================================

data = np.load(
    os.path.join(MODEL_PATH, "preprocessed_data.npz"),
    allow_pickle=True
)

X_test = data["X_test"]
y_test = data["y_test"]

# =====================================================
# LOAD MODEL CNN YANG SUDAH TERLATIH
# =====================================================

cnn_model = load_model(os.path.join(MODEL_PATH, "cnn_model.keras"))

# Reshape khusus untuk CNN (3 dimensi)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

y_prob_cnn = cnn_model.predict(X_test_cnn)
y_pred_cnn = (y_prob_cnn >= 0.5).astype(int).flatten()

# =====================================================
# AMBIL SAMPLE 10 DATA ACAK
# =====================================================

np.random.seed(42)
n_samples = 10
sample_idx = np.random.choice(len(X_test), n_samples, replace=False)

sample_results = pd.DataFrame({
    "No": range(1, n_samples + 1),
    "Label Aktual": y_test[sample_idx],
    "Prediksi Model": y_pred_cnn[sample_idx],
    "Probabilitas Stroke": y_prob_cnn[sample_idx].flatten().round(4),
})

sample_results["Status"] = np.where(
    sample_results["Label Aktual"] == sample_results["Prediksi Model"],
    "Benar", "Salah"
)

sample_results["Label Aktual"] = sample_results["Label Aktual"].map({0: "Tidak Stroke", 1: "Stroke"})
sample_results["Prediksi Model"] = sample_results["Prediksi Model"].map({0: "Tidak Stroke", 1: "Stroke"})

print(sample_results.to_string(index=False))

sample_results.to_csv(
    os.path.join(RESULT_PATH, "cnn_sample_prediction.csv"),
    index=False
)

print("\n✔ File cnn_sample_prediction.csv berhasil dibuat di folder results")