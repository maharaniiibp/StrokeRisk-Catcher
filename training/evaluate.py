import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# =====================================================
# PATH PROJECT
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models"
)

RESULT_PATH = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULT_PATH,
    exist_ok=True
)


# =====================================================
# LOAD DATA TESTING
# =====================================================

print("=" * 60)
print("MEMUAT DATA TESTING")
print("=" * 60)

data = np.load(
    os.path.join(
        MODEL_PATH,
        "preprocessed_data.npz"
    ),
    allow_pickle=True
)

X_test = data["X_test"]

y_test = data["y_test"]

print("Jumlah Data Testing :", X_test.shape[0])

print("Jumlah Feature      :", X_test.shape[1])


# =====================================================
# LOAD MODEL
# =====================================================

print("\n" + "=" * 60)
print("MEMUAT MODEL")
print("=" * 60)

mlp_model = load_model(
    os.path.join(
        MODEL_PATH,
        "mlp_model.keras"
    )
)

cnn_model = load_model(
    os.path.join(
        MODEL_PATH,
        "cnn_model.keras"
    )
)

print("MLP berhasil dimuat.")

print("CNN 1D berhasil dimuat.")


# =====================================================
# PREDIKSI MLP
# =====================================================

print("\nMelakukan prediksi menggunakan MLP...")

mlp_probability = mlp_model.predict(
    X_test,
    verbose=1
)

mlp_prediction = (
    mlp_probability >= 0.5
).astype(int).flatten()


# =====================================================
# RESHAPE DATA UNTUK CNN 1D
# =====================================================

X_test_cnn = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)


# =====================================================
# PREDIKSI CNN
# =====================================================

print("\nMelakukan prediksi menggunakan CNN 1D...")

cnn_probability = cnn_model.predict(
    X_test_cnn,
    verbose=1
)

cnn_prediction = (
    cnn_probability >= 0.5
).astype(int).flatten()


# =====================================================
# FUNCTION MENGHITUNG METRIK
# =====================================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }


# =====================================================
# HITUNG METRIK
# =====================================================

mlp_metrics = calculate_metrics(
    y_test,
    mlp_prediction
)

cnn_metrics = calculate_metrics(
    y_test,
    cnn_prediction
)


# =====================================================
# TAMPILKAN HASIL KOMPARASI
# =====================================================

print("\n" + "=" * 78)

print("HASIL EVALUASI DAN KOMPARASI MODEL")

print("=" * 78)

print(
    f"{'Model':<15}"
    f"{'Accuracy':<15}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1-Score':<15}"
)

print("-" * 78)

print(
    f"{'MLP':<15}"
    f"{mlp_metrics['Accuracy']:<15.4f}"
    f"{mlp_metrics['Precision']:<15.4f}"
    f"{mlp_metrics['Recall']:<15.4f}"
    f"{mlp_metrics['F1-Score']:<15.4f}"
)

print(
    f"{'CNN 1D':<15}"
    f"{cnn_metrics['Accuracy']:<15.4f}"
    f"{cnn_metrics['Precision']:<15.4f}"
    f"{cnn_metrics['Recall']:<15.4f}"
    f"{cnn_metrics['F1-Score']:<15.4f}"
)

print("=" * 78)


# =====================================================
# SIMPAN CSV KOMPARASI
# =====================================================

CSV_PATH = os.path.join(
    RESULT_PATH,
    "model_comparison.csv"
)

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ])

    writer.writerow([
        "MLP",
        mlp_metrics["Accuracy"],
        mlp_metrics["Precision"],
        mlp_metrics["Recall"],
        mlp_metrics["F1-Score"]
    ])

    writer.writerow([
        "CNN 1D",
        cnn_metrics["Accuracy"],
        cnn_metrics["Precision"],
        cnn_metrics["Recall"],
        cnn_metrics["F1-Score"]
    ])


# =====================================================
# GRAFIK KOMPARASI
# =====================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

mlp_values = [
    mlp_metrics[metric]
    for metric in metric_names
]

cnn_values = [
    cnn_metrics[metric]
    for metric in metric_names
]

x = np.arange(
    len(metric_names)
)

width = 0.35

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x - width / 2,
    mlp_values,
    width,
    label="MLP"
)

plt.bar(
    x + width / 2,
    cnn_values,
    width,
    label="CNN 1D"
)

plt.xlabel(
    "Metrik Evaluasi"
)

plt.ylabel(
    "Nilai"
)

plt.title(
    "Perbandingan Performa MLP dan CNN 1D"
)

plt.xticks(
    x,
    metric_names
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_PATH,
        "model_comparison.png"
    )
)

plt.close()


# =====================================================
# CONFUSION MATRIX MLP
# =====================================================

mlp_cm = confusion_matrix(
    y_test,
    mlp_prediction
)

mlp_display = ConfusionMatrixDisplay(
    confusion_matrix=mlp_cm,
    display_labels=[
        "Tidak Stroke",
        "Stroke"
    ]
)

mlp_display.plot()

plt.title(
    "Confusion Matrix MLP - Evaluasi Akhir"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_PATH,
        "evaluation_mlp_confusion_matrix.png"
    )
)

plt.close()


# =====================================================
# CONFUSION MATRIX CNN
# =====================================================

cnn_cm = confusion_matrix(
    y_test,
    cnn_prediction
)

cnn_display = ConfusionMatrixDisplay(
    confusion_matrix=cnn_cm,
    display_labels=[
        "Tidak Stroke",
        "Stroke"
    ]
)

cnn_display.plot()

plt.title(
    "Confusion Matrix CNN 1D - Evaluasi Akhir"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_PATH,
        "evaluation_cnn_confusion_matrix.png"
    )
)

plt.close()


# =====================================================
# MENENTUKAN MODEL TERBAIK
# =====================================================

# Studi kasus stroke merupakan klasifikasi tidak seimbang.
# F1-Score digunakan sebagai metrik utama karena mempertimbangkan
# keseimbangan antara Precision dan Recall.
#
# Jika F1-Score sama, Recall digunakan sebagai tie-breaker karena
# kemampuan mendeteksi pasien stroke penting dalam sistem skrining.

if mlp_metrics["F1-Score"] > cnn_metrics["F1-Score"]:

    best_model = "MLP"

    best_reason = (
        "MLP memiliki F1-Score lebih tinggi dibandingkan CNN 1D."
    )

elif cnn_metrics["F1-Score"] > mlp_metrics["F1-Score"]:

    best_model = "CNN 1D"

    best_reason = (
        "CNN 1D memiliki F1-Score lebih tinggi dibandingkan MLP."
    )

else:

    if mlp_metrics["Recall"] >= cnn_metrics["Recall"]:

        best_model = "MLP"

        best_reason = (
            "F1-Score kedua model sama, tetapi MLP "
            "memiliki Recall lebih tinggi atau sama."
        )

    else:

        best_model = "CNN 1D"

        best_reason = (
            "F1-Score kedua model sama, tetapi CNN 1D "
            "memiliki Recall lebih tinggi."
        )


# =====================================================
# SIMPAN REPORT KOMPARASI
# =====================================================

REPORT_PATH = os.path.join(
    RESULT_PATH,
    "model_comparison_report.txt"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "HASIL EVALUASI DAN KOMPARASI MODEL\n"
    )

    file.write(
        "=" * 65 + "\n\n"
    )

    file.write(
        "MLP\n"
    )

    file.write(
        f"Accuracy  : {mlp_metrics['Accuracy']:.4f}\n"
    )

    file.write(
        f"Precision : {mlp_metrics['Precision']:.4f}\n"
    )

    file.write(
        f"Recall    : {mlp_metrics['Recall']:.4f}\n"
    )

    file.write(
        f"F1-Score  : {mlp_metrics['F1-Score']:.4f}\n\n"
    )

    file.write(
        "CNN 1D\n"
    )

    file.write(
        f"Accuracy  : {cnn_metrics['Accuracy']:.4f}\n"
    )

    file.write(
        f"Precision : {cnn_metrics['Precision']:.4f}\n"
    )

    file.write(
        f"Recall    : {cnn_metrics['Recall']:.4f}\n"
    )

    file.write(
        f"F1-Score  : {cnn_metrics['F1-Score']:.4f}\n\n"
    )

    file.write(
        "=" * 65 + "\n"
    )

    file.write(
        f"MODEL TERBAIK : {best_model}\n"
    )

    file.write(
        f"ALASAN        : {best_reason}\n"
    )


# =====================================================
# HASIL AKHIR
# =====================================================

print("\nMODEL TERBAIK :", best_model)

print("ALASAN        :", best_reason)

print("\n" + "=" * 60)

print("EVALUASI DAN KOMPARASI SELESAI")

print("=" * 60)

print("\nFile yang dihasilkan:")

print("✔ results/model_comparison.csv")

print("✔ results/model_comparison.png")

print("✔ results/model_comparison_report.txt")

print("✔ results/evaluation_mlp_confusion_matrix.png")

print("✔ results/evaluation_cnn_confusion_matrix.png")