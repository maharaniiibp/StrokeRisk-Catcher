import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# PATH
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR,"models")
RESULT_PATH = os.path.join(BASE_DIR,"results")

os.makedirs(RESULT_PATH,exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

data=np.load(

    os.path.join(

        MODEL_PATH,

        "preprocessed_data.npz"

    ),

    allow_pickle=True

)

X_train=data["X_train"]
X_val=data["X_val"]
X_test=data["X_test"]

y_train=data["y_train"]
y_val=data["y_val"]
y_test=data["y_test"]

# =====================================================
# RESHAPE UNTUK CNN
# =====================================================

X_train=X_train.reshape(

    X_train.shape[0],

    X_train.shape[1],

    1

)

X_val=X_val.reshape(

    X_val.shape[0],

    X_val.shape[1],

    1

)

X_test=X_test.reshape(

    X_test.shape[0],

    X_test.shape[1],

    1

)

print(X_train.shape)

# =====================================================
# MODEL CNN
# =====================================================

model=Sequential([

    Conv1D(

        filters=64,

        kernel_size=2,

        activation="relu",

        input_shape=(10,1)

    ),

    BatchNormalization(),

    MaxPooling1D(

        pool_size=2

    ),

    Dropout(0.30),

    Conv1D(

        filters=32,

        kernel_size=2,

        activation="relu"

    ),

    BatchNormalization(),

    Flatten(),

    Dense(

        64,

        activation="relu"

    ),

    Dropout(0.20),

    Dense(

        32,

        activation="relu"

    ),

    Dense(

        1,

        activation="sigmoid"

    )

])

model.compile(

    optimizer=Adam(

        learning_rate=0.001

    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

model.summary()

early_stop=EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)
# =====================================================
# TRAINING CNN
# =====================================================

print("\n" + "=" * 60)
print("MULAI TRAINING CNN 1D")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =====================================================
# SIMPAN MODEL CNN
# =====================================================

CNN_MODEL_FILE = os.path.join(
    MODEL_PATH,
    "cnn_model.keras"
)

model.save(CNN_MODEL_FILE)

print("\nModel CNN berhasil disimpan.")

# =====================================================
# GRAFIK ACCURACY
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN 1D Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_PATH,
        "cnn_accuracy.png"
    )
)

plt.close()

# =====================================================
# GRAFIK LOSS
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN 1D Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_PATH,
        "cnn_loss.png"
    )
)

plt.close()

# =====================================================
# PREDIKSI DATA TESTING
# =====================================================

print("\nMelakukan prediksi data testing...")

y_prob = model.predict(
    X_test,
    verbose=1
)

y_pred = (
    y_prob >= 0.5
).astype(int).flatten()

# =====================================================
# HITUNG METRIK EVALUASI
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n" + "=" * 60)
print("HASIL EVALUASI CNN 1D")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Tidak Stroke",
        "Stroke"
    ]
)

disp.plot()

plt.title("Confusion Matrix CNN 1D")
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_PATH,
        "cnn_confusion_matrix.png"
    )
)

plt.close()

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Tidak Stroke",
        "Stroke"
    ],
    zero_division=0
)

print("\nCLASSIFICATION REPORT CNN 1D")
print(report)

with open(
    os.path.join(
        RESULT_PATH,
        "cnn_classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "HASIL EVALUASI CNN 1D\n"
    )

    file.write(
        "=" * 60 + "\n"
    )

    file.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )

    file.write(
        f"Precision : {precision:.4f}\n"
    )

    file.write(
        f"Recall    : {recall:.4f}\n"
    )

    file.write(
        f"F1 Score  : {f1:.4f}\n\n"
    )

    file.write(
        report
    )

# =====================================================
# SIMPAN METRIK CNN
# =====================================================

np.savez(
    os.path.join(
        RESULT_PATH,
        "cnn_metrics.npz"
    ),
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1_score=f1
)

# =====================================================
# SELESAI
# =====================================================

print("\n" + "=" * 60)
print("TRAINING DAN EVALUASI CNN 1D SELESAI")
print("=" * 60)

print("\nFile yang dihasilkan:")

print("✔ models/cnn_model.keras")
print("✔ results/cnn_accuracy.png")
print("✔ results/cnn_loss.png")
print("✔ results/cnn_confusion_matrix.png")
print("✔ results/cnn_classification_report.txt")
print("✔ results/cnn_metrics.npz")