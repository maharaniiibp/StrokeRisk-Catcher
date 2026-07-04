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
    ConfusionMatrixDisplay,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# PATH
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models")
RESULT_PATH = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_PATH, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("MEMBACA DATA PREPROCESSING")
print("=" * 60)

data = np.load(
    os.path.join(MODEL_PATH, "preprocessed_data.npz"),
    allow_pickle=True
)

X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

print("Training :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing :", X_test.shape)

# =====================================================
# MEMBANGUN MODEL MLP
# =====================================================

print("\nMembangun Model MLP...\n")

model = Sequential([

    Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    BatchNormalization(),

    Dropout(0.30),

    Dense(
        64,
        activation="relu"
    ),

    BatchNormalization(),

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

    metrics=[
        "accuracy"
    ]

)

model.summary()

# =====================================================
# EARLY STOPPING
# =====================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)

# =====================================================
# TRAINING
# =====================================================

print("\nMulai Training...\n")

history = model.fit(

    X_train,

    y_train,

    validation_data=(

        X_val,

        y_val

    ),

    epochs=50,

    batch_size=32,

    callbacks=[

        early_stop

    ],

    verbose=1

)

# =====================================================
# SIMPAN MODEL
# =====================================================

model.save(

    os.path.join(

        MODEL_PATH,

        "mlp_model.keras"

    )

)

print("\nModel berhasil disimpan.")

# =====================================================
# GRAFIK ACCURACY
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("MLP Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(

    os.path.join(

        RESULT_PATH,

        "mlp_accuracy.png"

    )

)

plt.close()

# =====================================================
# GRAFIK LOSS
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(

    history.history["loss"],

    label="Training Loss"

)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss"

)

plt.title("MLP Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(

    os.path.join(

        RESULT_PATH,

        "mlp_loss.png"

    )

)

plt.close()

# =====================================================
# PREDIKSI
# =====================================================

print("\nMelakukan Prediksi...\n")

y_prob = model.predict(X_test)

y_pred = (y_prob > 0.5).astype(int).flatten()

# =====================================================
# METRIK
# =====================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)

precision = precision_score(

    y_test,

    y_pred

)

recall = recall_score(

    y_test,

    y_pred

)

f1 = f1_score(

    y_test,

    y_pred

)

print("="*60)

print("HASIL EVALUASI MLP")

print("="*60)

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

    confusion_matrix=cm

)

disp.plot()

plt.title("Confusion Matrix MLP")

plt.savefig(

    os.path.join(

        RESULT_PATH,

        "mlp_confusion_matrix.png"

    )

)

plt.close()

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(

    y_test,

    y_pred

)

with open(

    os.path.join(

        RESULT_PATH,

        "mlp_classification_report.txt"

    ),

    "w"

) as f:

    f.write(report)

print("\nClassification Report")

print(report)

# =====================================================
# SELESAI
# =====================================================

print("="*60)

print("TRAINING MLP SELESAI")

print("="*60)

print("\nFile yang dihasilkan :")

print("✔ mlp_model.keras")

print("✔ mlp_accuracy.png")

print("✔ mlp_loss.png")

print("✔ mlp_confusion_matrix.png")

print("✔ mlp_classification_report.txt")