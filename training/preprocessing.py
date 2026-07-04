import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# =====================================================
# PATH PROJECT
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "dataset", "stroke.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_PATH, exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("MEMBACA DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(df.head())

# =====================================================
# INFORMASI DATASET
# =====================================================

print("\n" + "=" * 60)
print("INFORMASI DATASET")
print("=" * 60)

print(df.info())

print("\nJumlah Data :", df.shape[0])
print("Jumlah Kolom :", df.shape[1])

print("\nMissing Value")
print(df.isnull().sum())

# =====================================================
# DROP ID
# =====================================================

print("\nMenghapus kolom ID...")

df.drop("id", axis=1, inplace=True)

# =====================================================
# HAPUS GENDER OTHER
# =====================================================

print("Menghapus data dengan gender = Other...")

df = df[df["gender"] != "Other"]

# =====================================================
# MISSING VALUE BMI
# =====================================================

print("Mengisi missing value BMI...")

df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# =====================================================
# DUPLICATE
# =====================================================

print("Menghapus duplicate...")

df.drop_duplicates(inplace=True)

# =====================================================
# ENCODING
# =====================================================

print("Melakukan Label Encoding...")

categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

label_encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col])

    label_encoders[col] = encoder

joblib.dump(
    label_encoders,
    os.path.join(MODEL_PATH, "label_encoders.pkl")
)

# =====================================================
# FEATURE & TARGET
# =====================================================

X = df.drop("stroke", axis=1)

y = df["stroke"]

print("\nJumlah Feature :", X.shape[1])

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nData Training :", X_train.shape)
print("Data Validation :", X_val.shape)
print("Data Testing :", X_test.shape)

# =====================================================
# SCALING
# =====================================================

print("\nMelakukan Standard Scaling...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    os.path.join(MODEL_PATH, "scaler.pkl")
)

# =====================================================
# SMOTE
# =====================================================

print("Menjalankan SMOTE...")

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nSetelah SMOTE")

print(pd.Series(y_train).value_counts())

# =====================================================
# SIMPAN DATA
# =====================================================

np.savez(
    os.path.join(MODEL_PATH, "preprocessed_data.npz"),
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test
)

print("\n" + "=" * 60)
print("PREPROCESSING SELESAI")
print("=" * 60)

print("\nFile yang berhasil dibuat :")

print("✔ scaler.pkl")

print("✔ label_encoders.pkl")

print("✔ preprocessed_data.npz")