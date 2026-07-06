import os
import joblib
import pandas as pd

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model


# =====================================================
# PATH PROJECT
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models")

CNN_MODEL_PATH = os.path.join(MODEL_PATH, "cnn_model.keras")
SCALER_PATH = os.path.join(MODEL_PATH, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_PATH, "label_encoders.pkl")


# =====================================================
# LOAD MODEL DAN PREPROCESSOR
# =====================================================

print("=" * 60)
print("MEMUAT MODEL DAN PREPROCESSOR")
print("=" * 60)

model = load_model(CNN_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODER_PATH)

print("CNN 1D berhasil dimuat.")
print("Scaler berhasil dimuat.")
print("Label Encoder berhasil dimuat.")


# =====================================================
# FLASK APPLICATION
# =====================================================

app = Flask(__name__)


# =====================================================
# FEATURE COLUMNS
# =====================================================

FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status"
]


# =====================================================
# PREPROCESSING INPUT
# =====================================================

def preprocess_input(form_data):

    input_data = pd.DataFrame(
        [form_data],
        columns=FEATURE_COLUMNS
    )

    categorical_columns = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status"
    ]

    for column in categorical_columns:
        input_data[column] = label_encoders[column].transform(
            input_data[column]
        )

    numeric_columns = [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi"
    ]

    for column in numeric_columns:
        input_data[column] = pd.to_numeric(
            input_data[column]
        )

    # Memastikan urutan fitur sama dengan saat training
    input_data = input_data[FEATURE_COLUMNS]

    # Standardisasi menggunakan scaler hasil training
    input_scaled = scaler.transform(input_data)

    # Mengubah bentuk data agar sesuai input CNN 1D
    input_cnn = input_scaled.reshape(
        input_scaled.shape[0],
        input_scaled.shape[1],
        1
    )

    return input_cnn


# =====================================================
# LANDING PAGE
# =====================================================

@app.route("/", methods=["GET"])
def landing():

    return render_template("landing.html")


# =====================================================
# WEBSITE MACHINE LEARNING / PREDICTION
# =====================================================

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    result = None
    probability = None
    error = None

    if request.method == "POST":

        try:

            # Mengambil data dari form HTML
            form_data = {

                "gender":
                    request.form["gender"],

                "age":
                    request.form["age"],

                "hypertension":
                    request.form["hypertension"],

                "heart_disease":
                    request.form["heart_disease"],

                "ever_married":
                    request.form["ever_married"],

                "work_type":
                    request.form["work_type"],

                "Residence_type":
                    request.form["Residence_type"],

                "avg_glucose_level":
                    request.form["avg_glucose_level"],

                "bmi":
                    request.form["bmi"],

                "smoking_status":
                    request.form["smoking_status"]

            }

            # Preprocessing data
            input_cnn = preprocess_input(form_data)

            # Melakukan prediksi menggunakan CNN 1D
            prediction_result = model.predict(
                input_cnn,
                verbose=0
            )

            # Mengambil probabilitas prediksi
            probability = float(
                prediction_result[0][0]
            )

            # Threshold klasifikasi
            predicted_class = int(
                probability >= 0.5
            )

            # Menentukan hasil prediksi
            result = (
                "Terindikasi Risiko Stroke"
                if predicted_class == 1
                else
                "Tidak Terindikasi Risiko Stroke"
            )

            # Mengubah probabilitas menjadi persen
            probability = round(
                probability * 100,
                2
            )

        except Exception as exception:

            error = str(exception)

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        error=error
    )


# =====================================================
# RUN FLASK
# =====================================================

if __name__ == "__main__":

    app.run(debug=True)