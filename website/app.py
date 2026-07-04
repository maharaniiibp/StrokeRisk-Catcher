import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model


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


# =====================================================
# LOAD MODEL DAN PREPROCESSOR
# =====================================================

CNN_MODEL_PATH = os.path.join(
    MODEL_PATH,
    "cnn_model.keras"
)

SCALER_PATH = os.path.join(
    MODEL_PATH,
    "scaler.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_PATH,
    "label_encoders.pkl"
)


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
# NAMA DAN URUTAN FEATURE
# HARUS SAMA DENGAN DATA TRAINING
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
# FUNGSI PREPROCESSING INPUT WEBSITE
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

    # Encoding menggunakan encoder yang sama
    # seperti saat training.

    for column in categorical_columns:

        input_data[column] = label_encoders[
            column
        ].transform(
            input_data[column]
        )


    # Pastikan data numerik bertipe float.

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


    # Pastikan urutan kolom sama seperti training.

    input_data = input_data[
        FEATURE_COLUMNS
    ]


    # Scaling dengan scaler hasil training.

    input_scaled = scaler.transform(
        input_data
    )


    # CNN membutuhkan format:
    # (jumlah_sample, jumlah_feature, channel)

    input_cnn = input_scaled.reshape(
        input_scaled.shape[0],
        input_scaled.shape[1],
        1
    )

    return input_cnn


# =====================================================
# ROUTE HALAMAN UTAMA
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    probability = None

    error = None


    if request.method == "POST":

        try:

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


            # Preprocessing.

            input_cnn = preprocess_input(
                form_data
            )


            # Prediksi probabilitas stroke.

            prediction = model.predict(
                input_cnn,
                verbose=0
            )

            probability = float(
                prediction[0][0]
            )


            # Threshold sama dengan evaluasi model.

            predicted_class = int(
                probability >= 0.5
            )


            if predicted_class == 1:

                result = (
                    "Terindikasi Risiko Stroke"
                )

            else:

                result = (
                    "Tidak Terindikasi Risiko Stroke"
                )


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
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )