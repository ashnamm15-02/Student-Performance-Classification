
from flask import Flask, redirect, render_template, request, url_for
import pandas as pd
import joblib
import json
from math import isfinite
import os
from pathlib import Path
import sqlite3
from contextlib import closing
from uuid import uuid4

app = Flask(__name__)

# Load trained model
model = joblib.load("model/student_model.pkl")

# Load cleaned dataset
df = pd.read_csv("dataset/student_cleaned.csv")
RESULT_LABELS = ("FAIL", "PASS")
FEEDBACK_DB_PATH = Path(os.environ.get(
    "STUDENT_FEEDBACK_DB",
    Path(app.instance_path) / "prediction_feedback.sqlite3"
))
FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def initialize_feedback_database():
    with closing(sqlite3.connect(FEEDBACK_DB_PATH)) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS prediction_feedback (
                    prediction_id TEXT PRIMARY KEY,
                    predicted_result TEXT NOT NULL
                        CHECK (predicted_result IN ('FAIL', 'PASS')),
                    actual_result TEXT
                        CHECK (actual_result IS NULL OR actual_result IN ('FAIL', 'PASS'))
                )
            """)


def get_feedback_confusion_matrix():
    matrix = [[0, 0], [0, 0]]
    with closing(sqlite3.connect(FEEDBACK_DB_PATH)) as connection:
        records = connection.execute(
            "SELECT actual_result, predicted_result "
            "FROM prediction_feedback WHERE actual_result IS NOT NULL"
        ).fetchall()

    label_indexes = {label: index for index, label in enumerate(RESULT_LABELS)}
    for actual_result, predicted_result in records:
        matrix[label_indexes[actual_result]][label_indexes[predicted_result]] += 1

    return matrix


def update_metrics_with_feedback(metrics):
    metrics = dict(metrics)
    feedback_matrix = get_feedback_confusion_matrix()
    matrix = [
        [
            int(metrics["confusion_matrix"][row][column])
            + feedback_matrix[row][column]
            for column in range(2)
        ]
        for row in range(2)
    ]
    metrics["confusion_matrix"] = matrix
    metrics["feedback_count"] = sum(sum(row) for row in feedback_matrix)

    total = sum(sum(row) for row in matrix)
    if total:
        support = [sum(row) for row in matrix]
        predicted_count = [
            sum(matrix[row][column] for row in range(2))
            for column in range(2)
        ]
        precision = [
            matrix[index][index] / predicted_count[index]
            if predicted_count[index] else 0
            for index in range(2)
        ]
        recall = [
            matrix[index][index] / support[index]
            if support[index] else 0
            for index in range(2)
        ]
        f1_score = [
            2 * precision[index] * recall[index]
            / (precision[index] + recall[index])
            if precision[index] + recall[index] else 0
            for index in range(2)
        ]

        metrics["accuracy"] = round(
            (matrix[0][0] + matrix[1][1]) / total * 100, 2
        )
        metrics["precision"] = round(
            sum(support[index] * precision[index] for index in range(2))
            / total * 100, 2
        )
        metrics["recall"] = round(
            sum(support[index] * recall[index] for index in range(2))
            / total * 100, 2
        )
        metrics["f1_score"] = round(
            sum(support[index] * f1_score[index] for index in range(2))
            / total * 100, 2
        )

    return metrics


initialize_feedback_database()

# Model input columns
feature_columns = [
    "age",
    "gender",
    "country",
    "major",
    "university_year",
    "GPA",
    "study_hours_per_day",
    "class_attendance_percent",
    "sleep_hours",
    "screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "exercise_hours_per_week",
    "part_time_job",
    "relationship_status",
    "family_income_level",
    "internet_quality",
    "mental_stress_level",
    "AI_tool_usage_hours",
    "favorite_AI_tool",
    "note_taking_method",
    "exam_preparation_days",
    "coffee_consumption_per_day",
    "extracurricular_hours_per_week",
    "assignment_score"
]

# Create form fields automatically
form_fields = [{
    "name": "student_name",
    "label": "Student Name",
    "type": "text"
}]
category_options = {}

for column in feature_columns:

    if pd.api.types.is_string_dtype(df[column].dtype):
        options = sorted(
            df[column].dropna().astype(str).unique().tolist()
        )
        category_options[column] = options

        form_fields.append({
            "name": column,
            "type": "select",
            "options": options
        })

    else:

        form_fields.append({
            "name": column,
            "type": "number",
            "options": []
        })


@app.route("/")
def home():
    return render_template(
        "index.html",
        form_fields=form_fields
    )


@app.route("/predict", methods=["POST"])
def predict():
    student_name = request.form.get("student_name", "").strip()

    if not student_name:
        return render_template(
            "message.html",
            message="Please enter the student's name."
        ), 400

    input_data = {}

    for column in feature_columns:
        value = request.form.get(column, "").strip()

        if pd.api.types.is_string_dtype(df[column].dtype):
            if value not in category_options[column]:
                return render_template(
                    "message.html",
                    message="Please select a valid option for "
                    + column.replace("_", " ") + "."
                ), 400

            input_data[column] = value
        else:
            if not value:
                return render_template(
                    "message.html",
                    message="Please enter a number for "
                    + column.replace("_", " ") + "."
                ), 400

            try:
                input_data[column] = float(value)
            except ValueError:
                return render_template(
                    "message.html",
                    message="Please enter a valid number for "
                    + column.replace("_", " ") + "."
                ), 400

            if not isfinite(input_data[column]):
                return render_template(
                    "message.html",
                    message="Please enter a finite number for "
                    + column.replace("_", " ") + "."
                ), 400

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    prediction = model.predict(input_df)[0]
    prediction_id = uuid4().hex
    with closing(sqlite3.connect(FEEDBACK_DB_PATH)) as connection:
        with connection:
            connection.execute(
                "INSERT INTO prediction_feedback "
                "(prediction_id, predicted_result) VALUES (?, ?)",
                (prediction_id, prediction)
            )

    return render_template(
        "result.html",
        prediction=prediction,
        student_name=student_name,
        prediction_id=prediction_id
    )


@app.route("/record_actual_result", methods=["POST"])
def record_actual_result():
    prediction_id = request.form.get("prediction_id", "").strip()
    actual_result = request.form.get("actual_result", "").strip()

    if actual_result not in RESULT_LABELS:
        return render_template(
            "message.html",
            message="Please select the student's actual PASS or FAIL result."
        ), 400

    with closing(sqlite3.connect(FEEDBACK_DB_PATH)) as connection:
        with connection:
            cursor = connection.execute(
                "UPDATE prediction_feedback SET actual_result = ? "
                "WHERE prediction_id = ? AND actual_result IS NULL",
                (actual_result, prediction_id)
            )

    if cursor.rowcount != 1:
        return render_template(
            "message.html",
            message="This prediction was not found or its actual result "
            "has already been submitted."
        ), 400

    return redirect(url_for("accuracy", feedback_saved=1))


@app.route("/accuracy")
def accuracy():

    with open("model/metrics.json", "r") as file:
        metrics = json.load(file)

    return render_template(
        "accuracy.html",
        metrics=update_metrics_with_feedback(metrics),
        feedback_saved=request.args.get("feedback_saved") == "1"
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
