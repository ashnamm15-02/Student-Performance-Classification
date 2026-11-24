
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load cleaned dataset
df = pd.read_csv("dataset/student_cleaned.csv")

# 2. Separate input features and target
X = df.drop(columns=[
    "student_id",
    "final_exam_score",
    "Result"
])

y = df["Result"]

# 3. Identify categorical and numerical columns
categorical_columns = [
    column for column in X.columns
    if pd.api.types.is_string_dtype(X[column].dtype)
]

numerical_columns = [
    column for column in X.columns
    if column not in categorical_columns
]

# 4. Encode categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_columns),

        ("numerical", "passthrough", numerical_columns)
    ]
)

# 5. Create Decision Tree model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),

    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ))
])

# 6. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["FAIL", "PASS"]
)

# 10. Display model results
print("\nMODEL TRAINING COMPLETED!")

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(cm)

# 11. Save trained model
joblib.dump(model, "model/student_model.pkl")

print("\nModel saved successfully!")

# 12. Save evaluation metrics
metrics = {
    "model_name": "Decision Tree Classifier",
    "accuracy": round(accuracy * 100, 2),
    "precision": round(
        report["weighted avg"]["precision"] * 100, 2
    ),
    "recall": round(
        report["weighted avg"]["recall"] * 100, 2
    ),
    "f1_score": round(
        report["weighted avg"]["f1-score"] * 100, 2
    ),
    "confusion_matrix": cm.tolist()
}

with open("model/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel metrics saved successfully!")
print("\nALL TASKS COMPLETED!")
