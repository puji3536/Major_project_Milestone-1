from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

app = Flask(__name__)

# Load dataset
df = pd.read_csv("diseases.csv")

# Identify symptom columns dynamically
symptom_columns = [col for col in df.columns if "Symptom" in col]
df[symptom_columns] = df[symptom_columns].fillna("")  # Handle missing values

# Collect all unique symptoms (case insensitive)
all_symptoms = set()
for col in symptom_columns:
    all_symptoms.update(df[col].dropna().str.lower().unique())

# Encode symptoms using One-Hot Encoding
mlb = MultiLabelBinarizer()
mlb.fit([list(all_symptoms)])

# Convert symptoms into binary format for training
df["Symptoms"] = df[symptom_columns].apply(lambda x: list(filter(None, x.str.lower())), axis=1)
X = mlb.transform(df["Symptoms"])

# Encode diseases
disease_encoder = LabelEncoder()
df["EncodedDisease"] = disease_encoder.fit_transform(df["Disease"])
y = df["EncodedDisease"]

# Train Decision Tree Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

def preprocess_input(symptoms):
    """Convert user symptoms into encoded format"""
    symptoms = [s.strip().lower() for s in symptoms.split(",") if s.strip()]

    # Ensure at least 3 symptoms are given
    if len(symptoms) < 3:
        return "⚠️ Please enter at least 3 symptoms!", None

    # Encode symptoms
    symptoms_encoded = mlb.transform([symptoms])
    return None, symptoms_encoded

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "clear" in request.form:
            return render_template("index.html", clear=True)

        symptoms = request.form["symptoms"]
        warning, encoded_symptoms = preprocess_input(symptoms)

        if warning:
            return render_template("index.html", warning=warning, clear=False)

        # Predict Disease
        prediction = model.predict(encoded_symptoms)
        disease = disease_encoder.inverse_transform(prediction)[0]

        # Get Disease Details
        details = df[df["Disease"] == disease].iloc[0]

        return render_template(
            "index.html",
            result=disease,
            precautions=details["Precautions"],
            medications=details["Medications"],
            specialist=details["Specialist"],
            clear=False
        )

    return render_template("index.html", clear=False)

if __name__ == "__main__":
    app.run(debug=True)
