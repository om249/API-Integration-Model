from pathlib import Path
import joblib
from flask import Flask, render_template, request, jsonify

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "resume_classifier.joblib"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model not found. Run: python train_model.py")

model = joblib.load(MODEL_PATH)

EXAMPLES = {
    "Data Analyst": "Python, SQL, Pandas, Power BI, Excel, data cleaning, dashboard development and data visualization.",
    "Java Developer": "Java, OOP, Spring Boot, REST APIs, Hibernate, MySQL, Maven and backend development.",
    "Web Developer": "HTML, CSS, JavaScript, React, responsive design, REST API integration and frontend development.",
    "Data Scientist": "Python, Pandas, NumPy, Scikit-learn, machine learning, statistics, regression and classification.",
    "Software Developer": "Java, Python, DSA, OOP, algorithms, Git, SQL, debugging and software development."
}

def predict_role(text):
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    return ranked[0][0], float(ranked[0][1]), [
        {"role": role, "confidence": round(float(prob) * 100, 2)}
        for role, prob in ranked
    ]

@app.route("/")
def index():
    return render_template("index.html", examples=EXAMPLES)

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("resume") or "").strip()
    if not text:
        return jsonify({"error": "Please enter resume text."}), 400
    role, confidence, ranking = predict_role(text)
    return jsonify({
        "predicted_role": role,
        "confidence": round(confidence * 100, 2),
        "ranking": ranking
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "TF-IDF + Logistic Regression"})

if __name__ == "__main__":
    app.run(debug=True)
