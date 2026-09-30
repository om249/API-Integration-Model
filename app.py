from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from classifier_core import analyze_resume


# --------------------------------------------------
# FLASK APPLICATION
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# EXAMPLE INPUTS
# --------------------------------------------------

EXAMPLES = {

    "Data Analyst":

        "Python, SQL, Pandas, Power BI, Excel, "
        "data cleaning, dashboard development "
        "and data visualization.",


    "Java Developer":

        "Java, OOP, Spring Boot, REST APIs, "
        "Hibernate, MySQL, Maven and backend development.",


    "Web Developer":

        "HTML, CSS, JavaScript, React, "
        "responsive design, REST API integration "
        "and frontend development.",


    "Data Scientist":

        "Python, Pandas, NumPy, Scikit-learn, "
        "machine learning, statistics, regression "
        "and classification.",


    "Software Developer":

        "Java, Python, DSA, OOP, algorithms, "
        "Git, SQL, debugging and software development.",


    "Failure Case":

        "I enjoy photography, trekking, cooking "
        "and travelling. I have no technical experience."
}


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():

    return render_template(
        "index.html",
        examples=EXAMPLES
    )


# --------------------------------------------------
# PREDICTION API
# --------------------------------------------------

@app.post("/predict")
def predict():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        text = str(
            data.get("resume") or ""
        ).strip()


        result = analyze_resume(
            text
        )


        return jsonify(
            result
        )


    except Exception:

        app.logger.exception(
            "Prediction error"
        )


        return jsonify({

            "status": "error",

            "message":
                "The prediction service encountered "
                "an unexpected error."
        }), 500


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return jsonify({

        "status": "ok",

        "model":
            "TF-IDF + Logistic Regression",

        "features": [

            "input validation",

            "technical-domain detection",

            "confidence checking",

            "reference similarity",

            "prediction evidence",

            "error handling"
        ]
    })


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )