from pathlib import Path
import re
import joblib


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "model" / "resume_classifier.joblib"
REFERENCE_PATH = ROOT / "model" / "reference_data.joblib"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "Model not found. Please run 'python train_model.py' first."
    )


if not REFERENCE_PATH.exists():

    raise FileNotFoundError(
        "Reference data not found. Please run 'python train_model.py' first."
    )


model = joblib.load(MODEL_PATH)

reference_data = joblib.load(
    REFERENCE_PATH
)


# --------------------------------------------------
# TECHNICAL / RESUME TERMS
# --------------------------------------------------

DOMAIN_TERMS = {

    "python",
    "java",
    "javascript",
    "typescript",

    "sql",
    "mysql",
    "mongodb",

    "power bi",
    "tableau",
    "excel",

    "pandas",
    "numpy",

    "scikit",
    "sklearn",

    "tensorflow",
    "pytorch",

    "machine learning",
    "deep learning",
    "nlp",

    "html",
    "css",
    "react",
    "node",

    "spring",
    "spring boot",
    "hibernate",

    "rest",
    "api",

    "git",
    "github",

    "docker",
    "aws",
    "azure",

    "dsa",
    "algorithms",
    "oop",

    "c++",
    "c",

    "flask",
    "django",

    "data analysis",
    "data analyst",
    "data scientist",

    "software developer",
    "web developer",

    "backend",
    "frontend",

    "dashboard",
    "statistics",
    "analytics"
}


# --------------------------------------------------
# ROLE-SPECIFIC KEYWORDS
# --------------------------------------------------

ROLE_TERMS = {

    "Data Analyst": {

        "python",
        "sql",
        "pandas",
        "power bi",
        "excel",
        "tableau",
        "dashboard",
        "analytics",
        "data visualization",
        "dax"
    },

    "Data Scientist": {

        "python",
        "machine learning",
        "scikit",
        "sklearn",
        "tensorflow",
        "pytorch",
        "statistics",
        "deep learning",
        "nlp",
        "regression",
        "classification"
    },

    "Java Developer": {

        "java",
        "spring",
        "spring boot",
        "hibernate",
        "jpa",
        "maven",
        "jdbc",
        "rest",
        "mysql",
        "microservices"
    },

    "Web Developer": {

        "html",
        "css",
        "javascript",
        "react",
        "frontend",
        "bootstrap",
        "tailwind",
        "jquery",
        "node",
        "responsive"
    },

    "Software Developer": {

        "java",
        "python",
        "dsa",
        "algorithms",
        "oop",
        "git",
        "sql",
        "debugging",
        "software development",
        "programming"
    }
}


# --------------------------------------------------
# NORMALIZE TEXT
# --------------------------------------------------

def normalize_text(text):

    return re.sub(
        r"\s+",
        " ",
        text.lower().strip()
    )


# --------------------------------------------------
# FIND TERMS
# --------------------------------------------------

def find_terms(text, vocabulary):

    normalized = normalize_text(text)

    found = []

    for term in vocabulary:

        pattern = (
            r"(?<!\w)"
            + re.escape(term)
            + r"(?!\w)"
        )

        if re.search(pattern, normalized):

            found.append(term)

    return sorted(
        found,
        key=len,
        reverse=True
    )


# --------------------------------------------------
# REFERENCE SIMILARITY
# --------------------------------------------------

def calculate_similarity(text):

    vectorizer = model.named_steps["tfidf"]

    reference_matrix = vectorizer.transform(
        reference_data["texts"]
    )

    input_vector = vectorizer.transform(
        [text]
    )

    scores = (
        reference_matrix @ input_vector.T
    ).toarray().ravel()

    if len(scores) == 0:

        return 0.0

    return float(
        scores.max()
    )


# --------------------------------------------------
# MAIN INTELLIGENT ANALYSIS
# --------------------------------------------------

def analyze_resume(text):

    text = text.strip()


    # ----------------------------------------------
    # CHECK 1 — EMPTY / SHORT INPUT
    # ----------------------------------------------

    if len(text) < 20:

        return {

            "status": "error",

            "message":
                "Please enter at least 20 characters of resume-related text.",

            "predicted_role": None,

            "confidence": 0,

            "similarity": 0,

            "matched_terms": [],

            "role_evidence": []
        }


    # ----------------------------------------------
    # CHECK 2 — DOMAIN DETECTION
    # ----------------------------------------------

    domain_terms = find_terms(
        text,
        DOMAIN_TERMS
    )


    # No technical/resume terms
    if not domain_terms:

        return {

            "status": "warning",

            "message":
                "No recognizable technical or resume-related terms were found. "
                "Prediction was blocked to avoid a misleading result.",

            "predicted_role": None,

            "confidence": 0,

            "similarity": 0,

            "matched_terms": [],

            "role_evidence": []
        }


    # ----------------------------------------------
    # MODEL PREDICTION
    # ----------------------------------------------

    probabilities = model.predict_proba(
        [text]
    )[0]

    classes = model.classes_


    ranked = sorted(

        zip(
            classes,
            probabilities
        ),

        key=lambda x: x[1],

        reverse=True
    )


    predicted_role = ranked[0][0]

    probability = ranked[0][1]


    # ----------------------------------------------
    # SIMILARITY
    # ----------------------------------------------

    similarity = calculate_similarity(
        text
    )


    # ----------------------------------------------
    # ROLE EVIDENCE
    # ----------------------------------------------

    role_keywords = ROLE_TERMS.get(
        predicted_role,
        set()
    )

    evidence = find_terms(
        text,
        role_keywords
    )


    # ----------------------------------------------
    # CONFIDENCE CHECK
    # ----------------------------------------------

    LOW_CONFIDENCE_THRESHOLD = 0.45

    LOW_SIMILARITY_THRESHOLD = 0.15


    low_confidence = (
        probability
        < LOW_CONFIDENCE_THRESHOLD
    )


    low_similarity = (
        similarity
        < LOW_SIMILARITY_THRESHOLD
    )


    # ----------------------------------------------
    # FINAL STATUS
    # ----------------------------------------------

    if low_confidence or low_similarity:

        status = "warning"

        message = (

            "A possible job role was found, "
            "but the input has low model confidence "
            "or low similarity to the reference data. "
            "Treat this result as low-confidence."

        )

    else:

        status = "success"

        message = (
            "Prediction generated successfully."
        )


    # ----------------------------------------------
    # ROLE RANKING
    # ----------------------------------------------

    ranking = []

    for role, probability_value in ranked:

        ranking.append({

            "role": role,

            "confidence":
                round(
                    float(probability_value) * 100,
                    2
                )
        })


    # ----------------------------------------------
    # FINAL RESPONSE
    # ----------------------------------------------

    return {

        "status": status,

        "message": message,

        "predicted_role":
            predicted_role,

        "confidence":
            round(
                float(probability) * 100,
                2
            ),

        "similarity":
            round(
                float(similarity) * 100,
                2
            ),

        "matched_terms":
            domain_terms[:12],

        "role_evidence":
            evidence[:10],

        "ranking":
            ranking
    }