from classifier_core import analyze_resume


# --------------------------------------------------
# EVALUATION TEST CASES
# --------------------------------------------------

test_cases = [

    (
        "Normal - Data Analyst",

        """
        Python SQL Pandas Power BI Excel
        dashboard data visualization
        data cleaning analytics
        """
    ),

    (
        "Normal - Java Developer",

        """
        Java Spring Boot Hibernate
        MySQL REST API Maven OOP
        backend development
        """
    ),

    (
        "Normal - Web Developer",

        """
        HTML CSS JavaScript React
        responsive frontend web development
        REST API
        """
    ),

    (
        "Normal - Data Scientist",

        """
        Python Pandas Scikit-learn
        machine learning statistics
        regression classification
        """
    ),

    (
        "Normal - Software Developer",

        """
        Java Python DSA OOP algorithms
        Git SQL debugging software development
        """
    ),

    (
        "Failure - Too Short",

        "Python"
    ),

    (
        "Failure - Irrelevant Input",

        """
        I enjoy photography, trekking,
        cooking and travelling.
        I have no technical experience.
        """
    )
]


# --------------------------------------------------
# RUN EVALUATION
# --------------------------------------------------

print("=" * 70)

print(
    "AI RESUME JOB CLASSIFIER"
)

print(
    "TASK 3 EVALUATION"
)

print("=" * 70)


for test_name, text in test_cases:


    result = analyze_resume(
        text
    )


    print("\n")
    print("-" * 70)


    print(
        f"TEST CASE: {test_name}"
    )


    print(
        f"STATUS: {result['status']}"
    )


    print(
        f"MESSAGE: {result['message']}"
    )


    print(
        f"ROLE: {result.get('predicted_role')}"
    )


    print(
        f"CONFIDENCE: "
        f"{result.get('confidence')}%"
    )


    print(
        f"SIMILARITY: "
        f"{result.get('similarity')}%"
    )


    print(
        f"EVIDENCE: "
        f"{result.get('role_evidence', [])}"
    )


print("\n")
print("=" * 70)

print(
    "Evaluation completed."
)

print("=" * 70)