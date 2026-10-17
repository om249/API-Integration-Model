# AI Resume Job Classifier

An AI-powered web application that analyzes resume-related text and predicts a possible job category using Natural Language Processing (NLP) and Machine Learning.

**Task 4 – Final AI Application**

## 1. Project Overview

The AI Resume Job Classifier is designed to demonstrate how machine learning can help organize resume text into potential technical job categories.

The application analyzes the text provided by the user, identifies relevant technical terms, predicts a possible job role, and displays confidence, reference similarity, and supporting keywords. It also handles insufficient or unrelated input by displaying an error or warning.

This is an educational prototype, not a professional recruitment or hiring system.

## 2. Problem Statement

Manually organizing large numbers of resumes into suitable job categories can be time-consuming. This project explores how text classification can assist with the initial categorization of resume-related information.

The application supports five job categories:

- Data Analyst
- Data Scientist
- Java Developer
- Web Developer
- Software Developer

## 3. Objectives

- Build a beginner-friendly AI web application.
- Integrate a machine learning model into a Flask application.
- Predict potential job categories from resume text.
- Display model confidence and similarity to reference data.
- Show relevant keywords that support the predicted category.
- Handle empty, short, and unrelated inputs.
- Demonstrate model evaluation using normal and failure test cases.
- Document limitations, privacy, and ethical considerations.

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and model logic |
| Flask | Web application and prediction API |
| pandas | Dataset loading and processing |
| scikit-learn | Machine learning pipeline |
| TF-IDF | Text feature extraction |
| Logistic Regression | Job category classification |
| joblib | Saving and loading the trained model |
| HTML, CSS and JavaScript | Web interface |

The application uses a locally trained model and does not require an external AI API key.

## 5. How It Works

1. The user enters resume-related text.
2. The application validates the input.
3. A keyword-based check looks for recognizable technical or resume-related terms.
4. The TF-IDF and Logistic Regression pipeline predicts a job category.
5. The application calculates similarity to the saved reference texts.
6. Relevant keywords and category rankings are returned.
7. The application displays a prediction or a warning when the result may be unreliable.

### Architecture

```text
User enters resume text
          |
          v
     Flask interface
          |
          v
     Input validation
          |
          v
 Technical-domain detection
          |
          v
 TF-IDF + Logistic Regression
          |
          v
 Confidence and similarity checks
          |
          v
 Prediction + keyword evidence
          |
          v
 Result, warning, or error
```

## 6. Intelligent Features

### Resume Text Classification
Uses a TF-IDF vectorizer and Logistic Regression model to classify input into one of five job categories.

### Input Validation
Rejects text that is shorter than the configured minimum length.

### Technical-Domain Detection
Checks whether the input contains recognized technical or resume-related terms before attempting classification.

### Confidence Checking
Displays the model's highest class probability and warns when it falls below a prototype threshold.

### Reference Similarity
Compares the input's TF-IDF representation with the saved reference texts and reports the highest similarity score.

### Prediction Evidence
Displays matching technical keywords associated with the predicted category.

### Error Handling
Returns understandable error or warning messages for invalid, unrelated, or unsuccessful prediction requests.

**Important:** Confidence and similarity are different measures. Neither should be interpreted as a calibrated probability that the prediction is correct.

## 7. Dataset and Model

The original project report describes a dataset of 50 synthetic resume-skill descriptions across five job categories.

The model uses a stratified train-test split, with 75% of the records used for training and 25% for testing.

The training script reports accuracy, precision, recall, F1-score, and a confusion matrix.

Because the dataset is small and synthetic, its evaluation results do not establish performance on real-world resumes. Run the training script to obtain the metrics for your current dataset and model.

## 8. Project Structure

```text
AI-Resume-Job-Classifier/
├── app.py
├── classifier_core.py
├── train_model.py
├── evaluate.py
├── test_api.py
├── requirements.txt
├── README.md
├── RUN_WINDOWS.md
├── TASK2_REPORT.md
├── data/
│   └── resumes.csv
├── model/
│   ├── resume_classifier.joblib
│   └── reference_data.joblib
├── templates/
│   └── index.html
└── demo/
    ├── example_inputs_outputs.md
    └── task4_video_script.md
```

The two files in `model/` are generated by running `train_model.py`. The `data/`, `templates/`, and `demo/` directories must contain the corresponding project files.

## 9. Installation and Setup

### Prerequisites

- Python installed
- Git installed (for cloning the repository)
- A web browser

### Step 1: Clone the repository

```bash
git clone https://github.com/om249/API-Integration-Model.git
cd API-Integration-Model
```

### Step 2: Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Train the model

```bash
python train_model.py
```

This creates the trained classifier and reference data under the `model/` directory.

### Step 5: Start the application

```bash
python app.py
```

Open the local URL displayed in the terminal, normally:

http://127.0.0.1:5000

Use one of the example inputs or enter your own resume-related text.

## 10. Testing and Evaluation

Run the evaluation script:

```bash
python evaluate.py
```

The test cases include:

- Data Analyst resume text
- Java Developer resume text
- Web Developer resume text
- Data Scientist resume text
- Software Developer resume text
- Insufficient input
- Irrelevant input

The results demonstrate how the application responds to different inputs. They are examples of application behavior, not proof of production-level accuracy.

### API Test

Start the Flask application and run the following in a second terminal:

```bash
python test_api.py
```

The prediction endpoint is:

```text
POST /predict
```

It accepts JSON containing a `resume` field.

A health-check endpoint is also available:

```text
GET /health
```

## 11. Example Usage

**Example: Data Analyst**

Input:

```text
Python, SQL, Pandas, Power BI, Excel,
dashboard development and data visualization.
```

Expected category to test: Data Analyst.

**Example: Insufficient input**

Input:

```text
Python
```

Expected behavior: an input validation error.

**Example: Unrelated input**

Input:

```text
I enjoy photography, trekking, cooking and travelling.
I have no technical experience.
```

Expected behavior: a warning that recognizable technical or resume-related terms were not found.

Exact predictions and numeric scores should be taken from the running application rather than assumed in advance.

## 12. Limitations

- The dataset is small and synthetic.
- Performance on real resumes has not been established.
- The classifier supports only five predefined job categories.
- Important context, work experience, education, and transferable skills may be missed.
- Keyword detection may reject valid resumes that use unfamiliar terminology.
- Similarity measures text overlap in the TF-IDF feature space; it does not establish semantic equivalence.
- Model confidence is not a calibrated measure of correctness.
- The application does not currently establish whether a candidate is qualified for a specific job description.
- No production-grade authentication, database, or deployment infrastructure is included in the current prototype.

## 13. Ethics, Privacy and Responsible Use

- Use synthetic or appropriately authorized data for demonstrations.
- Do not upload real resumes containing personal information to a public repository.
- Remove unnecessary personal identifiers before testing.
- Do not use the classifier as the sole basis for hiring, rejection, or other consequential decisions.
- Do not interpret the predicted category as a measure of a person's ability or potential.
- Evaluate data quality, bias, and performance across relevant groups before considering real-world use.
- Keep a human reviewer involved in any recruitment-related workflow.
- Be transparent that the application is experimental and may produce incorrect predictions.

## 14. Future Improvements

- PDF and DOCX resume upload.
- Job description matching.
- Larger and more diverse datasets.
- Better evaluation on unseen, representative data.
- Semantic embeddings for text similarity.
- More comprehensive skill extraction and explanations.
- Additional job categories.
- A recruiter dashboard and secure data handling.
- Model comparison and systematic bias evaluation.
- Deployment with appropriate monitoring and access controls.

## 15. Final Outcome

This project demonstrates the integration of machine learning, text processing, input validation, similarity analysis, and a Flask interface into one beginner-friendly AI application.

It builds on the model integration work from Task 2 and the intelligent analysis enhancements from Task 3, while documenting the methodology, demo instructions, limitations, and ethical considerations required for the final project.

## 16. Repository

GitHub: https://github.com/om249/API-Integration-Model

**Project:** AI Resume Job Classifier  
**Task:** Task 4 – Final AI Application  
**Status:** Educational prototype
