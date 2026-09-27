# LinkedIn Video Explanation Script — Task 2

Hello everyone.

For Task 2, I built a small AI prototype called **AI Resume Job Classifier**.

This project is based on the problem I defined in Task 1: using AI to classify resume text into a suitable job category.

The prototype uses Python, Flask and scikit-learn. The machine learning pipeline uses **TF-IDF for text feature extraction and Logistic Regression for classification**.

I created a small labeled dataset containing resume-related skills and information for five categories: Software Developer, Data Analyst, Data Scientist, Web Developer and Java Developer.

First, the resume text is entered into the web interface.

For example, if I enter Python, SQL, Pandas, Power BI and Excel, the model predicts the Data Analyst category.

The application also displays a confidence ranking for the available job roles.

The project exposes a Flask `/predict` API endpoint, so the model can also be integrated into another application.

No secret API key is used in this project because the prototype uses a local machine-learning model.

I also included the training script, dataset, requirements file, examples and complete setup instructions in the GitHub repository.

This prototype is designed only to assist with initial resume categorization. It should not be used as the sole basis for hiring decisions.

Thank you.
