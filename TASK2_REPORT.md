# Task 2 — Model or API Integration

## Project
AI Resume Job Classifier

## Objective
Build a small AI prototype that predicts a suitable job role from resume-related text.

## Model
TF-IDF + Logistic Regression using scikit-learn.

## Dataset
50 synthetic resume-skill descriptions across five job-role categories.

## Evaluation
The model uses an 80/20 stratified train-test split and reports accuracy, precision, recall and F1-score.

### Actual training output

```text
Test accuracy: 100.00%
                    precision    recall  f1-score   support

      Data Analyst       1.00      1.00      1.00         3
    Data Scientist       1.00      1.00      1.00         2
    Java Developer       1.00      1.00      1.00         3
Software Developer       1.00      1.00      1.00         3
     Web Developer       1.00      1.00      1.00         2

          accuracy                           1.00        13
         macro avg       1.00      1.00      1.00        13
      weighted avg       1.00      1.00      1.00        13

Saved model to: /mnt/data/AI-Resume-Job-Classifier/model/resume_classifier.joblib

Spreadsheet runtime warmup failed during python startup
Traceback (most recent call last):
  File "/tmp/tmp.L2TH2Y5coc/artifact_tool_v2-2.8.22/artifact_tool/patches/warm_spreadsheet_runtime_on_startup.py", line 26, in warm_spreadsheet_runtime_on_startup
  File "/tmp/tmp.L2TH2Y5coc/artifact_tool_v2-2.8.22/artifact_tool/spreadsheet_warmup.py", line 785, in warm_spreadsheet_runtime
  File "/tmp/tmp.L2TH2Y5coc/artifact_tool_v2-2.8.22/artifact_tool/spreadsheet_warmup.py", line 720, in _warm_feature_flows
  File "/tmp/tmp.L2TH2Y5coc/artifact_tool_v2-2.8.22/artifact_tool/spreadsheet_warmup.py", line 704, in _warm_collaboration_flows
  File "/tmp/tmp.L2TH2Y5coc/artifact_tool_v2-2.8.22/artifact_tool/generated/interface/models.py", line 32317, in hydrate_crdt_from_proto
  File "/tmp/tmp.L2TH2Y5coc/artifact_tool_v2-2.8.22/artifact_tool/rpc/remote.py", line 749, in __call__
  File "/tmp/tmp.L2TH2Y5coc/artifact_tool_v2-2.8.22/artifact_tool/rpc/client.py", line 150, in call
artifact_tool.rpc.client.RemoteError: hydrateCrdtFromProto requires an empty collaborative document.
```

## Example
Input: Python, SQL, Pandas, Power BI, Excel, dashboard development

Expected category: Data Analyst

## Secret keys
None. The prototype uses a local scikit-learn model and does not require an API key.
