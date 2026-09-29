MPLAD-AI

AI-Powered MPLAD Risk & Fraud Intelligence Platform

MPLAD-AI is an AI-assisted decision-support platform designed to analyze MPLAD works, identify unusual patterns, prioritize works for review, and provide explainable indicators to support officer investigation.

Important: An anomaly or high priority score does not confirm fraud and does not make an official approval, rejection, or fraud determination.

Project Objective

MPLAD-AI helps officers screen MPLAD works by:

Analyzing historical work patterns

Detecting unusual combinations of cost and delay indicators

Assigning a relative priority risk score

Providing explainable assessment reasons

Allowing individual Work ID inspection

Allowing analysis of a new work

Key Features

Dashboard

Total works analyzed

AI-detected anomalies

High and elevated priority works

Priority risk distribution

Top states by AI-detected anomalies

Key dataset insights

Priority Review

Filter by AI status

Filter by risk level

Set minimum risk score

Search and inspect Work ID

View risk score and reasons

View recommended review action

Check New Work

Users can enter:

State

Work category

Sanction amount

Sanction delay

Work description

The system generates:

Risk score

Risk level

AI anomaly status

Cost deviation indicators

Delay indicators

Assessment reasons

Machine Learning

MPLAD-AI uses an Isolation Forest model for unsupervised anomaly detection.

The model uses:

LOG_COST_DEVIATION

LOG_STATE_COST_DEVIATION

LOG_SANCTION_DELAY

LOG_STATE_DELAY_DEVIATION

DESCRIPTION_WORDS

The model produces:

Anomaly Detected

No Anomaly Detected

An anomaly means an unusual pattern in the analyzed data. It does not confirm fraud.

Priority Risk Score

The separate Priority Risk Score uses relative:

Cost vs category median

Cost vs state median

Sanction delay

Sanction delay vs state median

The score is a review-priority indicator, not a fraud probability.

Technology Stack

Python

Streamlit

Pandas

NumPy

Scikit-learn

Joblib

Project Structure

MPLAD-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── mplad_work_clean.pkl
├── mplad_isolation_forest.pkl
├── mplad_scaler.pkl
├── mplad_category_medians.pkl
├── mplad_state_medians.pkl
└── mplad_state_delay_medians.pkl

How to Run Locally

1. Clone the repository

git clone https://github.com/YOUR-USERNAME/MPLAD-AI.git
cd MPLAD-AI

2. Install dependencies

pip install -r requirements.txt

3. Run the application

streamlit run app.py

Application Workflow

MPLAD Works
     ↓
Data Preparation
     ↓
Feature Engineering
     ↓
AI Anomaly Detection
     ↓
Priority Risk Assessment
     ↓
Explainable Reasons
     ↓
Officer Review
     ↓
Human Decision

Important Design Principle

MPLAD-AI separates AI Pattern Status from Priority Risk.

Therefore, a work can have:

High Priority + No Anomaly

or:

High Priority + Anomaly Detected

This is intentional because the two signals measure different aspects of the work.

Dataset

The current prototype uses an official MPLADS/eSAKSHI dataset snapshot. It is a point-in-time dataset and is not a live real-time integration with the MPLADS portal.

Limitations

The current system uses a point-in-time dataset snapshot.

Isolation Forest is unsupervised and does not use confirmed fraud labels.

Risk scores are relative to the analyzed dataset.

Risk levels are not official MPLADS risk categories.

Explanation thresholds are analytical indicators, not official government fraud rules.

AI outputs require human review and verification of supporting records.

The current text feature uses description word count.

Future Improvements

Live MPLADS/eSAKSHI data integration

Similar and duplicate work detection using text embeddings

Advanced NLP analysis

Data-quality anomaly detection

Historical trend monitoring

Officer review history

Model monitoring and recalibration

Authentication and role-based access

Production monitoring

Disclaimer

MPLAD-AI is an AI-assisted analytical prototype for review prioritization. Its outputs are indicators for further examination and should not be treated as proof of fraud or misconduct. Final decisions should be made by authorized officials using official records and applicable procedures.

License

This project is developed as an academic/hackathon prototype for Smart India Hackathon (SIH) 2026.