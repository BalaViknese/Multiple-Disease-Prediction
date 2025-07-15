# Multiple-Disease-Prediction

This is a Streamlit web application that predicts the risk of **multiple diseases**:
- Liver Disease
- Chronic Kidney Disease (CKD)
- Parkinson's Disease

The app uses **machine learning models** built with Python to assist in early detection and improve healthcare decision-making.

---

## **Project Overview**

This project is designed to:
- Predict the risk of multiple diseases based on patient data
- Provide quick and user-friendly predictions
- Help reduce diagnostic time and cost using ML models

The system includes:
- **Data preprocessing and cleaning**
- **Model training and evaluation** (Random Forest, XGBoost, Logistic Regression)
- **Interactive frontend** using Streamlit
- **Modular, scalable code** suitable for adding more diseases

---

## **Technologies Used**
- Python
- Streamlit
- scikit-learn
- XGBoost
- NumPy & Pandas
- Joblib

---

## **Features**
- Predicts:
  - Liver Disease from blood test data and demographics
  - Kidney Disease from lab test data and medical indicators
  - Parkinson's Disease from acoustic voice measurements
- Interactive and easy-to-use web UI
- Displays risk scores and prediction results
- Modular structure for future extension

---

## **Project Structure**
├── multi_disease_app.py # Streamlit application
├── liver_xgb_model.pkl # Saved liver disease model
├── kidney_rf_model.pkl # Saved kidney disease model
├── parkinsons_rf_model.pkl # Saved Parkinson's model
├── scaler.pkl # Liver scaler
├── kidney_scaler.pkl # Kidney scaler
├── parkinsons_scaler.pkl # Parkinson's scaler
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## **How to Run the App Locally**

1️ Clone the repository:

git clone https://github.com/yourusername/multiple-disease-prediction.git
cd multiple-disease-prediction
2️ Install dependencies:

pip install -r requirements.txt
3️ Run the Streamlit app:

streamlit run multi_disease_app.py

## **Prediction Models**
Disease	Algorithm	Notes
Liver Disease	XGBoost	Trained on Indian liver patient dataset
Kidney Disease	Random Forest	Trained on CKD dataset with medical tests
Parkinson's	Random Forest	Trained on voice measurement data

All models were trained with proper scaling and preprocessing, and saved using joblib.

## **Future Work**
Add more diseases (e.g., heart disease, diabetes)

Deploy on Streamlit Cloud or AWS

Add visualization dashboards

Improve interpretability using SHAP / LIME

## **About**
This project was built as part of a machine learning and data science portfolio to demonstrate:

End-to-end data science workflow

ML model building and evaluation

Deploying ML models to production using Streamlit

## **Author**
Bala Viknese
