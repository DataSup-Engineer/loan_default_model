# Loan Default Prediction System

A machine learning application that predicts whether a customer will repay their loan using XGBoost classification.

## Overview

This system uses XGBoost gradient boosting to predict loan default risk based on customer and loan characteristics. It includes:
- Jupyter notebook for data exploration, preprocessing, model training, and evaluation
- FastAPI web service for serving predictions via REST API
- Comprehensive data preprocessing pipeline
- Model persistence for deployment

## Features

- **Exploratory Data Analysis**: Comprehensive analysis of loan data with visualizations
- **Data Preprocessing**: Handles missing values, encodes categorical variables, and prepares features
- **XGBoost Model**: Binary classification model optimized for loan default prediction
- **Model Evaluation**: Accuracy, confusion matrix, precision, recall, F1-score, and ROC-AUC metrics
- **REST API**: FastAPI service with prediction and health check endpoints
- **Interactive Documentation**: Automatic Swagger UI for API testing

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure the dataset `Loan_Default_data.csv` is in the project directory

## Usage

### Training the Model

1. Open the Jupyter notebook:
```bash
jupyter notebook loan_default_model.ipynb
```

2. Run all cells sequentially to:
   - Load and explore the data
   - Perform exploratory data analysis
   - Preprocess the data
   - Train the XGBoost model
   - Evaluate model performance
   - Save the trained model and preprocessing artifacts

### Running the API Service

1. Start the FastAPI server:
```bash
uvicorn api:app --reload
```

2. Access the interactive API documentation at: `http://localhost:8000/docs`

3. Test the health check endpoint:
```bash
curl http://localhost:8000/health
```

4. Make a prediction request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 5000,
    "funded_amnt": 5000,
    "term": "36 months",
    "int_rate": 10.5,
    "installment": 161.0,
    "emp_length": "2 years",
    "home_ownership": "RENT",
    "annual_inc": 45000,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "dti": 15.5,
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "open_acc": 5,
    "pub_rec": 0,
    "revol_bal": 5000,
    "revol_util": "50%",
    "total_acc": 10
  }'
```

## Project Structure

```
loan-default-prediction/
├── loan_default_model.ipynb      # Main notebook with analysis and training
├── api.py                         # FastAPI service
├── model.pkl                      # Saved XGBoost model (generated)
├── preprocessor_info.pkl          # Preprocessing artifacts (generated)
├── Loan_Default_data.csv         # Dataset
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## API Endpoints

### Health Check
- **Endpoint**: `GET /health`
- **Response**: Server status and model readiness

### Prediction
- **Endpoint**: `POST /predict`
- **Request Body**: JSON with loan application features
- **Response**: Prediction result with probability and risk level

## Model Performance

The XGBoost model is evaluated using:
- Accuracy score
- Confusion matrix
- Precision, recall, and F1-score
- ROC-AUC score

Detailed performance metrics are available in the Jupyter notebook.

## Technology Stack

- **Python**: Core programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **scikit-learn**: Preprocessing and evaluation metrics
- **XGBoost**: Gradient boosting classifier
- **FastAPI**: Web framework for API
- **Pydantic**: Data validation
- **matplotlib/seaborn**: Data visualization
- **Jupyter**: Interactive notebook environment

## License

This project is for educational purposes.
