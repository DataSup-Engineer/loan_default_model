"""
Loan Default Prediction API

A FastAPI service that provides endpoints for predicting loan default risk
using a trained XGBoost model.
"""

# Import required libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# Initialize FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default risk using XGBoost",
    version="1.0.0"
)

# Global variables for model and preprocessor
model = None
preprocessor_info = None

print("FastAPI application initialized successfully!")


# Load model and preprocessing artifacts on startup
@app.on_event("startup")
async def load_model_and_preprocessor():
    """Load the trained model and preprocessing artifacts when the API starts."""
    global model, preprocessor_info
    
    try:
        # Load trained model
        model_path = 'model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Model loaded successfully from {model_path}")
        else:
            print(f"⚠ Warning: Model file not found at {model_path}")
            
        # Load preprocessing info
        preprocessor_path = 'preprocessor_info.pkl'
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor_info = pickle.load(f)
            print(f"✓ Preprocessor info loaded successfully from {preprocessor_path}")
        else:
            print(f"⚠ Warning: Preprocessor file not found at {preprocessor_path}")
            
    except Exception as e:
        print(f"✗ Error loading model or preprocessor: {str(e)}")
        raise


# Define Pydantic models for request and response
class LoanApplication(BaseModel):
    """Request model for loan application data."""
    loan_amnt: float = Field(..., description="Loan amount requested", example=5000.0)
    funded_amnt: float = Field(..., description="Amount funded", example=5000.0)
    term: str = Field(..., description="Loan term", example="36 months")
    int_rate: float = Field(..., description="Interest rate", example=10.5)
    installment: float = Field(..., description="Monthly installment", example=161.0)
    emp_length: str = Field(..., description="Employment length", example="2 years")
    home_ownership: str = Field(..., description="Home ownership status", example="RENT")
    annual_inc: float = Field(..., description="Annual income", example=45000.0)
    verification_status: str = Field(..., description="Income verification status", example="Verified")
    purpose: str = Field(..., description="Loan purpose", example="debt_consolidation")
    dti: float = Field(..., description="Debt-to-income ratio", example=15.5)
    delinq_2yrs: int = Field(..., description="Number of delinquencies in past 2 years", example=0)
    inq_last_6mths: int = Field(..., description="Credit inquiries in last 6 months", example=1)
    open_acc: int = Field(..., description="Number of open credit accounts", example=5)
    pub_rec: int = Field(..., description="Number of public records", example=0)
    revol_bal: float = Field(..., description="Revolving balance", example=5000.0)
    revol_util: str = Field(..., description="Revolving utilization", example="50%")
    total_acc: int = Field(..., description="Total credit accounts", example=10)
    
    class Config:
        schema_extra = {
            "example": {
                "loan_amnt": 5000.0,
                "funded_amnt": 5000.0,
                "term": "36 months",
                "int_rate": 10.5,
                "installment": 161.0,
                "emp_length": "2 years",
                "home_ownership": "RENT",
                "annual_inc": 45000.0,
                "verification_status": "Verified",
                "purpose": "debt_consolidation",
                "dti": 15.5,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1,
                "open_acc": 5,
                "pub_rec": 0,
                "revol_bal": 5000.0,
                "revol_util": "50%",
                "total_acc": 10
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    prediction: int = Field(..., description="Binary prediction (0=Repaid, 1=Defaulted)")
    probability: float = Field(..., description="Probability of default")
    risk_level: str = Field(..., description="Risk level (low, medium, high)")
    message: str = Field(..., description="Human-readable message")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.23,
                "risk_level": "low",
                "message": "Loan is likely to be repaid"
            }
        }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API status and model readiness.
    
    Returns:
        dict: Status information including model loaded status
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor_info is not None
    }


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_loan_default(application: LoanApplication):
    """
    Predict loan default risk for a given loan application.
    
    Args:
        application: LoanApplication object with loan details
        
    Returns:
        PredictionResponse: Prediction results with probability and risk level
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if preprocessor_info is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([application.dict()])
        
        # Apply preprocessing transformations
        # 1. Convert percentage string to numeric
        if 'revol_util' in input_data.columns:
            input_data['revol_util'] = input_data['revol_util'].str.replace('%', '').astype(float) / 100
        
        # 2. Apply label encoding for ordinal features
        label_encoders = preprocessor_info.get('label_encoders', {})
        for col, encoder in label_encoders.items():
            if col in input_data.columns:
                try:
                    input_data[col] = encoder.transform(input_data[col].astype(str))
                except ValueError:
                    # Handle unseen categories by using the most frequent class
                    input_data[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # 3. Apply one-hot encoding for nominal features
        # Get categorical columns that need one-hot encoding
        categorical_cols = input_data.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
        
        # 4. Align columns with training data
        feature_columns = preprocessor_info.get('feature_columns', [])
        
        # Add missing columns with 0
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Remove extra columns and reorder to match training
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0, 1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "low"
            message = "Loan is likely to be repaid"
        elif probability < 0.7:
            risk_level = "medium"
            message = "Loan has moderate default risk"
        else:
            risk_level = "high"
            message = "Loan has high default risk"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        dict: API information and available endpoints
    """
    return {
        "message": "Loan Default Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


# Error handling is already implemented in the endpoints above:
# - HTTPException with status 503 when model is not loaded
# - HTTPException with status 400 for invalid input (handled by Pydantic)
# - HTTPException with status 500 for prediction errors
# - Pydantic automatically validates input data and returns 422 for validation errors

if __name__ == "__main__":
    import uvicorn
    print("Starting Loan Default Prediction API...")
    print("API Documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
