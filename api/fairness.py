"""
Fairness API Router for CustomerAI Platform

This module provides API endpoints for fairness analysis and bias mitigation.
"""

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import json
import io
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from src.utils.date_provider import DateProvider

from src.fairness.bias_detector import BiasDetector
from src.fairness.mitigation import FairnessMitigation
from src.config.fairness_config import get_fairness_config
from src.utils.date_provider import DateProvider

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/fairness",
    tags=["fairness"],
    responses={404: {"description": "Not found"}}
)

# Load configuration
fairness_config = get_fairness_config()

# Initialize components with configuration
bias_detector = BiasDetector()
fairness_mitigation = FairnessMitigation()

# API rate limiting settings from configuration
RATE_LIMIT = fairness_config.get('api', 'rate_limit', default=100)  # requests per minute
MAX_PAYLOAD_SIZE_MB = fairness_config.get('api', 'max_payload_size_mb', default=10)
CACHE_TTL_SECONDS = fairness_config.get('api', 'cache_ttl_seconds', default=300)

# Helper function for API responses
def create_response(data: Any = None, error: bool = False, status_code: int = 200, 
                    message: str = None, details: Dict = None):
    """Create a standardized API response."""
    response = {
        "status": "error" if error else "success",
        "code": status_code,
        "timestamp": DateProvider.get_instance().iso_format()
    }
    
    if message:
        response["message"] = message
        
    if data is not None:
        response["data"] = data
        
    if details:
        response["details"] = details
        
    return response

@router.post("/analyze")
async def analyze_fairness(request_data: Dict):
    """
    Analyze fairness in a dataset.
    
    Request body should contain:
    - data: List of records to analyze
    - attributes: List of protected attribute column names
    - outcome_columns: List of outcome column names to analyze
    - positive_outcome_value: (Optional) Value considered as positive outcome
    """
    try:
        data = request_data.get("data", [])
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Data is required"
            )
        
        attributes = request_data.get("attributes", [])
        outcome_columns = request_data.get("outcome_columns", [])
        
        if not attributes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Attributes are required"
            )
            
        if not outcome_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Outcome columns are required"
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Get optional parameters
        positive_outcome_value = request_data.get("positive_outcome_value", None)
        threshold = request_data.get("threshold", 0.8)
        
        # Analyze fairness
        results = bias_detector.detect_outcome_bias(
            df, 
            attributes=attributes,
            outcome_columns=outcome_columns,
            positive_outcome_value=positive_outcome_value
        )
        
        # Generate report
        fairness_report = bias_detector.generate_fairness_report(results, threshold=threshold)
        
        # Generate mitigation recommendations
        mitigation_recommendations = fairness_mitigation.get_mitigation_recommendations(fairness_report)
        fairness_report["mitigation_recommendations"] = mitigation_recommendations
        
        return create_response(fairness_report)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Fairness analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fairness analysis failed"
        )

@router.post("/upload-analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    attributes: str = None,
    outcome_columns: str = None,
    threshold: float = 0.8
):
    """
    Upload a CSV file and analyze fairness.
    
    Form data should include:
    - file: CSV file to analyze
    - attributes: Comma-separated list of protected attribute column names
    - outcome_columns: Comma-separated list of outcome column names
    - threshold: (Optional) Fairness threshold value
    """
    try:
        # Validate inputs
        if not attributes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Protected attributes are required"
            )
            
        if not outcome_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Outcome columns are required"
            )
        
        # Parse attributes and outcome columns
        attr_list = [attr.strip() for attr in attributes.split(",")]
        outcome_list = [outcome.strip() for outcome in outcome_columns.split(",")]
        
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Validate columns
        missing_attrs = [attr for attr in attr_list if attr not in df.columns]
        if missing_attrs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Attributes not found in data: {', '.join(missing_attrs)}"
            )
            
        missing_outcomes = [outcome for outcome in outcome_list if outcome not in df.columns]
        if missing_outcomes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Outcome columns not found in data: {', '.join(missing_outcomes)}"
            )
        
        # Analyze fairness
        results = bias_detector.detect_outcome_bias(
            df, 
            attributes=attr_list,
            outcome_columns=outcome_list
        )
        
        # Generate report
        fairness_report = bias_detector.generate_fairness_report(results, threshold=threshold)
        
        # Generate mitigation recommendations
        mitigation_recommendations = fairness_mitigation.get_mitigation_recommendations(fairness_report)
        fairness_report["mitigation_recommendations"] = mitigation_recommendations
        
        # Add dataset summary
        fairness_report["dataset_summary"] = {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist()
        }
        
        return create_response(fairness_report)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"File upload and fairness analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload and fairness analysis failed"
        )

@router.post("/mitigate")
async def mitigate_bias(request_data: Dict):
    """
    Apply bias mitigation strategies to a dataset.
    
    Request body should contain:
    - data: List of records to mitigate bias in
    - attribute: Protected attribute column name
    - outcome_column: Outcome column name
    - strategy: Mitigation strategy to apply
    - strategy_params: (Optional) Additional parameters for the strategy
    """
    try:
        data = request_data.get("data", [])
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Data is required"
            )
        
        attribute = request_data.get("attribute")
        outcome_column = request_data.get("outcome_column")
        strategy = request_data.get("strategy")
        
        if not attribute:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Protected attribute is required"
            )
            
        if not outcome_column:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Outcome column is required"
            )
            
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mitigation strategy is required"
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Get strategy parameters
        strategy_params = request_data.get("strategy_params", {})
        
        # Apply mitigation strategy
        if strategy == "reweigh_samples":
            # Apply sample reweighting
            _, weights = fairness_mitigation.reweigh_samples(
                df,
                protected_attribute=attribute,
                outcome_column=outcome_column,
                **strategy_params
            )
            
            result = {
                "strategy": "reweigh_samples",
                "weights": weights.tolist(),
                "original_data": df.to_dict(orient="records")
            }
            
        elif strategy == "balanced_sampling":
            # Apply balanced sampling
            balanced_df = fairness_mitigation.balanced_sampling(
                df,
                protected_attribute=attribute,
                outcome_column=outcome_column,
                **strategy_params
            )
            
            result = {
                "strategy": "balanced_sampling",
                "original_size": len(df),
                "balanced_size": len(balanced_df),
                "balanced_data": balanced_df.to_dict(orient="records")
            }
            
        elif strategy == "disparate_impact_remover":
            # Get features to transform
            features = request_data.get("features", [])
            if not features:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Features to transform are required"
                )
            
            # Apply disparate impact remover
            transformed_df = fairness_mitigation.disparate_impact_remover(
                df,
                protected_attribute=attribute,
                features=features,
                **strategy_params
            )
            
            result = {
                "strategy": "disparate_impact_remover",
                "transformed_data": transformed_df.to_dict(orient="records")
            }
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported mitigation strategy: {strategy}"
            )
        
        return create_response(result)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Bias mitigation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bias mitigation failed"
        )

@router.post("/mitigate-predictions")
async def mitigate_predictions(request_data: Dict):
    """
    Apply bias mitigation strategies to model predictions.
    
    Request body should contain:
    - predictions: List of prediction probabilities or scores
    - true_values: List of true outcome values
    - protected_attributes: List of protected attribute values
    - strategy: Post-processing strategy to apply
    - strategy_params: (Optional) Additional parameters for the strategy
    """
    try:
        predictions = request_data.get("predictions", [])
        true_values = request_data.get("true_values", [])
        protected_attributes = request_data.get("protected_attributes", [])
        strategy = request_data.get("strategy")
        
        if not predictions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Predictions are required"
            )
            
        if not true_values:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="True values are required"
            )
            
        if not protected_attributes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Protected attributes are required"
            )
            
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mitigation strategy is required"
            )
            
        if len(predictions) != len(true_values) or len(predictions) != len(protected_attributes):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Predictions, true values, and protected attributes must have the same length"
            )
        
        # Convert to numpy arrays
        y_pred = np.array(predictions)
        y_true = np.array(true_values)
        protected_attrs = np.array(protected_attributes)
        
        # Get strategy parameters
        strategy_params = request_data.get("strategy_params", {})
        
        # Apply mitigation strategy
        if strategy == "equalized_odds_postprocessing":
            # Apply equalized odds post-processing
            threshold = strategy_params.get("threshold", 0.5)
            adjusted_predictions = fairness_mitigation.equalized_odds_postprocessing(
                y_pred,
                y_true,
                protected_attrs,
                threshold=threshold
            )
            
            result = {
                "strategy": "equalized_odds_postprocessing",
                "adjusted_predictions": adjusted_predictions.tolist()
            }
            
        elif strategy == "calibrated_equalized_odds":
            # Apply calibrated equalized odds
            cost_constraint = strategy_params.get("cost_constraint", "weighted")
            adjusted_proba, calibration_params = fairness_mitigation.calibrated_equalized_odds(
                y_pred,
                y_true,
                protected_attrs,
                cost_constraint=cost_constraint
            )
            
            # Convert numpy arrays in calibration_params to lists for JSON serialization
            serializable_params = {}
            for key, value in calibration_params.items():
                serializable_params[str(key)] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            
            result = {
                "strategy": "calibrated_equalized_odds",
                "adjusted_probabilities": adjusted_proba.tolist(),
                "calibration_parameters": serializable_params
            }
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported mitigation strategy: {strategy}"
            )
        
        return create_response(result)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction mitigation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction mitigation failed"
        )

@router.get("/strategies")
async def get_mitigation_strategies():
    """
    Get available bias mitigation strategies.
    """
    try:
        strategies = {
            "pre_processing": [
                {
                    "id": "reweigh_samples",
                    "name": "Sample Reweighting",
                    "description": "Assigns weights to training examples to ensure fairness across protected attribute groups",
                    "parameters": [
                        {"name": "protected_attribute", "type": "string", "required": True, "description": "Protected attribute column name"},
                        {"name": "outcome_column", "type": "string", "required": True, "description": "Outcome column name"},
                        {"name": "positive_outcome_value", "type": "any", "required": False, "description": "Value considered as positive outcome"}
                    ]
                },
                {
                    "id": "balanced_sampling",
                    "name": "Balanced Sampling",
                    "description": "Creates a balanced dataset through resampling to mitigate bias",
                    "parameters": [
                        {"name": "protected_attribute", "type": "string", "required": True, "description": "Protected attribute column name"},
                        {"name": "outcome_column", "type": "string", "required": True, "description": "Outcome column name"},
                        {"name": "positive_outcome_value", "type": "any", "required": False, "description": "Value considered as positive outcome"},
                        {"name": "random_state", "type": "integer", "required": False, "description": "Random seed for reproducibility"}
                    ]
                },
                {
                    "id": "disparate_impact_remover",
                    "name": "Disparate Impact Remover",
                    "description": "Transforms features to reduce correlation with protected attribute",
                    "parameters": [
                        {"name": "protected_attribute", "type": "string", "required": True, "description": "Protected attribute column name"},
                        {"name": "features", "type": "array", "required": True, "description": "List of feature columns to transform"},
                        {"name": "repair_level", "type": "number", "required": False, "description": "Level of repair (0.0 to 1.0, where 1.0 is full repair)"}
                    ]
                }
            ],
            "post_processing": [
                {
                    "id": "equalized_odds_postprocessing",
                    "name": "Equalized Odds Post-processing",
                    "description": "Adjusts prediction thresholds to achieve similar error rates across protected attribute groups",
                    "parameters": [
                        {"name": "threshold", "type": "number", "required": False, "description": "Initial classification threshold"}
                    ]
                },
                {
                    "id": "calibrated_equalized_odds",
                    "name": "Calibrated Equalized Odds",
                    "description": "Applies calibrated equalized odds to adjust prediction probabilities",
                    "parameters": [
                        {"name": "cost_constraint", "type": "string", "required": False, "description": "Type of cost constraint ('fpr', 'tpr', or 'weighted')"}
                    ]
                }
            ]
        }
        
        return create_response(strategies)
    except Exception as e:
        logger.error(f"Error getting mitigation strategies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get mitigation strategies"
        )
