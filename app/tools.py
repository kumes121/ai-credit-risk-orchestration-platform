import os
import io
import joblib
import shlex
import shap
import boto3
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from botocore.exceptions import ClientError
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# Import the centralized runtime configuration from config
from app.config import (
    XGBOOST_MODEL_PATH,
    ENVIRONMENT_MODE,
    AWS_REGION,
    AWS_S3_BUCKET_NAME,
)

# Global cache for the machine learning model to prevent reloading on every execution loop
MODEL_CACHE = None

# Using StreamHandler paired with Formatter for clean console logs
logger = logging.getLogger("credit_risk_inference")
logger.setLevel(logging.INFO)

# Create console handler and set layout rules
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)

# Append handler directly to your active logger instance
if not logger.handlers:
    logger.addHandler(console_handler)

def _load_model():
    """Lazy-loads or streams the serialized XGBoost pipeline from local disk or AWS S3."""
    global MODEL_CACHE
    if MODEL_CACHE is None:
        if ENVIRONMENT_MODE == "AWS":
            try:
                logger.info(f"[AWS S3 INITIALIZATION] Streaming model artifact from bucket: {AWS_S3_BUCKET_NAME} in {AWS_REGION}...")
                s3_client = boto3.client("s3", region_name=AWS_REGION)
                
                buffer = io.BytesIO()
                s3_client.download_fileobj(AWS_S3_BUCKET_NAME, XGBOOST_MODEL_PATH, buffer)
                buffer.seek(0)
                
                MODEL_CACHE = joblib.load(buffer)
                logger.info("[AWS S3 SUCCESS] XGBoost champion model loaded securely directly into RAM stream cache.")
                return MODEL_CACHE
            except ClientError as e:
                logger.error(f"[S3 ERROR] download permission or bucket pathway error: {str(e)}", exc_info=True)
                MODEL_CACHE = "MOCK_MODE"
            except Exception as e:
                logger.error(f"[AWS INITIALIZATION ERROR] AWS connection loop failed: {str(e)}", exc_info=True)
                MODEL_CACHE = "MOCK_MODE"
        else:
            try:
                if os.path.exists(XGBOOST_MODEL_PATH):
                    MODEL_CACHE = joblib.load(XGBOOST_MODEL_PATH)
                    logger.info(f"[TOOL INITIALIZATION] Successfully loaded model pipeline from local paths: {XGBOOST_MODEL_PATH}")
                else:
                    logger.warning(f"[WARN] Local path {XGBOOST_MODEL_PATH} not found. Triggering zero-cost simulated MOCK_MODE.")
                    MODEL_CACHE = "MOCK_MODE"
            except Exception as e:
                logger.error(f"[LOCAL BOOT ERROR] Failed to read local model file from disk: {str(e)}", exc_info=True)
                MODEL_CACHE = "MOCK_MODE"
                
    return MODEL_CACHE

@tool
def predict_loan_default_risk(application_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes real-time machine learning inference against a borrower profile.
    Accepts raw application features, engineered features, and routes them through data cleaning,
    and applies the champion 0.321 risk threshold.
    """

    model_pipeline = _load_model()

     # 1. Extract raw numerical features from ingestion
    income = float(application_data.get("income", 0))
    debt_to_income = float(application_data.get("debt_to_income", 0))
    property_value = float(application_data.get("property_value", 0))
    loan_amount = float(application_data.get("loan_amount", 0))
    
    # 2. PHASE 1 MATCHING FEATURE ENGINEERING
    # Apply your exact capstone logic: log transforms and log-scaled ratio math
    log_income = np.log(income) if income > 0 else 0
    log_loan_amount = np.log(loan_amount) if loan_amount > 0 else 0
    
    # Calculate loan_to_income using your exact project formula
    loan_to_income = log_loan_amount / (log_income + 1e-6)
    
    # 3. Structure the DataFrame to match the exact input feature definitions of your Phase 1 training matrix
    raw_df = pd.DataFrame([{
        "income": income,
        "debt_to_income": debt_to_income,
        "property_value": property_value,
        "loan_amount": loan_amount,
        "loan_to_income": loan_to_income  # ✅ Critical engineered feature appended
    }])
    
    # 4. Handle Inference Evaluation
    if model_pipeline == "MOCK_MODE":
        # Safe offline mathematical fallback simulating your model parameters
        prob = 0.68 if debt_to_income > 43.0 or loan_to_income > 4.5 else 0.18
    else:
        try:
            # Production Execution: Pass through the Scikit-Learn ColumnTransformer pipeline
            prob_array = model_pipeline.predict_proba(raw_df)
            prob = float(prob_array[0][1])  # Extract risk probability index for Default=1
            logger.info(f"[INFERENCE SUCCESS] Calculated dynamic default risk probability: {prob:.4f}")
        except Exception as e:
            logger.error(f"[INFERENCE FAILURE] Critical pipeline structural feature mismatch matrix crash: {str(e)}", exc_info=True)
            prob = 0.50  # Operational neutral fallback state

    # 5. Apply your UC Berkeley capstone optimized threshold (0.321)
    threshold = 0.321
    decision = "REJECTED" if prob >= threshold else "APPROVED"
    
    return {
        "raw_risk_probability": round(prob, 4),
        "optimized_decision_threshold": threshold,
        "underwriting_action": decision
    }

@tool
def query_credit_policy_kb(query_keyword: str) -> str:
    """
    Queries the internal corporate lending policy knowledge base.
    Use this to pull specific underwriting thresholds or compliance guidelines for credit lines.
    """
    # Local vector simulation database mapping enterprise underwriting criteria
    policy_kb = {
        "debt_to_income": "Lending Policy Guideline Section 4.12: Conventional mortgage applications showing a secondary Debt-to-Income (DTI) ratio exceeding 43.0% are restricted from automated tier-1 approval pathways unless offset by substantial cash reserves.",
        "loan_to_value": "Compliance Regulation Section 2.5: Loan-to-Value (LTV) ratios exceeding 80.0% require mandatory Private Mortgage Insurance (PMI) masking indicators and must pass secondary asset evaluation checks.",
        "default": "Risk Management Framework: High-risk classifications require explicit Adverse Action statements documenting non-linear risk drivers via localized SHAP explainability matrices."
    }
    
    # Simple semantic fallback routing
    normalized_query = query_keyword.lower()
    for key, value in policy_kb.items():
        if key in normalized_query:
            return value
            
    return "Lending Policy Note: Standard underwriting guidelines apply. Ensure Debt-to-Income limits match current interest tier regulations."

async def get_mcp_tools() -> List[BaseTool]:
    """
    Spins up a persistent connection to the configured MCP server
    and discovers tools to inject directly into the LangGraph state machine.
    """
    server_cmd = os.getenv("MCP_SERVER_COMMAND")
    server_args_raw = os.getenv("MCP_SERVER_ARGS", "")
    
    # Return empty if MCP environment controls are not active
    if not server_cmd:
        print("[MCP INITIALIZATION] No MCP server command found. Skipping runtime discovery.")
        return []
        
    # Safely parse arguments out into a structured list for subprocess execution
    server_args = shlex.split(server_args_raw)
    
    try:
        print(f"[MCP CONNECTING] Initializing protocol session with: {server_cmd} {server_args}")
        
        # ✅ Initialize the official multi-server standard client
        # For a filesystem server, it establishes a reliable background stdio pipe channel
        mcp_client = MultiServerMCPClient(
            {
                "enterprise_compliance_resource": {
                    "transport": "stdio",
                    "command": server_cmd,
                    "args": server_args,
                }
            }
        )
        
        # Discover and format tools into standard LangChain BaseTools
        discovered_tools = await mcp_client.get_tools()
        print(f"[MCP SUCCESS] Discovered {len(discovered_tools)} external tools via protocol standard.")
        return discovered_tools
        
    except Exception as e:
        print(f"[MCP ERROR] Protocol initialization failed: {str(e)}. Falling back to local tools.")
        return []


# Global cache for the SHAP Explainer object to optimize runtime compute footprints
SHAP_EXPLAINER_CACHE = None


# Update this function inside your app/tools.py file:
def _get_shap_explainer(model):
    """Lazy-loads and caches the SHAP TreeExplainer instance."""
    global SHAP_EXPLAINER_CACHE
    if (
        SHAP_EXPLAINER_CACHE is None
        and model != "MOCK_MODE"
        and model != "BROKEN_MODEL_ARTIFACT_ERROR"
    ):
        try:
            # Peel back the Pipeline/ColumnTransformer layers to isolate the raw estimator
            if hasattr(model, "named_steps") and "classifier" in model.named_steps:
                native_booster = model.named_steps["classifier"]
            else:
                native_booster = model

            SHAP_EXPLAINER_CACHE = shap.TreeExplainer(native_booster)
            logger.info(
                "[XAI SUCCESS] SHAP TreeExplainer initialized dynamically on production booster weights."
            )
        except Exception as e:
            logger.error(
                f"[XAI INITIALIZATION WARN] Failed to initialize SHAP TreeExplainer wrapper: {str(e)}",
                exc_info=True,
            )
            SHAP_EXPLAINER_CACHE = None
    return SHAP_EXPLAINER_CACHE


@tool
def generate_shap_risk_explanations(application_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes mathematical SHAP feature contribution values for an underwriting profile.
    Returns impact labels indicating which metrics pushed the risk profile upward or downward.
    """
    model = _load_model()
    
    # 1. Re-apply your Phase 1 capstone matching feature mapping vector
    income = float(application_data.get("income", 0))
    debt_to_income = float(application_data.get("debt_to_income", 0))
    property_value = float(application_data.get("property_value", 0))
    loan_amount = float(application_data.get("loan_amount", 0))
    
    log_income = np.log(income) if income > 0 else 0
    log_loan_amount = np.log(loan_amount) if loan_amount > 0 else 0
    loan_to_income = log_loan_amount / (log_income + 1e-6)
    
    # ✅ FIX 1: Enforce the exact ordered column sequence string mapping of your training matrix
    feature_names = ["income", "debt_to_income", "property_value", "loan_amount", "loan_to_income"]
    raw_vector = [income, debt_to_income, property_value, loan_amount, loan_to_income]
    
    # 2. Fallback handling for offline mock testing
    explainer = _get_shap_explainer(model)
    if model == "MOCK_MODE" or explainer is None:
        # Simulate local feature importance weights matching model bounds
        dti_impact = "HIGH ADVOCATE FOR REJECTION" if debt_to_income > 43.0 else "NEUTRAL"
        return {
            "primary_risk_driver": "debt_to_income", 
            "mathematical_impact_profile": {
                "income": -0.045, 
                "debt_to_income": 0.182 if debt_to_income > 43.0 else 0.02, 
                "property_value": -0.012, 
                "loan_amount": 0.088, 
                "loan_to_income": 0.054
            }
        }

    if model == "BROKEN_MODEL_ARTIFACT_ERROR":
        error_label = "ERROR: PIPELINE OFFLINE"
        return {
            "primary_risk_driver": "ERROR: NO ARTIFACT",
            "mathematical_impact_profile": {name: error_label for name in feature_names}
        }

    try:
        # 3. Production Model Explanation Extraction
        # ✅ FIX 2: Structure the dataframe explicitly ensuring no alphabetical sorting scrambles keys
        raw_df = pd.DataFrame([raw_vector], columns=feature_names)
        raw_df = raw_df[feature_names]
        
        # Compute the attribution array block matrix
        shap_values = explainer(raw_df)
        
        # ✅ FIX 3: Safe multi-dimensional array slicing to isolate single-row attributions safely
        if hasattr(shap_values, "values") and len(shap_values.values.shape) > 1:
            val_array = shap_values.values[0]
            # Handle situations where SHAP yields an extra classification dimension [rows, features, classes]
            if len(val_array.shape) > 1:
                val_array = val_array[:, 1]
        else:
            val_array = shap_values[0] if isinstance(shap_values, list) else shap_values
        
        # Pair feature names to their respective localized attribution coefficients
        metrics_map = {name: round(float(val), 4) for name, val in zip(feature_names, val_array)}
        
        # Extract the single highest risk contributor feature vector
        primary_driver = max(metrics_map, key=lambda k: abs(metrics_map[k]))
        
        return {
            "primary_risk_driver": primary_driver,
            "mathematical_impact_profile": metrics_map
        }
    except Exception as e:
        logger.error(f"[XAI EXTRACTION FAILURE] SHAP tree matrix computation calculation collapsed: {str(e)}", exc_info=True)
        return {
            "primary_risk_driver": "debt_to_income", 
            "error": f"Attribution mapping failure: {str(e)}"
        }
