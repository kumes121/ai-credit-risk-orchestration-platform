import os

# =====================================================================
# PLATFORM MODAL EXECUTION SWITCH
# Set to 'LOCAL' for free offline testing, or 'AWS' for cloud deployments
# =====================================================================
ENVIRONMENT_MODE = os.getenv("ENVIRONMENT_MODE", "LOCAL").upper()

# =====================================================================
# CORE AI MODEL CONFIGURATIONS
# =====================================================================
# Offline model fallback via Ollama vs Cloud provider endpoints
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5-coder:3b")
# Pointing to the explicit canonical model ID for Claude 3.5 Sonnet on Bedrock
AMAZON_BEDROCK_MODEL_ID = os.getenv("AMAZON_BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")


# =====================================================================
# INFRASTRUCTURE & STORAGE ENDPOINTS (AWS)
# =====================================================================
# Region fallback explicitly set to your target architecture region 'us-east-2'
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# Production S3 bucket name mapping for model artifact downloads
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "ai-credit-risk-model-artifacts")

XGBOOST_MODEL_PATH = os.getenv("XGBOOST_MODEL_PATH", "models/xgboost_champion.joblib")
AWS_AURORA_POSTGRES_URI = os.getenv("AWS_AURORA_POSTGRES_URI", "postgresql://db-user:pass@localhost:5432/metadata")

# =====================================================================
# SECURITY GUARDRAIL TERMS
# =====================================================================
PROHIBITED_BIAS_TERMS = [
    "age-based", 
    "gender-profile", 
    "demographic zip", 
    "protected class"
]

# =====================================================================
# OBSERVABILITY & TELEMETRY CONFIGURATIONS (LANGFUSE)
# =====================================================================
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-mock-local-key-12345")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-mock-local-key-12345")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

# Toggle flag to cleanly disable remote tracking during standalone test sweeps
ENABLE_TELEMETRY = os.getenv("ENABLE_TELEMETRY", "TRUE").upper() == "TRUE"

# Extend the existing settings mapping dictionary
def get_active_model_settings() -> dict:
    return {
        "mode": ENVIRONMENT_MODE,
        "model_path": XGBOOST_MODEL_PATH,
        "compliance_filters": PROHIBITED_BIAS_TERMS,
        "engine": OLLAMA_MODEL_NAME if ENVIRONMENT_MODE == "LOCAL" else AMAZON_BEDROCK_MODEL_ID,
        "telemetry_active": ENABLE_TELEMETRY,
        "aws_region": AWS_REGION,              # Exported for easy metadata visibility
        "s3_bucket": AWS_S3_BUCKET_NAME         #Exported for download tracking checks
    }