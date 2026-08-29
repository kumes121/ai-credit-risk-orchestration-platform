from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class LoanApplicationRequest(BaseModel):
    """
    Contract verification schema enforcing strict data types for incoming credit profiles.
    Maps directly to the feature attributes utilized by the core XGBoost model.
    """
    borrower_id: str = Field(
        ..., 
        description="Unique system alpha-numeric identifier for the credit applicant.",
        examples=["HL-9821"]
    )
    income: float = Field(
        ..., 
        description="Annual gross verified income of the primary borrower in USD.", 
        gt=0,
        examples=[125000.00]
    )
    debt_to_income: float = Field(
        ..., 
        description="Calculated Debt-to-Income (DTI) ratio percentage inclusive of recurring liabilities.", 
        ge=0.0, 
        le=100.0,
        examples=[41.5]
    )
    property_value: float = Field(
        ..., 
        description="Appraised asset market valuation for the underlying real estate property.", 
        gt=0,
        examples=[420000.00]
    )
    loan_amount: float = Field(
        ..., 
        description="Total primary principal credit balance requested for mortgage issuance.", 
        gt=0,
        examples=[315000.00]
    )

class UnderwritingResponse(BaseModel):
    """
    Structured platform response returning predictive evaluations alongside 
    agentic synthesis and compliance validation states.
    """
    borrower_id: str
    default_probability: float = Field(..., description="Raw predictive model risk probability score.")
    underwriting_decision: str = Field(..., description="Automated risk tier evaluation (e.g., APPROVED, REJECTED).")
    compliance_passed: bool = Field(..., description="Flag indicating if the text passed fair-lending guardrails checks.")
    automated_credit_memo: str = Field(..., description="Plain-English explanation memo mapping model metrics to policy vectors (e.g., Credit Approval Summary or Adverse Action Text).")
    execution_trace_id: Optional[str] = Field(None, description="LangSmith trace tracking reference token.")

class ComplianceLinterPayload(BaseModel):
    """Contract payload passing generated text artifacts into deterministic regex/PII masking blocks."""
    text_content: str
    strict_mode: bool = True
