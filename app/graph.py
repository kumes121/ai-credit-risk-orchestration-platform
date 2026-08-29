import os
from typing import Dict, Any, List
from typing_extensions import TypedDict

# Core LangChain and LangGraph orchestration imports
from langchain_aws import ChatBedrockConverse 
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Centralized application environment configurations
from app.config import (
    ENVIRONMENT_MODE, 
    OLLAMA_MODEL_NAME, 
    AMAZON_BEDROCK_MODEL_ID, 
    PROHIBITED_BIAS_TERMS
)

# Machine learning wrappers, standard policy bases, and the asynchronous MCP discovery layer
from app.tools import (
    predict_loan_default_risk, 
    query_credit_policy_kb, 
    get_mcp_tools, 
    generate_shap_risk_explanations
)

# =====================================================================
# 1. DEFINE CENTRAL SHARED PLATFORM STATE
# =====================================================================
class UnderwritingState(TypedDict):
    """Tracks the continuous execution context passing across graph nodes."""
    borrower_id: str
    raw_features: Dict[str, Any]
    vector_policy_context: str
    model_prediction: Dict[str, Any]
    generated_credit_memo: str
    compliance_passed: bool
    loop_retry_count: int
    validation_error_logs: str

# =====================================================================
# 2. GRAPH NODE IMPLEMENTATIONS
# =====================================================================

def fetch_policy_context_node(state: UnderwritingState) -> Dict[str, Any]:
    """Node 1: Evaluates applicant profile keywords to query vector policy limits."""
    policy_text = query_credit_policy_kb.invoke("debt_to_income")
    return {"vector_policy_context": policy_text}

def execute_risk_model_node(state: UnderwritingState) -> Dict[str, Any]:
    """Node 2: Intercepts raw inputs, invokes XGBoost, and generates SHAP explanations."""
    # Execute core model predictive logic
    prediction_result = predict_loan_default_risk.invoke({"application_data": state["raw_features"]})
    
    # Calculate local SHAP explainability matrices for compliance audit trails
    xai_attribution = generate_shap_risk_explanations.invoke({"application_data": state["raw_features"]})
    
    # Inject XAI metrics cleanly directly into the model prediction dictionary payload block
    prediction_result["shap_explainability"] = xai_attribution
    
    return {"model_prediction": prediction_result}

def _get_active_llm_engine():
    """Dynamically provisions the targeted LLM provider based on active environment modes."""
    if ENVIRONMENT_MODE == "AWS":
        # Target your explicit default region 'us-east-2'
        aws_region = os.getenv("AWS_REGION", "us-east-2")
        print(f"[AWS BEDROCK CLIENT] Initializing cloud engine: {AMAZON_BEDROCK_MODEL_ID} in {aws_region}")
        
        # Using ChatBedrockConverse bypasses the model_kwargs dictionary instantiation error
        return ChatBedrockConverse(
            model=AMAZON_BEDROCK_MODEL_ID,
            region_name=aws_region,
            temperature=0.1
        )
    else:
        print(f"[LOCAL ENGINE] Initializing local development engine: {OLLAMA_MODEL_NAME}")
        return ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.1)

def synthesize_credit_memo_node(state: UnderwritingState) -> Dict[str, Any]:
    """Node 3: Generates plain-English text mapping model probability to internal policies."""
    # We pull the live active model engine configuration mapping out of the environment fallback bounds
    llm = _get_active_llm_engine()
    
    shap_data = state['model_prediction'].get('shap_explainability', {})

    base_prompt = (
            "You are an expert AI Credit Risk Underwriter at a Tier-1 financial institution.\n"
            "Your task is to draft a clean, compliant Credit Underwriting Memo based strictly on these parameters:\n\n"
            "--- LIVE BORROWER PROFILE ---\n"
            f"Borrower Reference ID: {state['borrower_id']}\n"
            f"Submitted Debt-to-Income (DTI) Ratio: {state['raw_features'].get('debt_to_income', 0)}%\n"
            f"XGBoost Default Probability: {state['model_prediction'].get('raw_risk_probability', 0.0)}\n"
            f"Automated System Decision: {state['model_prediction'].get('underwriting_action', 'DENIED')}\n\n"
            "--- EXPLAINABLE AI ATTRIBUTIONS (SHAP) ---\n"
            f"Primary Underwriting Risk Driver: {shap_data.get('primary_risk_driver', 'N/A')}\n"
            f"Complete Model Feature Attributions: {shap_data.get('mathematical_impact_profile', {})}\n\n"
            "--- COMPLIANCE & LENDING POLICY RULES ---\n"
            f"{state['vector_policy_context']}\n\n"
            "INSTRUCTION: Synthesize a professional memo detailing why the borrower was accepted or rejected. "
            "Do not repeat example metrics from the policy guidelines; quote only the live borrower profile attributes."
        )

    
    if state["loop_retry_count"] > 0:
        base_prompt += (
            f"--- CRITICAL CORRECTION REQUIRED ---\n"
            f"Your previous draft failed compliance routing with the following validation error:\n"
            f"'{state['validation_error_logs']}'\n"
            f"Re-write the credit memo to completely remediate this violation while preserving the original model metrics.\n\n"
        )
        
    base_prompt += "Draft Output (Include explicit Risk Analysis and Adverse Action justification if rejected):"
    
    memo_response = llm.invoke(base_prompt)
    memo_text = memo_response.content if hasattr(memo_response, 'content') else str(memo_response)
    
    return {"generated_credit_memo": memo_text}

def audit_compliance_guardrails_node(state: UnderwritingState) -> Dict[str, Any]:
    """Node 4: Evaluates text artifacts against rigid fair-lending regulatory filters."""
    memo = state["generated_credit_memo"].lower()
    error_logs = ""
    passed = True
    
    for term in PROHIBITED_BIAS_TERMS:
        if term in memo:
            passed = False
            error_logs = f"Compliance Violation: Detected prohibited demographic bias pattern '{term}' in generated underwriting text."
            break
            
    return {
        "compliance_passed": passed, 
        "validation_error_logs": error_logs,
        "loop_retry_count": state["loop_retry_count"] + (0 if passed else 1)
    }

def agent_guardrails_input_filter(user_input_prompt: str) -> bool:
    """
    Local input guardrail routing logic protecting system prompt layers.
    Intercepts prompt injections, jailbreaks, or accidental PII entries.
    """
    malicious_indicators = ["ignore previous instructions", "system override", "reveal backend rules"]
    normalized_input = user_input_prompt.lower()
    
    for indicator in malicious_indicators:
        if indicator in normalized_input:
            print(f"[GUARDRAIL VIOLATION] Input blocked due to pattern match: '{indicator}'")
            return False
    return True

# =====================================================================
# 3. CONDITIONAL ROUTING EDGE
# =====================================================================

def route_compliance_gate(state: UnderwritingState) -> str:
    """Evaluates validation state flags to execute auto-healing loops or terminate processing."""
    if state["compliance_passed"]:
        print("[ROUTE GATE] Memo passed compliance validation. Finalizing execution thread.")
        return "approved"
        
    if state["loop_retry_count"] >= 3:
        print("[ROUTE GATE] Maximum auto-healing retry threshold reached. Hard-stopping token pipeline.")
        return "max_retries"
        
    print(f"[ROUTE GATE] Compliance violation caught! Routing to Auto-Healing Loop (Attempt {state['loop_retry_count']}/3)")
    return "needs_remediation"

# =====================================================================
# 4. ASYNCHRONOUS GRAPH COMPILATION LIFECYCLE
# =====================================================================

async def compile_underwriting_graph():
    """
    Asynchronously builds, links, and compiles the LangGraph state machine.
    Discovers all external MCP protocol servers at runtime and binds them
    to the active local LLM tool belt during server initialization.
    """
    print(f"[GRAPH INITIALIZATION] Bootstrapping local AI engine ({OLLAMA_MODEL_NAME})...")
    # 1. Initialize your custom local model instance
    llm = _get_active_llm_engine()
    
    # 2. Build local core business tools list
    core_tools = [predict_loan_default_risk, query_credit_policy_kb]
    
    # 3. Discover and append protocol-driven enterprise tools at runtime
    print("[GRAPH INITIALIZATION] Contacting MCP discovery broker layer...")
    mcp_tools = await get_mcp_tools()
    all_functional_tools = core_tools + mcp_tools
    
    # 4. Bind the combined toolbelt directly to the local model
    if hasattr(llm, "bind_tools"):
        llm = llm.bind_tools(all_functional_tools)
    
    # 5. Assemble the Directed Acyclic Graph (DAG) Struct
    workflow_graph = StateGraph(UnderwritingState)

    # Register Nodes
    workflow_graph.add_node("policy_fetcher", fetch_policy_context_node)
    workflow_graph.add_node("risk_predictor", execute_risk_model_node)
    workflow_graph.add_node("memo_synthesizer", synthesize_credit_memo_node)
    workflow_graph.add_node("compliance_linter", audit_compliance_guardrails_node)

    # Wire Edges
    workflow_graph.add_edge(START, "policy_fetcher")
    workflow_graph.add_edge("policy_fetcher", "risk_predictor")
    workflow_graph.add_edge("risk_predictor", "memo_synthesizer")
    workflow_graph.add_edge("memo_synthesizer", "compliance_linter")

    # Wire Self-Healing Conditional Loop Path
    workflow_graph.add_conditional_edges(
        "compliance_linter",
        route_compliance_gate,
        {
            "approved": END,
            "max_retries": END,
            "needs_remediation": "memo_synthesizer"
        }
    )

    # 6. Compile with a Local In-Memory Checkpointer
    local_memory = MemorySaver()
    compiled_platform_graph = workflow_graph.compile(checkpointer=local_memory)
    
    print(f"[GRAPH SUCCESS] Successfully compiled underwriting engine with {len(all_functional_tools)} active runtime tools.")
    return compiled_platform_graph