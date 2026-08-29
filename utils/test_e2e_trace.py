import os
import sys
import asyncio

# Ensure the root project directory is on the path for clean app/ imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check command line arguments to allow dynamic testing of AWS Bedrock mode
if len(sys.argv) > 1 and sys.argv[1].upper() == "AWS":
    os.environ["ENVIRONMENT_MODE"] = "AWS"
    print("[TRACE CONFIG] Activating AWS Cloud Production Integration Mode...")
else:
    os.environ["ENVIRONMENT_MODE"] = "LOCAL"
    os.environ["OLLAMA_MODEL_NAME"] = "qwen2.5-coder:3b"
    print("[TRACE CONFIG] Activating Zero-Cost Local Sandbox Mode...")

# Force the MCP command to empty for this trace script to skip active sub-processes
os.environ["MCP_SERVER_COMMAND"] = "" 

from app.graph import compile_underwriting_graph

async def run_end_to_end_trace():
    print("=" * 70)
    print("🚀 INITIALIZING END-TO-END AGENTIC UNDERWRITING PLATFORM TRACE")
    print("=" * 70)
    
    # 1. Simulate the FastAPI asynchronous startup compilation loop
    print("\n[STEP 1/3] Compiling LangGraph State Machine & Discovering Tools...")
    try:
        orchestration_engine = await compile_underwriting_graph()
        print("✅ Orchestration engine successfully compiled.")
    except Exception as e:
        print(f"❌ Critical Compilation Failure: {str(e)}")
        return

    # 2. Build a mock credit profile payload aligned precisely with your graph schema
    print("\n[STEP 2/3] Constructing Mock Borrower Profile Payload...")
    # Flattened the features layout to ensure exact structural alignment with state lookups
    mock_payload = {
        "borrower_id": "MOCK-HL-2026",
        "raw_features": {
            "income": 145000.0,
            "debt_to_income": 41.2,
            "property_value": 480000.0,
            "loan_amount": 310000.0
        },
        "vector_policy_context": "",
        "model_prediction": {},
        "generated_credit_memo": "",
        "compliance_passed": False,
        "loop_retry_count": 0,
        "validation_error_logs": ""
    }
    print(f"-> Ingesting Profile: {mock_payload['borrower_id']}")
    print(f"-> Financial Metrics: DTI {mock_payload['raw_features']['debt_to_income']}% | Income ${mock_payload['raw_features']['income']:,}")

    # 3. Trigger the asynchronous graph pipeline using a unique thread tracking token
    print("\n[STEP 3/3] Executing Asynchronous Multi-Node Processing Loop...")
    config = {"configurable": {"thread_id": "trace_session_001"}}
    
    try:
        final_state = await orchestration_engine.ainvoke(mock_payload, config=config)
        
        print("\n" + "=" * 70)
        print("🎉 TRACE RUN COMPLETED SUCCESSFULLY - PROCESSING METRICS")
        print("=" * 70)
        print(f"👤 Borrower Profile ID  : {final_state['borrower_id']}")
        print(f"📈 Default Probability  : {final_state['model_prediction'].get('raw_risk_probability', 'N/A')}")
        print(f"⚖️ Underwriting Action  : {final_state['model_prediction'].get('underwriting_action', 'N/A')}")
        print(f"🛡️ Compliance Pass Gate : {final_state['compliance_passed']}")
        print(f"🔄 Auto-Healing Loops   : {final_state['loop_retry_count']} / 3")
        
        print("\n📄 Generated Credit Memo Snippet:")
        print("-" * 70)
        # Display the first 300 characters of the synthesized narrative text
        memo_snippet = final_state.get('generated_credit_memo', '')
        print(memo_snippet[:300] + ("..." if len(memo_snippet) > 300 else ""))
        print("-" * 70)

    except Exception as e:
        print(f"❌ Critical Runtime Pipeline Failure: {str(e)}")

if __name__ == "__main__":
    # Execute the asynchronous test loop inside the Python engine
    asyncio.run(run_end_to_end_trace())
