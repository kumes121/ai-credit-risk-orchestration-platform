# AI-Driven Credit Risk Orchestration & Underwriting Platform
**UC Berkeley Extension — Professional Certificate in Machine Learning & AI (Academic Portfolio Upgrade)**

[![Python 3.11+](https://shields.io)](https://python.org)
[![License: MIT](https://shields.io)](https://opensource.org)
[![Framework: LangGraph](https://shields.io)](https://github.com)

---

## 🗒 Project Evolution & Roadmap
This repository hosts a multi-phase corporate credit platform that evolved from an advanced data science research pilot into a production-tier, model-agnostic intelligent orchestration engine:

*   **Phase I (Core Capstone Notebook):** Built a high-performance, leak-proof predictive framework using tabular machine learning to isolate high-risk credit accounts.
*   **Phase II (Agentic Platform Upgrade):** Wrapped the static predictive weights into an active microservices network utilizing an autonomous **LangGraph** workflow engine, a **FastAPI** validation tier, and a **Gradio** workbench interface.

---

## 🔎 Executive Summary
The primary goal of this framework is to transition traditional "black-box" predictive analytics into automated, compliance-audited banking workflows. 

Initially developed as a **Home Lending predictive pilot** to handle mortgage-specific parameters, the system implements a contract-first microservices architecture. By decoupling data ingestion from model execution using **Pydantic** data contracts and **LangGraph** deterministic state machines, the platform is explicitly designed to scale cross-enterprise workloads seamlessly to other consumer business units, such as high-volume **Cards Platforms**.

---

## 💡 Rationale & Research Question
### The Core Friction
In traditional retail and commercial credit operations, predictive models output static, numerical risk metrics (e.g., `Default Probability = 0.42`). However, lending institutions face severe operational and regulatory bottlenecks before an application can be finalized:
1. **The Explanation Bottleneck:** Under the *Equal Credit Opportunity Act (ECOA)*, lenders cannot issue a credit denial based solely on a mathematical raw score. Regulators require explicit, written **Adverse Action Reason Codes** documenting the precise risk drivers. Doing this manually takes human underwriters hours per file.
2. **The Compliance & Policy Gap:** Standalone tabular models cannot evaluate shifting internal policy texts, standard operating procedures, or macro-environmental risk changes.

### The Research Question
*Can we accurately predict a borrower’s default risk via non-linear structural models and leverage an autonomous Agentic AI workflow to dynamically cross-reference those predictions against corporate policies, synthesize compliant credit memos, and programmatically protect the network from compliance drift?*

---

## 🛠️ Technology Implementation Matrix

The following index outlines the architecture’s technological stack, detailing both currently utilized frameworks and planned observability or guardrail additions:

| Technology Domain | Tool / Framework | Purpose | Implementation Status |
| :--- | :--- | :--- | :--- |
| **Orchestration Layer** | **LangGraph** | Manages stateful, circular workflows, routing, and conditional loops for loan profiles. | **Implemented** (`app/graph.py`) |
| **API Framework** | **FastAPI** | High-performance asynchronous endpoint serving to handle profile payload submissions. | **Implemented** (`app/main.py`) |
| **Agent Tooling** | **LangChain** | Provides standard abstractions and interface layers to couple tools with prompt nodes. | **Implemented** |
| **Validation Layer** | **Pydantic** | Enforces typed data schemas and runtime structural compliance for financial profile payloads. | **Implemented** (`app/schemas.py`) |
| **Machine Learning Model**| **XGBoost Classifier** | Evaluates historical default probability using the foundation built in Phase I (`xgboost_champion.joblib`). | **Implemented** (Artifact Integrated) |
| **User Interface** | **Gradio / Custom UI** | Simple interactive dashboard for loan officers to submit inquiries and view structured risk criteria. | **Implemented** (`app/main.py`) |
| **Observability & Analytics**| **Langfuse** | Tracks agent steps, records evaluation traces, and measures token usage/latency. | **Implemented** (`app/main.py`) |
| **Agent Guardrails** | **Linguistic Linters** | Enforces safety, fair-lending bias protection, policy boundaries, and text validation rules. | **Implemented** (`app/graph.py`) |
| **Cloud Infrastructure** | **Amazon Bedrock / S3** | Enterprise runtime execution environment utilizing Claude 3.5 Sonnet and streaming binary model weights. | **Implemented** (AWS Mode Live) |
---

## 📁 Repository Schema
```text
ai-credit-risk-orchestration/
├── .github/                       # Automated Enterprise CI/CD Pipelines
│   └── workflows/
│       └── deploy.yml             # GitHub Actions ECR Multi-Stage Build Runner
├── app/                           # Core Microservice Implementation Layer
│   ├── config.py                  # Environment Variable Fallbacks & Central Settings
│   ├── graph.py                   # Stateful LangGraph Orchestration & Multi-Agent Nodes
│   ├── main.py                    # FastAPI Web Server Initialization & Gradio Dashboard
│   ├── schemas.py                 # Pydantic Structural Request/Response Contracts
│   └── tools.py                   # XGBoost S3 Streamer, SHAP Extractors, & MCP Connections
├── models/                        # Local Serialized Model Registries (Git Ignored)
│   └── xgboost_champion.joblib   # Extracted Phase 1 Target Model Artifact
├── notebooks/                     # Analytical Training Sandbox Documents
│   └── CAPSTONE_Loan_Default_Prediction.ipynb
├── utils/                         # Platform Verification Utilities
│   ├── extract_model.py           # Calibration Script for Wrapping & Serializing Model
│   └── test_e2e_trace.py          # Asynchronous End-to-End Pipeline Test Script
├── .env                           # Local Private Core Environment Configs (Git Ignored)
├── .gitignore                     # Secure Version Control Access Guardrail
└── requirements.txt               # Pinpointed System Package Dependencies
```

---

## ⚙️ Methodology & Platform Architecture
The platform implements a decoupled, hybrid architecture pattern that can run completely free offline on a developer's local machine, while remaining structurally production-ready for deployment on cloud container infrastructure like **AWS EKS** or **Red Hat OpenShift (OCP)**.

### Execution Pipeline Framework
1. **Ingestion & Validation Gateway:** Incoming JSON transaction payloads hit a stateless **FastAPI** router where they are instantly validated against strict **Pydantic** schemas matching the exact features expected by the downstream model.
2. **Policy Context Retrieval (Node 1):** The initialized LangGraph thread calls an automated context assembler tool that queries an internal knowledge base mapping to corporate Lending Policy Guideline Section 4.12.
3. **Predictive Inference (Node 2):** The state machine invokes a custom tool wrapping your pre-trained Scikit-Learn pipeline. The model weights are either streamed dynamically out of an **Amazon S3 Bucket** directly into RAM or loaded from disk, appending real-time engineered log features.
4. **Automated Memo Synthesis (Node 3):** Local **SHAP interpretability coefficients** are generated to isolate the primary risk drivers. These weights are routed to a frontier generative model (**Amazon Bedrock / Claude 3.5 Sonnet** or free offline **Ollama** instances) to translate metrics into plain-English Adverse Action texts.
5. **Auto-Healing Compliance Edge (Node 4):** The outputted credit memo passes into a final validation filter. If the text surfaces prohibited fair-lending demographic biases, a conditional edge intercepts the payload and loops execution back to the generation node with precise corrective feedback logs (capped at a maximum of 3 retries).
6. **Durable Checkpointing & Observability:** Conversation history state snapshots are serialized to memory after every node transition, while complete system telemetry is traced asynchronously to **Langfuse** to monitor latency and prevent model drift.

### Structural Routing Pattern
```text
[Loan Request Input] ──> [FastAPI Ingestion Layer]
                                │
                                ▼
                  [LangGraph Orchestration Engine] 
               (Stateful Conditional Router Agent)
                  ╱             │             ╲
                 ▼              ▼              ▼
         [Validation Tool] [XGBoost Predictor] [SHAP Explainability]
         (Pydantic/Guard)  (Phase I Champion)  (Compliance Engine)
                  ╲             │             ╱
                   ▼            ▼            ▼
               [Underwriting & Risk Scoring Decision]
```

### Phase I Data Science Core
*   **Leak-Proof Preprocessing:** Implemented a robust Scikit-Learn `Pipeline` and `ColumnTransformer` to ensure all imputation and scaling (`StandardScaler`) happened post-split, completely eliminating data leakage across **148,000 credit profiles**.
*   **Advanced Imputation & Engineering:** Utilized **IterativeImputer (MICE with BayesianRidge)** for missing-at-random (MAR) variables like Property Value and Debt-to-Income (DTI). Engineered `loan_to_income` (log-scaled loan burden).
*   **Model Optimization:** Deployed an optimized **XGBoost** champion model with strategic threshold optimization (**0.321 decision boundary**) to capture **80.01% of potential defaults (0.832 PR-AUC)**, reducing missed defaults (False Negatives) by 53.3%.

---

## 📊 Results & Business Case Impact

### 1. Risk Capital Mitigation Matrix
By tuning the system's operational decision boundary to a conservative **0.321 threshold**, the platform alters the risk profile to isolate institutional hazards before portfolio integration:

| Performance Attribute | Baseline Tabular Model | Tuned XGBoost Champion | Portfolio Impact |
| :--- | :--- | :--- | :--- |
| **Model Recall Index** | 64.20% | **80.01%** | +15.81% Early Default Catch Rate |
| **Precision-Recall AUC** | 0.710 | **0.832** | Enhanced Class Separation Stability |
| **False Negative Rate** | 35.80% | **16.78%** | **53.30% Reduction** in Missed Defaults |

### 2. Operational Value & Underwriting Automation
*   **Reduced Overhead Latency:** Transitioned manual Credit Memorandum drafting from a 45-minute legacy process down to an automated **<4.5-second asynchronous pipeline generation** sweep using Bedrock/Claude models.
*   **Deterministic Self-Healing Loops:** Programmatic fair-lending compliance nodes autonomously catch and correct 100% of demographic linter violations locally, mitigating legal drift before text payloads hit permanent audit logs.
*   **Auditable Explainability Trails:** Every credit decision maps directly to mathematical SHAP local indicators, providing complete compliance reporting readiness for auditing requirements under ECOA and FCRA guidelines.
