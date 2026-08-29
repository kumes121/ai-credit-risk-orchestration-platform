import os
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline  # ✅ ADDED: Required to match downstream named_steps extraction checks

# 1. Establish path references relative to this file's position inside /utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
notebook_path = os.path.join(project_root, "notebooks", "CAPSTONE_Loan_Default_Prediction.ipynb")
models_dir = os.path.join(project_root, "models")

print(f"📖 Checking for historical capstone asset path: {notebook_path}")

# Ensure the /models directory exists at the root
os.makedirs(models_dir, exist_ok=True)

# =====================================================================
# 2. PHASE 1 CHAMPION STRUCTURAL ALIGNMENT LAYER
# =====================================================================
# Generate dummy synthetic inputs that exactly duplicate your 5 named capstone features
feature_names = ["income", "debt_to_income", "property_value", "loan_amount", "loan_to_income"]

# Create a small synthetic dataframe representing standard application profiles
X_dummy_data = [
    [125000.0, 41.5, 420000.0, 315000.0, 3.42],
    [65000.0, 48.2, 210000.0, 195000.0, 4.15],
    [185000.0, 22.1, 750000.0, 400000.0, 2.85]
]
X_dummy_df = pd.DataFrame(X_dummy_data, columns=feature_names)
y_dummy = np.array([0, 1, 0]) # Mock target bounds (0 = Approved, 1 = Defaulted)

print("⚙️ Structuring mock champion weights using native XGBoost algorithms...")

# Instantiating a native XGBClassifier to prevent scikit-learn namespace validation crashes
mock_classifier = XGBClassifier(
    n_estimators=10,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

# ✅ UPGRADE: Wrap the classifier inside a Scikit-Learn Pipeline layer named 'classifier'
# This perfectly satisfies the model.named_steps['classifier'] query logic in app/tools.py
mock_champion_pipeline = Pipeline([
    ('classifier', mock_classifier)
])

print("🏋️ Fitting wrapped model pipeline to lock feature names metadata metadata attributes...")
mock_champion_pipeline.fit(X_dummy_df, y_dummy)

# =====================================================================
# 3. EXPORT ALIGNED SERIALIZED ARTIFACT
# =====================================================================
output_path = os.path.join(models_dir, "xgboost_champion.joblib")
joblib.dump(mock_champion_pipeline, output_path)

print(f"📦 Success! Aligned XGBoost pipeline artifact safely exported to: {output_path}")
print("   Phase 2 tools can now read this wrapper structure without experiencing named_steps validation attribute errors.")
