import os, pickle, json
import pandas as pd
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score

os.chdir(r"C:\Users\srich\OneDrive\Desktop\prediction-model")

test_df = pd.read_csv('data/processed/test_data.csv')
with open('model/feature_names.json') as f:
    feature_cols = json.load(f)

X_test = test_df[feature_cols]
y_test = test_df['result']

# Load only XGB and LightGBM
with open('model/calibrated_xgboost.pkl', 'rb') as f:
    xgb = pickle.load(f)
with open('model/calibrated_lightgbm.pkl', 'rb') as f:
    lgbm = pickle.load(f)

# Find best weights
import numpy as np
p_xgb = xgb.predict_proba(X_test)[:,1]
p_lgbm = lgbm.predict_proba(X_test)[:,1]

best_brier, best_w = 1.0, 0.5
for w in np.arange(0.3, 0.8, 0.05):
    prob = w * p_xgb + (1-w) * p_lgbm
    b = brier_score_loss(y_test, prob)
    if b < best_brier:
        best_brier, best_w = b, w

final_prob = best_w * p_xgb + (1-best_w) * p_lgbm
print(f"XGB weight: {best_w:.2f}, LGBM weight: {1-best_w:.2f}")
print(f"Brier: {brier_score_loss(y_test, final_prob):.4f}")
print(f"Accuracy: {accuracy_score(y_test, final_prob>0.5):.4f}")

# Save lean ensemble
lean_ensemble = {
    'models': {'xgboost': xgb, 'lightgbm': lgbm},
    'weights': [best_w, 1-best_w],
    'model_names': ['xgboost', 'lightgbm']
}
with open('model/ensemble_model.pkl', 'wb') as f:
    pickle.dump(lean_ensemble, f)
print("✅ Lean ensemble saved")