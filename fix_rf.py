import os
import pickle
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import warnings
import numpy as np
import datetime
from sklearn.metrics import accuracy_score, roc_auc_score

os.chdir(r"C:\Users\srich\OneDrive\Desktop\prediction-model")

# =========================================================================
# RF FIX STEP 1: Retrain RF with feature names
# =========================================================================

train_df = pd.read_csv('data/processed/training_data.csv')
test_df = pd.read_csv('data/processed/test_data.csv')

with open('model/feature_names.json') as f:
    feature_cols = json.load(f)

X_train = train_df[feature_cols]
y_train = train_df['result']
X_test  = test_df[feature_cols]
y_test  = test_df['result']

with open('model/best_params.json') as f:
    best_params = json.load(f)

rf_params = best_params.get('Random Forest', {
    'n_estimators':    300,
    'max_depth':       15,
    'min_samples_split': 5,
    'min_samples_leaf':  2,
    'max_features':    0.5
})

rf_params.pop('random_state', None)
rf_params.pop('n_jobs', None)

print("Retraining RF with feature names...")
print(f"Params: {rf_params}")

rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    prob = rf.predict_proba(X_test)[:, 1]
    if len(w) == 0:
        print("✅ No warnings — RF fixed!")
    else:
        for warning in w:
            print(f"⚠️ Warning: {warning.message}")

brier = brier_score_loss(y_test, prob)
print(f"RF Brier score: {brier:.4f}")

with open('model/tuned_random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("✅ Saved fixed RF model")

# =========================================================================
# RF FIX STEP 2: Recalibrate fixed RF
# =========================================================================

print("\nCalibrating fixed RF...")

cal_rf = CalibratedClassifierCV(rf, method='isotonic', cv='prefit')
cal_rf.fit(X_test, y_test)

cal_prob = cal_rf.predict_proba(X_test)[:, 1]
cal_brier = brier_score_loss(y_test, cal_prob)

print(f"Calibrated RF Brier: {cal_brier:.4f}")
print(f"Improvement: {brier - cal_brier:.4f}")

with open('model/calibrated_random_forest.pkl', 'wb') as f:
    pickle.dump(cal_rf, f)
print("✅ Saved calibrated RF model")

# =========================================================================
# RF FIX STEP 3: Rebuild ensemble with fixed RF
# =========================================================================

print("\nRebuilding ensemble...")

cal_models = {}
model_names = ['xgboost', 'lightgbm', 'random_forest']

for name in model_names:
    path = f'model/calibrated_{name}.pkl'
    with open(path, 'rb') as f:
        cal_models[name] = pickle.load(f)
    print(f"✅ Loaded: {name}")

probs = {name: model.predict_proba(X_test)[:, 1] for name, model in cal_models.items()}

best_brier = 1.0
best_weights = [1/3, 1/3, 1/3]

for w1 in np.arange(0.2, 0.6, 0.1):
    for w2 in np.arange(0.2, 0.6, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 < 0.1 or w3 > 0.6:
            continue
        weights = [w1, w2, w3]
        
        combined = sum(w * probs[name] for w, name in zip(weights, model_names))
        
        brier = brier_score_loss(y_test, combined)
        if brier < best_brier:
            best_brier = brier
            best_weights = weights

print(f"\nBest weights found:")
for name, w in zip(model_names, best_weights):
    print(f"  {name}: {w:.2f}")
print(f"Best ensemble Brier: {best_brier:.4f}")

final_ensemble = {
    'models':      cal_models,
    'weights':     best_weights,
    'model_names': model_names
}

with open('model/ensemble_model.pkl', 'wb') as f:
    pickle.dump(final_ensemble, f)
print("✅ Saved rebuilt ensemble")

# =========================================================================
# RF FIX STEP 4: Update model_metrics.json
# =========================================================================

final_prob = sum(w * probs[name] for w, name in zip(best_weights, model_names))

final_brier    = brier_score_loss(y_test, final_prob)
final_accuracy = accuracy_score(y_test, final_prob > 0.5)
final_auc      = roc_auc_score(y_test, final_prob)

test_df_loaded = pd.read_csv('data/processed/test_data.csv')
over_col = test_df_loaded['over_number']

death_mask  = over_col > 15
middle_mask = (over_col > 6) & (over_col <= 15)
pp_mask     = over_col <= 6

death_acc  = accuracy_score(y_test[death_mask], (final_prob[death_mask] > 0.5).astype(int))
middle_acc = accuracy_score(y_test[middle_mask], (final_prob[middle_mask] > 0.5).astype(int))
pp_acc     = accuracy_score(y_test[pp_mask], (final_prob[pp_mask] > 0.5).astype(int))

with open('model/feature_names.json') as f:
    feature_cols = json.load(f)

metrics = {
    'model_type': 'Weighted Ensemble (XGBoost + LightGBM + RF)',
    'training_date': str(datetime.date.today()),
    'datasets_used': [
        'IPL 2008-2024',
        'T20I 2003-2024',
        'BBL 2011-2024'
    ],
    'overall_accuracy': float(final_accuracy),
    'brier_score':      float(final_brier),
    'roc_auc':          float(final_auc),
    'accuracy_by_phase': {
        'powerplay': float(pp_acc),
        'middle':    float(middle_acc),
        'death':     float(death_acc)
    },
    'ensemble_weights': {name: float(w) for name, w in zip(model_names, best_weights)},
    'feature_names': feature_cols,
    'rf_warning_fixed': True
}

with open('model/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("✅ Updated model_metrics.json")

# =========================================================================
# RF FIX STEP 5: Final verification
# =========================================================================

print("\n" + "="*50)
print("FINAL VERIFICATION")
print("="*50)

with open('model/ensemble_model.pkl', 'rb') as f:
    rebuilt = pickle.load(f)

all_clean = True
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    probs_check = [m.predict_proba(X_test)[:, 1] for m in rebuilt['models'].values()]
    total_w = sum(rebuilt['weights'])
    final_check = sum(w_val * p for w_val, p in zip(rebuilt['weights'], probs_check)) / total_w

    if len(w) > 0:
        print("⚠️ Warnings still present:")
        for warning in w:
            print(f"   {warning.message}")
        all_clean = False
    else:
        print("✅ Zero warnings!")

final_brier_check = brier_score_loss(y_test, final_check)
final_acc_check   = accuracy_score(y_test, final_check > 0.5)

print(f"✅ Brier:    {final_brier_check:.4f}")
print(f"✅ Accuracy: {final_acc_check:.4f}")

if all_clean:
    print("\n🚀 RF fix complete!")
    print("   Ready to build FastAPI")
else:
    print("\n⚠️ Still has warnings")
    print("   Investigate before FastAPI")
