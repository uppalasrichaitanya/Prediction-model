import pandas as pd
import numpy as np
import json, pickle, time
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

print("Loading data...")
train_df = pd.read_csv('data/processed/training_data.csv')
test_df = pd.read_csv('data/processed/test_data.csv')
with open('model/feature_names.json') as f:
    feature_cols = json.load(f)

X_train = train_df[feature_cols].values
y_train = train_df['result'].values
X_test = test_df[feature_cols].values
y_test = test_df['result'].values

with open('model/best_params.json') as f:
    best_params = json.load(f)

tuned_models = {}
tuned_results = []

model_classes = {
    'XGBoost': (XGBClassifier, {'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1, 'verbosity': 0}),
    'LightGBM': (LGBMClassifier, {'random_state': 42, 'n_jobs': -1, 'verbose': -1}),
    'Random Forest': (RandomForestClassifier, {'random_state': 42, 'n_jobs': -1})
}

for name, (ModelClass, fixed_params) in model_classes.items():
    if name not in best_params: continue
    print(f'Training tuned {name}...')
    params = {**best_params[name], **fixed_params}
    model = ModelClass(**params)
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    result = {
        'model': f'{name} (tuned)',
        'accuracy': accuracy_score(y_test, y_pred),
        'brier': brier_score_loss(y_test, y_prob),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'train_time': train_time
    }
    tuned_results.append(result)
    tuned_models[name] = model
    safe = name.lower().replace(' ', '_')
    with open(f'model/tuned_{safe}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f'  Brier: {result["brier"]:.4f}  Accuracy: {result["accuracy"]:.4f}')

tuned_df = pd.DataFrame(tuned_results).sort_values('brier')
print('\\n=== TUNED MODEL RESULTS ===')
print(tuned_df.to_string(index=False))
