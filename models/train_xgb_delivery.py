# train_xgb_delivery.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
from math import sqrt
import os

# -------------------------
# Config - change path if needed
# -------------------------
INPUT_CSV = "D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv"
MODEL_OUT = "xgb_delivery_model.pkl"
ENC_OUT = "xgb_delivery_label_encoders.pkl"
FEATURES_OUT = "xgb_delivery_features.pkl"
RANDOM_STATE = 42

print("Loading:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)

# Basic cleaning
for c in ['batsman','bowler','venue','phase','dismissal_type']:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# Create binary target 'is_wicket' if missing
if 'is_wicket' not in df.columns:
    df['is_wicket'] = (~df['dismissal_type'].isna()).astype(int)

# ensure numeric runs column
if 'runs_scored' not in df.columns:
    df['runs_scored'] = 0

# derive simple 'over' if available, else 0
if 'over' not in df.columns:
    if 'ball_no' in df.columns:
        df['over'] = df['ball_no'].astype(str).apply(lambda x: int(float(x)) if x and x!='nan' else 0)
    else:
        df['over'] = 0

# derive phase if missing
def phase_from_over(o):
    try: o = int(o)
    except: return 'middle'
    if o <= 6: return 'powerplay'
    elif o <= 15: return 'middle'
    else: return 'death'

df['phase'] = df['phase'].replace('nan','').fillna('')
df.loc[df['phase'].isin(['','nan']), 'phase'] = df.loc[df['phase'].isin(['','nan']), 'over'].apply(phase_from_over)

# lowercase keys for mapping
df['batsman_l'] = df['batsman'].str.lower()
df['bowler_l'] = df['bowler'].str.lower()
df['venue_l'] = df['venue'].str.lower()
df['phase_l'] = df['phase'].str.lower()

# ---- Feature engineering (per-delivery features) ----
print("Engineering features...")

# recent batting form: average runs in last 3 matches (approx by match_id if exists)
if 'match_id' in df.columns:
    bat_match = df.groupby(['batsman_l','match_id']).agg(match_runs=('runs_scored','sum')).reset_index()
    bat_match = bat_match.sort_values(['batsman_l','match_id'])
    bat_match['recent_bat_form'] = bat_match.groupby('batsman_l')['match_runs'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    # merge back: join on batsman & match_id
    df = df.merge(bat_match[['batsman_l','match_id','recent_bat_form']], on=['batsman_l','match_id'], how='left')
else:
    df['recent_bat_form'] = 0.0

# bowler recent wickets (last 50 balls)
df = df.sort_values(['bowler_l'])
df['bowler_wickets_last50'] = df.groupby('bowler_l')['is_wicket'].transform(lambda x: x.shift(1).rolling(50, min_periods=1).sum())
df['bowler_wickets_last50'] = df['bowler_wickets_last50'].fillna(0)

# head-to-head recent: runs vs bowler in last 50 deliveries for the batsman
df['batsman_runs_vs_bowler_last50'] = df.groupby(['batsman_l','bowler_l'])['runs_scored'].transform(lambda x: x.shift(1).rolling(50, min_periods=1).sum())
df['batsman_runs_vs_bowler_last50'] = df['batsman_runs_vs_bowler_last50'].fillna(0)

# phase-specific features: runs scored by batsman in this phase historically
bat_phase = df.groupby(['batsman_l','phase_l']).agg(bat_phase_runs=('runs_scored','sum'), bat_phase_balls=('batsman_l','count')).reset_index()
bat_phase['bat_phase_rpb'] = bat_phase['bat_phase_runs'] / bat_phase['bat_phase_balls'].replace(0,1)
df = df.merge(bat_phase[['batsman_l','phase_l','bat_phase_rpb']], on=['batsman_l','phase_l'], how='left')
df['bat_phase_rpb'] = df['bat_phase_rpb'].fillna(0)

# bowler-phase wicket rate
bow_phase = df.groupby(['bowler_l','phase_l']).agg(bp_wickets=('is_wicket','sum'), bp_balls=('bowler_l','count')).reset_index()
bow_phase['bp_wicket_rate'] = bow_phase['bp_wickets'] / bow_phase['bp_balls'].replace(0,1)
df = df.merge(bow_phase[['bowler_l','phase_l','bp_wicket_rate']], on=['bowler_l','phase_l'], how='left')
df['bp_wicket_rate'] = df['bp_wicket_rate'].fillna(0)

# simple interaction: is_left_right if you have data of batting/bowling style (optional) -- placeholder 0
df['handedness_match'] = 0

# choose features for model
feature_cols = [
    'batsman_l','bowler_l','venue_l','phase_l',
    'recent_bat_form','bowler_wickets_last50','batsman_runs_vs_bowler_last50',
    'bat_phase_rpb','bp_wicket_rate'
]

# ensure these columns exist
for c in feature_cols:
    if c not in df.columns:
        df[c] = 0

# drop rows with missing essential cols
df = df.dropna(subset=['batsman_l','bowler_l','venue_l','phase_l'])

# target
y = df['is_wicket'].astype(int)

# encode categorical columns with LabelEncoder dictionaries
categorical_cols = ['batsman_l','bowler_l','venue_l','phase_l']
encoders = {}
X = df[feature_cols].copy()

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = X[col].astype(str).fillna('unknown')
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# numeric columns
num_cols = [c for c in feature_cols if c not in categorical_cols]
X[num_cols] = X[num_cols].fillna(0)

# train-test split (stratify by target due to imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# handle imbalance via scale_pos_weight
pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

print("Training XGB Classifier (delivery-level)...")
model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='logloss',
    tree_method='hist'
)

model.fit(X_train, y_train)

# Evaluate
y_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test))>1 else 0.0

print("Accuracy: {:.3f}  ROC-AUC: {:.3f}".format(acc, roc))
print("Classification report:")
print(classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(model, MODEL_OUT)
joblib.dump(encoders, ENC_OUT)
joblib.dump(feature_cols, FEATURES_OUT)
print("Saved model and encoders:", MODEL_OUT, ENC_OUT, FEATURES_OUT)
