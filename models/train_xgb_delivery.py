import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Config
# -------------------------
INPUT_CSV = "D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv"
MODEL_OUT = "xgb_delivery_model.pkl"
ENC_OUT = "xgb_delivery_label_encoders.pkl"
FEATURES_OUT = "xgb_delivery_features.pkl"
RANDOM_STATE = 42

print("📂 Loading:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)

# -------------------------
# Data Cleaning
# -------------------------
for c in ['batsman', 'bowler', 'venue', 'phase', 'dismissal_type']:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.lower()

if 'is_wicket' not in df.columns:
    df['is_wicket'] = (~df['dismissal_type'].isna()).astype(int)

if 'runs_scored' not in df.columns:
    df['runs_scored'] = 0

if 'over' not in df.columns:
    if 'ball_no' in df.columns:
        df['over'] = df['ball_no'].astype(str).apply(lambda x: int(float(x)) if x and x != 'nan' else 0)
    else:
        df['over'] = 0

def phase_from_over(o):
    try:
        o = int(o)
    except:
        return 'middle'
    if o <= 6: return 'powerplay'
    elif o <= 15: return 'middle'
    else: return 'death'

df['phase'] = df['phase'].replace('nan', '').fillna('')
df.loc[df['phase'].isin(['', 'nan']), 'phase'] = df['over'].apply(phase_from_over)

df['batsman_l'] = df['batsman'].str.lower()
df['bowler_l'] = df['bowler'].str.lower()
df['venue_l'] = df['venue'].str.lower()
df['phase_l'] = df['phase'].str.lower()

# -------------------------
# Feature Engineering
# -------------------------
print("⚙️ Engineering features...")

# recent batting form (average in last 3 matches)
if 'match_id' in df.columns:
    bat_match = df.groupby(['batsman_l', 'match_id']).agg(match_runs=('runs_scored', 'sum')).reset_index()
    bat_match = bat_match.sort_values(['batsman_l', 'match_id'])
    bat_match['recent_bat_form'] = bat_match.groupby('batsman_l')['match_runs'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df = df.merge(bat_match[['batsman_l', 'match_id', 'recent_bat_form']],
                  on=['batsman_l', 'match_id'], how='left')
else:
    df['recent_bat_form'] = df.groupby('batsman_l')['runs_scored'].transform('mean')

# bowler recent wickets (last 50 balls)
df = df.sort_values(['bowler_l'])
df['bowler_wickets_last50'] = df.groupby('bowler_l')['is_wicket'].transform(
    lambda x: x.shift(1).rolling(50, min_periods=1).sum())
df['bowler_wickets_last50'] = df['bowler_wickets_last50'].fillna(0)

# batsman runs vs bowler (last 50 deliveries)
df['batsman_runs_vs_bowler_last50'] = df.groupby(['batsman_l', 'bowler_l'])['runs_scored'].transform(
    lambda x: x.shift(1).rolling(50, min_periods=1).sum())
df['batsman_runs_vs_bowler_last50'] = df['batsman_runs_vs_bowler_last50'].fillna(0)

# batsman phase performance
bat_phase = df.groupby(['batsman_l', 'phase_l']).agg(
    bat_phase_runs=('runs_scored', 'sum'),
    bat_phase_balls=('batsman_l', 'count')
).reset_index()
bat_phase['bat_phase_rpb'] = bat_phase['bat_phase_runs'] / bat_phase['bat_phase_balls'].replace(0, 1)
df = df.merge(bat_phase[['batsman_l', 'phase_l', 'bat_phase_rpb']],
              on=['batsman_l', 'phase_l'], how='left')
df['bat_phase_rpb'] = df['bat_phase_rpb'].fillna(0)

# bowler phase wicket rate
bow_phase = df.groupby(['bowler_l', 'phase_l']).agg(
    bp_wickets=('is_wicket', 'sum'),
    bp_balls=('bowler_l', 'count')
).reset_index()
bow_phase['bp_wicket_rate'] = bow_phase['bp_wickets'] / bow_phase['bp_balls'].replace(0, 1)
df = df.merge(bow_phase[['bowler_l', 'phase_l', 'bp_wicket_rate']],
              on=['bowler_l', 'phase_l'], how='left')
df['bp_wicket_rate'] = df['bp_wicket_rate'].fillna(0)

# venue wicket rate
venue_stats = df.groupby('venue_l').agg(
    venue_wicket_rate=('is_wicket', 'mean'),
    venue_avg_runs=('runs_scored', 'mean')
).reset_index()
df = df.merge(venue_stats, on='venue_l', how='left')

df['handedness_match'] = 0

# -------------------------
# Feature set
# -------------------------
feature_cols = [
    'batsman_l', 'bowler_l', 'venue_l', 'phase_l',
    'recent_bat_form', 'bowler_wickets_last50',
    'batsman_runs_vs_bowler_last50', 'bat_phase_rpb',
    'bp_wicket_rate', 'venue_wicket_rate', 'venue_avg_runs'
]

for c in feature_cols:
    if c not in df.columns:
        df[c] = 0

df = df.dropna(subset=['batsman_l', 'bowler_l', 'venue_l', 'phase_l'])
y = df['is_wicket'].astype(int)

# Label Encoding
categorical_cols = ['batsman_l', 'bowler_l', 'venue_l', 'phase_l']
encoders = {}
X = df[feature_cols].copy()

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = X[col].astype(str).fillna('unknown')
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

num_cols = [c for c in feature_cols if c not in categorical_cols]
X[num_cols] = X[num_cols].fillna(0)

# -------------------------
# Handle imbalance via SMOTE
# -------------------------
print("🧮 Applying SMOTE balancing...")
sm = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.25)  # 1 wicket : 4 non-wicket
X_res, y_res = sm.fit_resample(X, y)

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, stratify=y_res, random_state=RANDOM_STATE
)

# -------------------------
# Model Training
# -------------------------
print("🚀 Training optimized XGBoost model...")
model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=400,
    learning_rate=0.04,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.3,
    reg_lambda=1.2,
    reg_alpha=0.4,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    tree_method='hist'
)

model.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"Accuracy: {acc:.3f}")
print(f"ROC-AUC: {roc:.3f}")
print(f"F1-score (for wicket): {f1:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# Save Artifacts
# -------------------------
joblib.dump(model, MODEL_OUT)
joblib.dump(encoders, ENC_OUT)
joblib.dump(feature_cols, FEATURES_OUT)
print("\n✅ Saved model and encoders:", MODEL_OUT, ENC_OUT, FEATURES_OUT)
