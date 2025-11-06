# xgb_matchup_model.py (Final Role-Based Version)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

# -------------------------------
# 1. Load Datasets
# -------------------------------
stats_df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv")
roles_df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/models/player_roles_cleaned.csv")

# Create is_wicket if not exists
if 'is_wicket' not in stats_df.columns:
    stats_df['is_wicket'] = (~stats_df['dismissal_type'].isna()).astype(int)

# Drop missing essentials
stats_df = stats_df.dropna(subset=['batsman', 'bowler', 'venue', 'phase', 'runs_scored'])

# Convert date if available
if 'date' in stats_df.columns:
    stats_df['date'] = pd.to_datetime(stats_df['date'], errors='coerce')

# -------------------------------
# 2. Merge Player Role Info (simplified for your CSV)
# -------------------------------
roles_df.columns = roles_df.columns.str.strip().str.lower()
roles_df['player_name'] = roles_df['player_name'].astype(str).str.strip().str.lower()

stats_df['batsman'] = stats_df['batsman'].astype(str).str.strip().str.lower()
stats_df['bowler'] = stats_df['bowler'].astype(str).str.strip().str.lower()

# Merge only role info (franchise optional)
stats_df = stats_df.merge(
    roles_df[['player_name', 'role']],
    how='left',
    left_on='bowler',
    right_on='player_name',
    suffixes=('', '_bowler')
)
stats_df = stats_df.merge(
    roles_df[['player_name', 'role']],
    how='left',
    left_on='batsman',
    right_on='player_name',
    suffixes=('', '_batsman')
)

# Rename for clarity
stats_df.rename(columns={
    'role': 'bowler_role',
    'role_batsman': 'batsman_role'
}, inplace=True)

# Fill missing roles
stats_df['bowler_role'] = stats_df['bowler_role'].fillna('unknown')
stats_df['batsman_role'] = stats_df['batsman_role'].fillna('unknown')

# -------------------------------
# 3. Feature Engineering
# -------------------------------
# Recent batting form (mean of last 3 innings)
stats_df['recent_bat_form'] = stats_df.groupby('batsman')['runs_scored'].transform(lambda x: x.rolling(3, min_periods=1).mean())

# Bowler effectiveness (wickets per 10 balls)
stats_df['bowler_form'] = stats_df.groupby('bowler')['is_wicket'].transform(lambda x: x.rolling(10, min_periods=1).sum())

# Convert phase to numeric representation
phase_map = {'powerplay': 1, 'middle': 2, 'death': 3}
stats_df['phase_num'] = stats_df['phase'].str.lower().map(phase_map).fillna(2)

# Strike rate differential proxy
stats_df['sr_diff'] = stats_df['runs_scored'] - (stats_df['bowler_form'] * 10)

# -------------------------------
# 4. Feature Selection
# -------------------------------
features = [
    'batsman', 'bowler', 'venue', 'phase', 
    'recent_bat_form', 'bowler_form', 'phase_num',
    'bowler_role', 'batsman_role', 'sr_diff'
]
target = 'is_wicket'

df = stats_df[features + [target]].fillna(0)

# -------------------------------
# 5. Encode Categoricals
# -------------------------------
encoders = {}
for col in ['batsman', 'bowler', 'venue', 'phase', 'bowler_role', 'batsman_role']:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

# -------------------------------
# 6. Train-Test Split
# -------------------------------
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

imbalance_ratio = (len(y) - y.sum()) / y.sum()

# -------------------------------
# 7. Train XGBoost Model
# -------------------------------
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=imbalance_ratio,
    max_depth=7,
    n_estimators=300,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# 8. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("✅ Role-Enhanced Model Evaluation")
print("----------------------------------")
print(f"Accuracy: {acc:.3f}")
print(f"ROC AUC : {roc:.3f}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 9. Save Model & Artifacts
# -------------------------------
joblib.dump(model, "xgb_dismissal_model.pkl")
joblib.dump(encoders, "xgb_label_encoders.pkl")
joblib.dump(features, "xgb_model_features.pkl")

print("\n💾 Model, encoders, and features saved successfully!")
