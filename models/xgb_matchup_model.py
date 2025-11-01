# xgb_matchup_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv")

# Create is_wicket if not exists
if 'is_wicket' not in df.columns:
    df['is_wicket'] = (~df['dismissal_type'].isna()).astype(int)

# Drop rows with missing core columns
df = df.dropna(subset=['batsman', 'bowler', 'venue', 'phase', 'runs_scored'])

# Convert date if available
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# -------------------------------
# 2. Feature Engineering
# -------------------------------
# Recent batting form (mean of last 3 innings)
df['recent_bat_form'] = df.groupby('batsman')['runs_scored'].transform(lambda x: x.rolling(3, min_periods=1).mean())

# Bowler effectiveness (wickets per 10 balls)
df['bowler_form'] = df.groupby('bowler')['is_wicket'].transform(lambda x: x.rolling(10, min_periods=1).sum())

# -------------------------------
# 3. Select Features
# -------------------------------
features = ['batsman', 'bowler', 'venue', 'phase', 'recent_bat_form', 'bowler_form']
target = 'is_wicket'

df = df[features + [target]].fillna(0)

# Encode categoricals
encoders = {}
for col in ['batsman', 'bowler', 'venue', 'phase']:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

# -------------------------------
# 4. Split Data
# -------------------------------
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Handle imbalance
imbalance_ratio = (len(y) - y.sum()) / y.sum()

# -------------------------------
# 5. Train XGBoost
# -------------------------------
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=imbalance_ratio,
    max_depth=6,
    n_estimators=250,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("✅ Model Evaluation")
print("------------------------")
print(f"Accuracy: {acc:.3f}")
print(f"ROC AUC : {roc:.3f}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 7. Save Model & Encoders
# -------------------------------
joblib.dump(model, "xgb_dismissal_model.pkl")
joblib.dump(encoders, "xgb_label_encoders.pkl")
joblib.dump(features, "xgb_model_features.pkl")

print("\n💾 Model, encoders, and features saved successfully!")
