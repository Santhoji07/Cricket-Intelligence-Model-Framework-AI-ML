# svm_match_outcome.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ---------- Step 1: Load dataset ----------
# Make sure match_results.csv has: match_id, team, opponent, venue, avg_team_runs, avg_team_wkts, match_result
df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/match_results.csv")

# Check required columns
required_cols = {'team', 'opponent', 'venue', 'avg_team_runs', 'avg_team_wkts', 'match_result'}
if not required_cols.issubset(set(df.columns)):
    raise ValueError(f"CSV must have columns: {required_cols}")

# ---------- Step 2: Data Cleaning ----------
print("Checking missing values before cleaning:")
print(df.isnull().sum())

# Drop rows with missing categorical or target columns
df = df.dropna(subset=['team', 'opponent', 'venue', 'match_result'])

# For numeric columns, fill missing values with the mean of the column
df['avg_team_runs'] = df['avg_team_runs'].fillna(df['avg_team_runs'].mean())
df['avg_team_wkts'] = df['avg_team_wkts'].fillna(df['avg_team_wkts'].mean())

print("Missing values after cleaning:")
print(df.isnull().sum())

# ---------- Step 3: Encode categorical variables ----------
# One-hot encode team, opponent, venue
X_cat = pd.get_dummies(df[['team', 'opponent', 'venue']])

# Numeric features
X_num = df[['avg_team_runs', 'avg_team_wkts']]

# Combine features
X = pd.concat([X_cat, X_num], axis=1)

# Target variable
y = df['match_result']

# ---------- Step 4: Split data into training and testing sets ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Step 5: Train SVM ----------
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

# ---------- Step 6: Evaluate ----------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- Step 7: Save the trained model and features ----------
joblib.dump(model, "svm_match_outcome_model.pkl")
joblib.dump(X.columns.tolist(), "svm_match_outcome_features.pkl")
print("\nModel and feature list saved for Streamlit prediction.")
