# svm_match_outcome.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ---------- Step 1: Load dataset ----------
df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/match_results.csv")

# Ensure required columns exist
required_cols = {'team', 'opponent', 'venue', 'avg_team_runs', 'avg_team_wkts', 'match_result'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must have columns: {required_cols}")

# ---------- Step 2: Data Cleaning ----------
print("Missing values before cleaning:")
print(df.isnull().sum())

# Drop rows with missing categorical or target fields
df = df.dropna(subset=['team', 'opponent', 'venue', 'match_result'])

# Fill numeric NaNs with mean
df['avg_team_runs'] = df['avg_team_runs'].fillna(df['avg_team_runs'].mean())
df['avg_team_wkts'] = df['avg_team_wkts'].fillna(df['avg_team_wkts'].mean())

# Remove classes with less than 2 samples (stratify needs >= 2 per class)
class_counts = df['match_result'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df['match_result'].isin(valid_classes)]

print("Class distribution after cleaning:")
print(df['match_result'].value_counts())

# ---------- Step 3: Feature Encoding ----------
X_cat = pd.get_dummies(df[['team', 'opponent', 'venue']])
X_num = df[['avg_team_runs', 'avg_team_wkts']]
X = pd.concat([X_cat, X_num], axis=1)
y = df['match_result']

# ---------- Step 4: Split train/test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Step 5: Train SVM ----------
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

# ---------- Step 6: Evaluation ----------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- Step 7: Save model & features ----------
joblib.dump(model, "svm_match_outcome_model.pkl")
joblib.dump(X.columns.tolist(), "svm_match_outcome_features.pkl")
print("\n✅ Model and feature list saved!")

