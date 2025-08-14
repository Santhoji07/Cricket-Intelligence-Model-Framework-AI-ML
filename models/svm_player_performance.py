# svm_player_performance.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ----- Step 1: Load data -----
df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv")

# Ensure required columns exist
required_cols = {'player_name', 'role', 'runs', 'bat_avg', 'bat_sr', 'wickets', 'econ', 'venue'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must have columns: {required_cols}")

# ----- Step 2: Create Target 'performance_category' -----
# Example: Use batting avg + wickets as criteria
def categorize_performance(row):
    if row['bat_avg'] >= 40 or row['wickets'] >= 3:
        return "High"
    elif row['bat_avg'] >= 25 or row['wickets'] >= 1:
        return "Medium"
    else:
        return "Low"

df['performance_category'] = df.apply(categorize_performance, axis=1)

print("Class counts:")
print(df['performance_category'].value_counts())

# ----- Step 3: Prepare Features -----
# One-hot encode role and venue
cat_features = pd.get_dummies(df[['role', 'venue']])

# Numeric features
num_features = df[['runs', 'bat_avg', 'bat_sr', 'wickets', 'econ']]

# Combine features
X = pd.concat([cat_features, num_features], axis=1)
y = df['performance_category']

# ----- Step 4: Split -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----- Step 5: Train SVM -----
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

# ----- Step 6: Evaluate -----
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ----- Step 7: Save model & features -----
joblib.dump(model, "svm_player_performance_model.pkl")
joblib.dump(X.columns.tolist(), "svm_player_performance_features.pkl")
print("\n✅ Player performance model saved!")
