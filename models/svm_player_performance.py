import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ------------------------------
# 1. Load Data
# ------------------------------
df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv")

req = {'player_name', 'role', 'runs', 'bat_avg', 'bat_sr', 'wickets', 'econ', 'venue'}
if not req.issubset(df.columns):
    raise ValueError(f"Missing columns: {req - set(df.columns)}")

df.fillna(0, inplace=True)
df['role'] = df['role'].str.lower()

# ------------------------------
# 2. Define Role Groups
# ------------------------------
def role_group(r):
    if any(x in r for x in ['bowler', 'spinner']):
        return 'bowler'
    return 'batter'

df['group'] = df['role'].apply(role_group)


# ------------------------------
# 3. Label Performance – Realistic T20 Logic
# ------------------------------
# --- For Batters ---
# T20 key: Strike rate & consistency matter > average
# "Impact Index" = (SR factor * 0.6) + (Average factor * 0.3) + (Consistency * 0.1)
# --- For Batters ---
def perf_batter(row):
    # Main batting impact
    sr_factor = min(row['bat_sr'] / 160, 1.0)
    avg_factor = min(row['bat_avg'] / 40, 1.0)
    runs_factor = min(row['runs'] / 500, 1.0)
    batting_impact = (0.6 * sr_factor) + (0.3 * avg_factor) + (0.1 * runs_factor)

    # Secondary bowling bonus (for all-rounders)
    bowling_bonus = 0
    if row['wickets'] >= 2:
        bowling_bonus += 0.15
    if row['econ'] < 8 and row['econ'] > 0:
        bowling_bonus += 0.1

    final_score = min(batting_impact + bowling_bonus, 1.0)

    if final_score >= 0.8:
        return 'High'
    elif final_score >= 0.55:
        return 'Medium'
    else:
        return 'Low'


# --- For Bowlers ---
def perf_bowler(row):
    # Main bowling impact
    wicket_factor = min(row['wickets'] / 3, 1.0)
    econ_factor = 1 - min(row['econ'] / 10, 1.0)
    runs_factor = 1 - min(row['runs'] / 60, 1.0)
    bowling_impact = (0.5 * wicket_factor) + (0.4 * econ_factor) + (0.1 * runs_factor)

    # Secondary batting bonus (for all-rounders)
    batting_bonus = 0
    if row['bat_avg'] >= 25:
        batting_bonus += 0.1
    if row['bat_sr'] >= 140:
        batting_bonus += 0.1
    if row['runs'] >= 30:
        batting_bonus += 0.1

    final_score = min(bowling_impact + batting_bonus, 1.0)

    if final_score >= 0.8:
        return 'High'
    elif final_score >= 0.55:
        return 'Medium'
    else:
        return 'Low'



# Apply respective logic
df['performance_category'] = df.apply(
    lambda r: perf_bowler(r) if r['group'] == 'bowler' else perf_batter(r),
    axis=1
)

print("📊 Performance category distribution:")
print(df['performance_category'].value_counts())


# ------------------------------
# 4. Train Two Separate Models
# ------------------------------
models = {}
feature_lists = {}

for grp in ['batter', 'bowler']:
    sub = df[df['group'] == grp].copy()

    if grp == 'batter':
        X = sub[['runs', 'bat_avg', 'bat_sr']]
    else:
        X = sub[['wickets', 'econ', 'runs']]

    # One-hot encode venue (adds venue bias)
    X = pd.concat([X, pd.get_dummies(sub[['venue']], drop_first=True)], axis=1)
    y = sub['performance_category']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Define pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])

    # Grid search for hyperparameter tuning
    grid = GridSearchCV(
        pipe,
        {
            'svm__C': [0.5, 1, 3, 5],
            'svm__kernel': ['rbf', 'poly'],
            'svm__gamma': ['scale', 0.1]
        },
        cv=4, scoring='accuracy', n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    # Evaluate model
    y_pred = best.predict(X_test)
    print(f"\n=== {grp.upper()} MODEL ===")
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 3)}")
    print(classification_report(y_test, y_pred))

    models[grp] = best
    feature_lists[grp] = X.columns.tolist()

# ------------------------------
# 5. Save Models
# ------------------------------
joblib.dump(models['batter'], "svm_batter_model.pkl")
joblib.dump(models['bowler'], "svm_bowler_model.pkl")
joblib.dump(feature_lists, "svm_player_performance_features.pkl")

print("\n💾 Saved separate SVM models for batters and bowlers.")

# ------------------------------
# 6. Visualization for Research Paper
# ------------------------------
import matplotlib.pyplot as plt

# --- 1. Performance Category Distribution ---
dist = df['performance_category'].value_counts().sort_index()
plt.figure(figsize=(6,4))
dist.plot(kind='bar', color=['#d62728', '#ff7f0e', '#2ca02c'])
plt.title("Player Performance Category Distribution", fontsize=14, weight='bold')
plt.xlabel("Performance Category")
plt.ylabel("Count of Players")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("svm_output_distribution.png", dpi=300)
plt.show()


# --- 2. Accuracy Comparison (Batter vs Bowler) ---
plt.figure(figsize=(5,4))
plt.bar(['Batter Model', 'Bowler Model'], [0.867, 0.918], color=['#1f77b4','#9467bd'])
plt.title("SVM Model Accuracy Comparison", fontsize=14, weight='bold')
plt.ylabel("Accuracy")
plt.ylim(0,1)
for i, v in enumerate([0.867, 0.918]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10, weight='bold')
plt.tight_layout()
plt.savefig("svm_accuracy_comparison.png", dpi=300)
plt.show()


# --- 3. Class-wise Performance ---
classes = ['High', 'Medium', 'Low']
batter_f1 = [0.83, 0.84, 0.95]   # from your output
bowler_f1 = [0.00, 0.89, 0.96]

x = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(7,5))
plt.bar(x - width/2, batter_f1, width, label='Batter Model', color='#1f77b4')
plt.bar(x + width/2, bowler_f1, width, label='Bowler Model', color='#9467bd')
plt.title("SVM F1-Score by Class", fontsize=14, weight='bold')
plt.ylabel("F1-Score")
plt.xticks(x, classes)
plt.ylim(0,1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("svm_f1_score_comparison.png", dpi=300)
plt.show()

print("\n📊 Visualization images saved as:")
print("1️⃣ svm_output_distribution.png")
print("2️⃣ svm_accuracy_comparison.png")
print("3️⃣ svm_f1_score_comparison.png")

