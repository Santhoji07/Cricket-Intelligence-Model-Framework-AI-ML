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
# 1. Load data
# ------------------------------
df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv")

req = {'player_name','role','runs','bat_avg','bat_sr','wickets','econ','venue'}
if not req.issubset(df.columns):
    raise ValueError(f"Missing columns: {req - set(df.columns)}")

df.fillna(0, inplace=True)
df['role'] = df['role'].str.lower()

# ------------------------------
# 2. Define role groups
# ------------------------------
def role_group(r):
    if any(x in r for x in ['bowler','spinner']):
        return 'bowler'
    return 'batter'

df['group'] = df['role'].apply(role_group)

# ------------------------------
# 3. Label performance (logical)
# ------------------------------
def perf_batter(row):
    if row['bat_avg'] >= 45 or row['bat_sr'] >= 130:
        return 'High'
    elif row['bat_avg'] >= 30:
        return 'Medium'
    else:
        return 'Low'

def perf_bowler(row):
    if row['wickets'] >= 3 and row['econ'] < 7:
        return 'High'
    elif row['wickets'] >= 1:
        return 'Medium'
    else:
        return 'Low'

df['performance_category'] = df.apply(
    lambda r: perf_bowler(r) if r['group']=='bowler' else perf_batter(r),
    axis=1
)

print(df['performance_category'].value_counts())

# ------------------------------
# 4. Train two separate models
# ------------------------------
models = {}
feature_lists = {}

for grp in ['batter','bowler']:
    sub = df[df['group']==grp].copy()
    if grp=='batter':
        X = sub[['runs','bat_avg','bat_sr']]
    else:
        X = sub[['wickets','econ','runs']]
    X = pd.concat([X, pd.get_dummies(sub[['venue']], drop_first=True)], axis=1)
    y = sub['performance_category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])

    grid = GridSearchCV(
        pipe,
        {'svm__C':[0.5,1,5],'svm__kernel':['rbf','poly'],'svm__gamma':['scale',0.1]},
        cv=4, scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    y_pred = best.predict(X_test)
    print(f"\n=== {grp.upper()} MODEL ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred),3))
    print(classification_report(y_test, y_pred))

    models[grp] = best
    feature_lists[grp] = X.columns.tolist()

# ------------------------------
# 5. Save both models
# ------------------------------
joblib.dump(models['batter'], "svm_batter_model.pkl")
joblib.dump(models['bowler'], "svm_bowler_model.pkl")
joblib.dump(feature_lists, "svm_player_performance_features.pkl")

print("\n💾 Saved separate SVM models for batters and bowlers.")
