import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load ball-by-ball data
df = pd.read_csv('D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv')

# Prepare 'is_wicket' target column from dismissal_type
if 'is_wicket' not in df.columns:
    df['is_wicket'] = (~df['dismissal_type'].isna()).astype(int)

# Parse dates & sort for rolling calculations
df['match_date'] = pd.to_datetime(df['date'])
df = df.sort_values(['batsman', 'match_date'])

# Aggregate per match, batsman, bowler, venue, phase — average wicket rate
agg = df.groupby(['match_id', 'batsman', 'bowler', 'venue', 'phase']).agg({
    'is_wicket': 'max'  # 1 if dismissed in that match/phase/venue/bowler, else 0
}).reset_index()

# Recent form of batsman (avg runs last 3 innings excluding current)
df['recent_form'] = df.groupby('batsman')['runs_scored']\
                    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
recent = df.groupby(['match_id', 'batsman'])['recent_form'].mean().reset_index()
agg = agg.merge(recent, on=['match_id', 'batsman'], how='left')

# Bowler wickets last 5 balls at venue (exclude current)
df['bowler_wickets_venue'] = df.groupby(['bowler', 'venue'])['is_wicket']\
                              .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
bowler_last5 = df.groupby(['match_id', 'bowler', 'venue'])['bowler_wickets_venue'].mean().reset_index()
agg = agg.merge(bowler_last5, on=['match_id', 'bowler', 'venue'], how='left')

# Features and target
cat_cols = ['batsman', 'bowler', 'venue', 'phase']
num_cols = ['recent_form', 'bowler_wickets_venue']

X = agg[cat_cols + num_cols].fillna(0)
y = agg['is_wicket']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Preprocessing + SVM pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

svm_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('svc', SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=42))
])

svm_pipeline.fit(X_train, y_train)

# Evaluate
acc = svm_pipeline.score(X_test, y_test)
print(f"Validation Accuracy (dismissal prediction): {acc:.3f}")

# Save model & feature list
joblib.dump(svm_pipeline, 'svm_dismissal_model.pkl')
joblib.dump(cat_cols + num_cols, 'svm_dismissal_features.pkl')
