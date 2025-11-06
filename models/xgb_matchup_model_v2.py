# xgb_matchup_model_v2.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -------------------------
# Config
# -------------------------
INPUT_CSV = "D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv"
MODEL_OUT = "xgb_matchup_model_v2.pkl"
ENC_OUT = "xgb_label_encoders_v2.pkl"
FEATURES_OUT = "xgb_features_v2.pkl"
RANDOM_STATE = 42

print(f"Loading data: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# -------------------------
# Basic cleanup
# -------------------------
for c in ['batsman','bowler','venue','phase']:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

if 'is_wicket' not in df.columns:
    df['is_wicket'] = (~df.get('dismissal_type', pd.Series([np.nan]*len(df))).isna()).astype(int)

df['batsman_l'] = df['batsman'].str.lower()
df['bowler_l'] = df['bowler'].str.lower()
df['venue_l'] = df['venue'].str.lower()
df['phase_l'] = df['phase'].str.lower().replace('', 'middle')

# derive phase if missing
if 'over' not in df.columns:
    df['over'] = 0
def phase_from_over(o):
    try: o=int(o)
    except: return 'middle'
    if o<=6: return 'powerplay'
    elif o<=15: return 'middle'
    else: return 'death'
df['phase_l'] = df['phase_l'].replace('nan','middle')
df.loc[df['phase_l'].isin(['','nan']), 'phase_l'] = df['over'].apply(phase_from_over)

if 'ball_id' not in df.columns:
    df = df.reset_index().rename(columns={'index':'ball_id'})

# -------------------------
# Aggregates
# -------------------------
print("Engineering aggregates...")

# head-to-head
hh = df.groupby(['batsman_l','bowler_l']).agg(
    balls_vs_bowler=('ball_id','count'),
    runs_vs_bowler=('runs_scored','sum'),
    dismissals_vs_bowler=('is_wicket','sum')
).reset_index()
hh['sr_vs_bowler'] = np.where(hh['balls_vs_bowler']>0, hh['runs_vs_bowler']/hh['balls_vs_bowler']*100, 0)

# batsman-venue
bat_v = df.groupby(['batsman_l','venue_l']).agg(
    batsman_balls_venue=('ball_id','count'),
    batsman_runs_venue=('runs_scored','sum'),
    batsman_dismissals_venue=('is_wicket','sum')
).reset_index()
bat_v['batsman_sr_venue'] = np.where(bat_v['batsman_balls_venue']>0, bat_v['batsman_runs_venue']/bat_v['batsman_balls_venue']*100, 0)
bat_v['batsman_avg_venue'] = np.where(bat_v['batsman_dismissals_venue']>0, bat_v['batsman_runs_venue']/bat_v['batsman_dismissals_venue'], bat_v['batsman_runs_venue'])

# bowler-venue
bow_v = df.groupby(['bowler_l','venue_l']).agg(
    bowler_balls_venue=('ball_id','count'),
    bowler_runs_venue=('runs_scored','sum'),
    bowler_wickets_venue=('is_wicket','sum')
).reset_index()
bow_v['bowler_overs_venue'] = np.where(bow_v['bowler_balls_venue']>0, bow_v['bowler_balls_venue']/6.0, 0)
bow_v['bowler_econ_venue'] = np.where(bow_v['bowler_overs_venue']>0, bow_v['bowler_runs_venue']/bow_v['bowler_overs_venue'], 0)
bow_v['bowler_wicket_rate_venue'] = np.where(bow_v['bowler_balls_venue']>0, bow_v['bowler_wickets_venue']/bow_v['bowler_balls_venue'], 0)

# bowler-phase
bow_phase = df.groupby(['bowler_l','phase_l']).agg(
    bp_balls=('ball_id','count'),
    bp_runs=('runs_scored','sum'),
    bp_wickets=('is_wicket','sum')
).reset_index()
bow_phase['bp_wicket_rate'] = np.where(bow_phase['bp_balls']>0, bow_phase['bp_wickets']/bow_phase['bp_balls'], 0)
bow_phase['bp_econ'] = np.where(bow_phase['bp_balls']>0, bow_phase['bp_runs']/(bow_phase['bp_balls']/6.0), 0)

# batsman-phase
bat_phase = df.groupby(['batsman_l','phase_l']).agg(
    batp_balls=('ball_id','count'),
    batp_runs=('runs_scored','sum'),
    batp_dismissals=('is_wicket','sum')
).reset_index()
bat_phase['batp_runs_per_ball'] = np.where(bat_phase['batp_balls']>0, bat_phase['batp_runs']/bat_phase['batp_balls'], 0)

# -------------------------
# Build pair dataset
# -------------------------
print("Building pair-level dataset...")

pairs = hh.copy()

venue_map = (
    df.groupby(['batsman_l','bowler_l'])['venue_l']
    .agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) else 'unknown')
    .reset_index()
)
phase_map = (
    df.groupby(['batsman_l','bowler_l'])['phase_l']
    .agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) else 'middle')
    .reset_index()
)

pairs = pairs.merge(venue_map, on=['batsman_l','bowler_l'], how='left')
pairs = pairs.merge(phase_map, on=['batsman_l','bowler_l'], how='left')

pairs = pairs.merge(bat_v, how='left', on=['batsman_l','venue_l'])
pairs = pairs.merge(bow_v, how='left', on=['bowler_l','venue_l'])
pairs = pairs.merge(bow_phase, how='left', on=['bowler_l','phase_l'])
pairs = pairs.merge(bat_phase, how='left', on=['batsman_l','phase_l'])

pairs = pairs.rename(columns={
    'balls_vs_bowler':'balls_between',
    'runs_vs_bowler':'runs_between',
    'dismissals_vs_bowler':'dismissals_between'
})

numeric_cols = pairs.select_dtypes(include=[np.number]).columns
pairs[numeric_cols] = pairs[numeric_cols].fillna(0)

pairs['dismissal_likelihood'] = np.where(
    pairs['balls_between']>0,
    pairs['dismissals_between']/pairs['balls_between'],
    0.0
)
pairs = pairs[pairs['balls_between']>=2].reset_index(drop=True)

# -------------------------
# Prepare features
# -------------------------
feature_columns = [
    'batsman_l','bowler_l','venue_l','phase_l',
    'batsman_balls_venue','batsman_runs_venue','batsman_sr_venue','batsman_avg_venue',
    'bowler_balls_venue','bowler_runs_venue','bowler_wickets_venue','bowler_econ_venue','bowler_wicket_rate_venue',
    'bp_wicket_rate','bp_econ','batp_runs_per_ball',
    'balls_between','runs_between','sr_vs_bowler','dismissals_between'
]

for c in feature_columns:
    if c not in pairs.columns:
        pairs[c] = 0

X = pairs[feature_columns].copy()
y = pairs['dismissal_likelihood']

categorical_cols = ['batsman_l','bowler_l','venue_l','phase_l']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = X[col].astype(str).fillna('unknown')
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

final_features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# -------------------------
# Train model
# -------------------------
print("Training XGBoost Regressor...")
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.04,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    tree_method='hist'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"✅ Evaluation -> RMSE: {rmse:.5f}, R²: {r2:.5f}")

# -------------------------
# Save
# -------------------------
joblib.dump(model, MODEL_OUT)
joblib.dump(encoders, ENC_OUT)
joblib.dump(final_features, FEATURES_OUT)
print(f"💾 Saved model, encoders, features to:\n{MODEL_OUT}\n{ENC_OUT}\n{FEATURES_OUT}")
print("Done.")
