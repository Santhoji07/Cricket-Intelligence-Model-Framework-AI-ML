import streamlit as st
import pandas as pd
from ga_team_selector import CricketTeamGA
from ap_opponent_analysis import run_apriori_matchups

# ---------------------------------------------------------------------
# File paths
STATS_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv"
ROLES_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv"
BALL_BY_BALL_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv"

ROLE_DISPLAY = {
    'opener': 'Opener',
    'middle_order': 'Middle order',
    'wicket_keeper': 'Wicket-Keeper',
    'finisher': 'Finisher',
    'spinner': 'Spinner',
    'fast_bowler': 'Fast Bowler'
}
ROLE_ORDER = ['Opener', 'Middle order', 'Wicket-Keeper', 'Finisher', 'Spinner', 'Fast Bowler']

# ---------- Utility formatting functions ----------
def format_roles(df):
    df = df.copy()
    if 'role' in df.columns:
        df['role'] = df['role'].map(ROLE_DISPLAY).fillna(df['role'])
    return df

def sort_by_role(df):
    df = df.copy()
    if 'role' in df.columns:
        df['role_order_index'] = df['role'].map(lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 99)
        df = df.sort_values(by='role_order_index').drop(columns=['role_order_index'])
    return df

def format_floats(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=['float', 'float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].apply(
            lambda x: int(x) if pd.notnull(x) and float(x).is_integer()
            else round(x, 2) if pd.notnull(x)
            else x
        )
    return df

def clean_table(df):
    cols_to_drop = [c for c in df.columns if 'lower' in c or 'venue' in c or c == 'franchise']
    return df.drop(columns=cols_to_drop, errors='ignore')

def show_table(df, start_index=1):
    df = df.copy()
    df.index = range(start_index, start_index + len(df))
    st.dataframe(df)

# ---------------------------------------------------------------------
st.title("Cricket Intelligence Model - Best XI Selector & Apriori Matchup Analysis")

df_roles_raw = pd.read_csv(ROLES_FILE)
df_stats_raw = pd.read_csv(STATS_FILE)

franchise_display = sorted(df_roles_raw['franchise'].dropna().unique())
venue_display = sorted(df_stats_raw['venue'].dropna().unique())

input_team_display = st.selectbox("Select Franchise", franchise_display)
input_venue_display = st.selectbox("Select Venue", venue_display)

input_team = input_team_display.strip().lower()
input_venue = input_venue_display.strip().lower()

# ---------- Step 1: Best XI Selection ----------
if st.button("Select Best XI"):
    try:
        ga_model = CricketTeamGA(STATS_FILE, ROLES_FILE)
        best_team = ga_model.run_ga(input_team, input_venue)

        st.session_state.best_xi = best_team

        st.subheader(f"{input_team_display} Squad List")
        squad_df = clean_table(ga_model.franchise_list.copy())
        squad_df = format_floats(sort_by_role(format_roles(squad_df)))
        show_table(squad_df, start_index=1)

        st.subheader(f"Player Pool Used (min_matches ≥ {ga_model.min_matches_used})")
        pool_df = clean_table(ga_model.player_pool.copy())
        pool_df = format_floats(sort_by_role(format_roles(pool_df)))
        show_table(pool_df, start_index=1)

        st.subheader("Role Counts")
        role_counts_data = [
            {"Role": ROLE_DISPLAY[r], "Count": int(ga_model.role_counts.get(r, 0))}
            for r in ['opener','middle_order','wicket_keeper','finisher','spinner','fast_bowler']
        ]
        role_counts_df = pd.DataFrame(role_counts_data)
        show_table(role_counts_df, start_index=1)

        best_team_df = clean_table(best_team.copy())
        best_team_df = format_floats(sort_by_role(format_roles(best_team_df)))
        st.subheader(f'Selected Best Playing XI for {input_venue_display}')
        show_table(best_team_df, start_index=1)

        leftover_df = ga_model.player_pool[~ga_model.player_pool['player_name'].isin(best_team['player_name'])].copy()
        if not leftover_df.empty:
            leftover_df = leftover_df.groupby('player_name', as_index=False).agg({
                'role': 'first',
                'matches': 'sum',
                'runs': 'sum',
                'bat_avg': 'mean',
                'bat_sr': 'mean',
                'wickets': 'sum',
                'econ': 'mean',
                'indian': 'first'
            })
        leftover_df = clean_table(leftover_df)
        leftover_df = format_floats(sort_by_role(format_roles(leftover_df)))
        st.subheader("Players Left Out from Player Pool")
        show_table(leftover_df, start_index=len(best_team_df)+1)

        st.success(f"Total Fitness Score: {ga_model.fitness(best_team):.2f}")

        if "fallback" in str(ga_model.min_matches_used).lower():
            st.info("Not enough venue stats, showing best squad XI (aggregated stats) instead.")

    except Exception as e:
        st.error(str(e))

# ---------- Step 2: Opponent & Apriori ----------
if 'best_xi' in st.session_state:
    st.markdown("---")
    st.subheader("Opponent Setup (for Apriori Analysis)")

    opponent_team = st.selectbox("Select Opponent Franchise", franchise_display, key="opponent_team")
    opponent_squad = df_roles_raw[df_roles_raw['franchise'] == opponent_team]['player_name'].tolist()
    opponent_xi = st.multiselect("Select Opponent Playing XI", opponent_squad, key="opponent_xi")

    if st.button("Run Apriori Matchup Analysis"):
        if not opponent_xi or len(opponent_xi) != 11:
            st.warning("Please select exactly 11 players for opponent XI.")
        else:
            ball_df = pd.read_csv(BALL_BY_BALL_FILE)
            try:
                results_df = run_apriori_matchups(
                    my_xi=st.session_state.best_xi,
                    opponent_xi=opponent_xi,
                    ball_df=ball_df
                )

                if results_df.empty:
                    st.warning("No strong historical patterns found for these matchups.")
                else:
                    # Parse columns with original case from ball_df
                    def extract_parts_with_case(row):
                        ant = [s.strip() for s in row['antecedents'].split(',')]
                        cons = [s.strip() for s in row['consequents'].split(',')]

                        bowler = next((x.replace('bowler:', '').strip() for x in ant if x.startswith('bowler:')), '')
                        batsman = next((x.replace('batsman:', '').strip() for x in cons if x.startswith('batsman:')), '')
                        venue = next((x.replace('venue:', '').strip() for x in ant if x.startswith('venue:')), '')
                        phase = next((x.replace('phase:', '').strip() for x in ant if x.startswith('phase:')), '')
                        dismissal_type = next((x.replace('dismissal:', '').strip() for x in ant if x.startswith('dismissal:')), '')

                        # Restore case from ball_df
                        bowler_case = ball_df.loc[ball_df['bowler'].str.lower() == bowler.lower(), 'bowler'].head(1)
                        batsman_case = ball_df.loc[ball_df['batsman'].str.lower() == batsman.lower(), 'batsman'].head(1)
                        venue_case = ball_df.loc[ball_df['venue'].str.lower() == venue.lower(), 'venue'].head(1)

                        bowler = bowler_case.iloc[0] if not bowler_case.empty else bowler.title()
                        batsman = batsman_case.iloc[0] if not batsman_case.empty else batsman.title()
                        venue = venue_case.iloc[0] if not venue_case.empty else venue.title()
                        phase = phase.capitalize() if phase else ''

                        return pd.Series({
                            'Bowler': bowler, 
                            'Batsman': batsman,
                            'Venue': venue,
                            'Phase': phase,
                            'Dismissal': dismissal_type,
                            'Support': row['support'],
                            'Confidence': row['confidence'],
                            'Lift': row['lift']
                        })

                    pretty_df = results_df.apply(extract_parts_with_case, axis=1)

                    # Deduplicate: keep highest Lift per Bowler-Batsman
                    deduped = (
                        pretty_df
                        .sort_values(['Bowler', 'Batsman', 'Lift', 'Confidence'], ascending=[True, True, False, False])
                        .drop_duplicates(['Bowler', 'Batsman'])
                    )

                    # Sort and take top N
                    top_n = 20
                    top_results = deduped.sort_values(['Lift', 'Confidence', 'Support'], ascending=False).head(top_n)

                    # Remove old index & add Sl.No
                    top_results = top_results.reset_index(drop=True)
                    top_results['Sl.No'] = range(1, len(top_results) + 1)

                    # Reorder columns so Sl.No is first
                    cols = ['Sl.No'] + [c for c in top_results.columns if c != 'Sl.No']
                    top_results = top_results[cols]

                    # Show without the index
                    st.subheader("🔝 Best Unique Bowler-Batsman Matchups")
                    st.dataframe(top_results.set_index('Sl.No'))
                    # This will show Sl.No as first column, no default index


                    # Tactical Summary (plain sentences without numbers/metrics)
                    summary_lines = []
                    for _, row in top_results.iterrows():
                        # Decide the context text
                        if row.Phase and row.Venue:
                            context = f"in the {row.Phase} overs at {row.Venue}"
                        elif row.Phase and not row.Venue:
                            context = f"in the {row.Phase} overs"
                        elif not row.Phase and row.Venue:
                            context = f"at {row.Venue}"
                        else:
                            context = ""

                        # Build final sentence
                        if context:
                            sentence = f"{row.Bowler} bowling {context} has a high historical success against {row.Batsman}."
                        else:
                            sentence = f"{row.Bowler} bowling has a high historical success against {row.Batsman}."

                        summary_lines.append(sentence)
                        

                    st.subheader("📝 Tactical Insights")
                    for i, sentence in enumerate(summary_lines, start=1):
                        st.write(f"{i}. {sentence}")

            except Exception as e:
                st.error(f"Apriori error: {e}")

# ---------- Step 3: SVM Match Outcome Prediction ----------
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Cricket Match Outcome Predictor", layout="centered")

# ---------- Load saved model ----------
try:
    svm_model = joblib.load("svm_match_outcome_model.pkl")
    feature_cols = joblib.load("svm_match_outcome_features.pkl")
except:
    st.error("⚠️ Model not found. Please run svm_match_outcome.py first.")
    st.stop()

# ---------- Load datasets ----------
player_roles_df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv")
player_stats_df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv")

# Remove NaN and ensure all are strings
team_list = sorted(
    player_roles_df['franchise']
    .dropna()              # remove missing values
    .astype(str)           # ensure string type
    .unique()
)

venue_list = sorted(player_stats_df['venue'].unique())

st.title("🏏 Match Outcome Prediction (SVM)")

# ---------- Select teams and venue ----------
team_a = st.selectbox("Select Your Team (GA Best XI)", team_list)
team_b = st.selectbox("Select Opponent Team (Apriori Insights)", team_list)
venue = st.selectbox("Select Match Venue", venue_list)

# ---------- Calculate average stats ----------
def calculate_team_averages(team_name):
    players = player_roles_df[player_roles_df['franchise'] == team_name]['player_name']
    stats = player_stats_df[player_stats_df['player_name'].isin(players)]
    avg_runs = stats['runs'].mean() if not stats.empty else 0
    avg_wkts = stats['wickets'].mean() if not stats.empty else 0
    return round(avg_runs, 2), round(avg_wkts, 2)

avg_runs, avg_wkts = calculate_team_averages(team_a)

# Allow manual override
avg_team_runs = st.number_input("Average Team Runs (adjust if needed)", value=avg_runs, step=0.1)
avg_team_wkts = st.number_input("Average Team Wickets (adjust if needed)", value=avg_wkts, step=0.1)

# ---------- Predict ----------
if st.button("🔮 Predict Outcome"):
    # Create input dataframe with all 0s
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Fill in numeric features
    if 'avg_team_runs' in input_df.columns:
        input_df['avg_team_runs'] = avg_team_runs
    if 'avg_team_wkts' in input_df.columns:
        input_df['avg_team_wkts'] = avg_team_wkts

    # Set one-hot encoded categorical features
    t_col = f"team_{team_a}"
    o_col = f"opponent_{team_b}"
    v_col = f"venue_{venue}"

    if t_col in input_df.columns: input_df[t_col] = 1
    if o_col in input_df.columns: input_df[o_col] = 1
    if v_col in input_df.columns: input_df[v_col] = 1

    # Predict
    pred = svm_model.predict(input_df)[0]
    proba = svm_model.predict_proba(input_df)[0]
    confidence = max(proba) * 100

    st.success(f"Predicted Result: **{pred.upper()}**")
    st.info(f"Prediction Confidence: {confidence:.2f}%")

# --- Squad XI Batch Prediction (Venue Specific) ---
st.markdown("---")
st.subheader("🧑‍🤝‍🧑 Squad XI Player Performance Category Prediction (Venue Specific)")

# Load data
player_stats_df = pd.read_csv(
    "D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv"
)
df_roles_raw = pd.read_csv(  # Roles dataset (franchise -> player mapping)
    "D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv"
)

# Load player performance model & features
try:
    pp_model = joblib.load("svm_player_performance_model.pkl")
    pp_features = joblib.load("svm_player_performance_features.pkl")
except Exception as e:
    st.error(f"⚠️ Player performance model or features not found: {e}")
    st.stop()

# 1️⃣ Select Squad/Franchise
franchise_display = sorted(df_roles_raw['franchise'].dropna().unique())
selected_squad = st.selectbox(
    "Select Your Squad / Franchise",
    franchise_display,
    key="squad_xi_franchise"
)

# 2️⃣ Select Players only from that squad
squad_players = df_roles_raw[df_roles_raw['franchise'] == selected_squad]['player_name'].dropna().unique().tolist()
selected_players = st.multiselect(
    f"Select Playing XI from {selected_squad} (exactly 11 players)",
    sorted(squad_players),
    key="squad_xi_players",
    max_selections=11
)

# 3️⃣ Select Venue
venues_list = sorted(player_stats_df['venue'].dropna().unique())
selected_venue = st.selectbox(
    "Select Venue",
    venues_list,
    key="squad_xi_venue"
)

# 4️⃣ Predict Button
if st.button("🟢 Predict Squad XI Performance", key="squad_xi_button"):
    if len(selected_players) != 11:
        st.error("Please select exactly 11 players.")
    else:
        results = []
        missing_from_venue = []

        for player in selected_players:
            # Filter stats for player at this venue
            p_row = player_stats_df[
                (player_stats_df['player_name'] == player) &
                (player_stats_df['venue'] == selected_venue)
            ]

            if not p_row.empty:
                role = p_row.iloc[0]['role']
                runs = p_row.iloc[0]['runs']
                bat_avg = p_row.iloc[0]['bat_avg']
                bat_sr = p_row.iloc[0]['bat_sr']
                wickets = p_row.iloc[0]['wickets']
                econ = p_row.iloc[0]['econ']

                # Build input data for model
                input_df = pd.DataFrame(0, index=[0], columns=pp_features)
                input_df['runs'] = runs if not pd.isna(runs) else 0
                input_df['bat_avg'] = bat_avg if not pd.isna(bat_avg) else 0
                input_df['bat_sr'] = bat_sr if not pd.isna(bat_sr) else 0
                input_df['wickets'] = wickets if not pd.isna(wickets) else 0
                input_df['econ'] = econ if not pd.isna(econ) else 0

                role_col = f"role_{role}"
                venue_col = f"venue_{selected_venue}"
                if role_col in input_df.columns:
                    input_df[role_col] = 1
                if venue_col in input_df.columns:
                    input_df[venue_col] = 1

                # Make prediction
                pred = pp_model.predict(input_df)[0]
                proba = pp_model.predict_proba(input_df)[0]
                confidence = max(proba) * 100

                results.append({
                    'Squad': selected_squad,
                    'Player': player,
                    'Role': role,
                    'Venue': selected_venue,
                    'Predicted Category': pred,
                    'Confidence (%)': f"{confidence:.2f}"
                })
            else:
                missing_from_venue.append(player)

        if results:
            st.write("### Squad XI Prediction Results")
            st.dataframe(pd.DataFrame(results))

            # Summary of categories
            st.write("### Category Summary")
            st.table(pd.DataFrame(results)['Predicted Category'].value_counts().reset_index().rename(columns={'index':'Category','Predicted Category':'Count'}))

        if missing_from_venue:
            st.warning(f"No venue stats found for: {', '.join(missing_from_venue)} at {selected_venue}")

#SVM Matchup Model
# SVM Matchup Model - Corrected Batting vs Bowling Logic
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------
# Load trained SVM model and feature list
# ---------------------------------------------
try:
    svm_model = joblib.load('svm_dismissal_model.pkl')
    model_features = joblib.load('svm_dismissal_features.pkl')
except Exception as e:
    st.error(f"❌ Failed to load SVM model or features: {e}")
    st.stop()

# ---------------------------------------------
# Load data
# ---------------------------------------------
stats_df = pd.read_csv('D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv')
player_roles = pd.read_csv('player_roles_cleaned.csv')

if 'is_wicket' not in stats_df.columns:
    stats_df['is_wicket'] = (~stats_df['dismissal_type'].isna()).astype(int)

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.title("🏏 SVM-Based Dismissal Matchup Predictor")

franchises = sorted(player_roles['franchise'].dropna().astype(str).unique())

batting_team = st.selectbox("🏏 Select Batting Team", franchises)
bowling_team = st.selectbox("🎯 Select Bowling Team", franchises)

batting_players = sorted(
    player_roles[player_roles['franchise'] == batting_team]['player_name'].unique()
)
bowling_players = sorted(
    player_roles[player_roles['franchise'] == bowling_team]['player_name'].unique()
)

batting_xi = st.multiselect("Select Batting XI (max 11)", batting_players, max_selections=11)
bowling_xi = st.multiselect("Select Bowling XI (max 11)", bowling_players, max_selections=11)

venue_list = sorted(stats_df['venue'].dropna().unique())
phase_list = sorted(stats_df['phase'].dropna().unique())

venue = st.selectbox("🏟️ Select Venue", venue_list)
phase = st.selectbox("⚡ Select Phase", phase_list)

# ---------------------------------------------
# Helper function to build features for model
# ---------------------------------------------
def build_features(batsman, bowler, venue, phase, stats):
    # Recent batting form (avg of last 3 innings)
    bat_matches = stats[stats['batsman'] == batsman].sort_values('date')
    recent_scores = bat_matches.groupby('match_id')['runs_scored'].sum().shift(1).dropna().tail(3)
    recent_avg = recent_scores.mean() if len(recent_scores) > 0 else 0

    # Bowler’s recent wicket trend at this venue (last 5 balls)
    bowler_venue = stats[(stats['bowler'] == bowler) & (stats['venue'] == venue)].sort_values('date')
    last5_wkts = bowler_venue['is_wicket'].shift(1).dropna().tail(5).sum() if len(bowler_venue) > 0 else 0

    return {
        'batsman': batsman,
        'bowler': bowler,
        'venue': venue,
        'phase': phase if phase else "",
        'recent_form': recent_avg,
        'bowler_wickets_venue': last5_wkts
    }

# ---------------------------------------------
# Prediction Logic
# ---------------------------------------------
if st.button("🔮 Predict Dismissal Likelihood & Advantage"):
    if not batting_xi or not bowling_xi:
        st.warning("⚠️ Please select at least one player from both batting and bowling teams.")
    else:
        insights = []

        # ✅ Only batsmen (batting XI) vs bowlers (bowling XI)
        for batsman in batting_xi:
            for bowler in bowling_xi:
                features = build_features(batsman, bowler, venue, phase, stats_df)
                model_df = pd.DataFrame([features], columns=model_features).fillna(0)

                try:
                    prob = svm_model.predict_proba(model_df)[0][1] * 100
                except Exception:
                    prob = np.nan

                if np.isnan(prob):
                    continue

                # Only show when model believes dismissal is likely (≥70%)
                if prob >= 0:
                    insight = (
                        f"**{batsman}** has a **{prob:.1f}%** chance of being dismissed "
                        f"by **{bowler}** at **{venue}** in **{phase}**.<br>"
                        f"🎯 Advantage: **{bowler}**"
                    )
                    insights.append((prob, insight))

        # ---------------------------------------------
        # Display Top Matchups
        # ---------------------------------------------
        if insights:
            insights.sort(key=lambda x: x[0], reverse=True)
            st.markdown("### 🔥 Top High-Risk Dismissal Matchups (≥40%)")
            for _, text in insights[:10]:
                st.markdown(text, unsafe_allow_html=True)
        else:
            st.info("😎 No strong (≥40%) dismissal matchups found for these teams.")


#XBoost Matchup Model

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------
# Load Model & Data
# --------------------------------
try:
    model = joblib.load("xgb_dismissal_model.pkl")
    encoders = joblib.load("xgb_label_encoders.pkl")
    features = joblib.load("xgb_model_features.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

stats_df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv")
roles_df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv")

# --------------------------------
# Data Preparation
# --------------------------------
if "is_wicket" not in stats_df.columns:
    stats_df["is_wicket"] = (~stats_df["dismissal_type"].isna()).astype(int)

# --------------------------------
# Streamlit UI
# --------------------------------
st.title("🏏 Dismissal Prediction Model (XGBoost)")
st.markdown("### Predict the chance of a **batsman getting dismissed by a bowler** under given conditions.")

franchises = sorted(roles_df["franchise"].dropna().unique())

batting_team = st.selectbox("Select Batting Team", franchises)
bowling_team = st.selectbox("Select Bowling Team", franchises)

batters = sorted(roles_df[roles_df["franchise"] == batting_team]["player_name"].unique())
bowlers = sorted(roles_df[roles_df["franchise"] == bowling_team]["player_name"].unique())

batting_xi = st.multiselect("Select Batting XI", batters, max_selections=11)
bowling_xi = st.multiselect("Select Bowling XI", bowlers, max_selections=11)

venue_list = sorted(stats_df["venue"].dropna().unique())
phase_list = sorted(stats_df["phase"].dropna().unique())

venue = st.selectbox("Select Venue", venue_list)
phase = st.selectbox("Select Phase", phase_list)

# --------------------------------
# Build Input Features
# --------------------------------
def build_features(batsman, bowler, venue, phase, stats):
    recent_bat_form = stats[stats["batsman"] == batsman]["runs_scored"].tail(3).mean()
    bowler_form = stats[stats["bowler"] == bowler]["is_wicket"].tail(10).sum()

    return {
        "batsman": batsman,
        "bowler": bowler,
        "venue": venue,
        "phase": phase,
        "recent_bat_form": recent_bat_form if not np.isnan(recent_bat_form) else 0,
        "bowler_form": bowler_form if not np.isnan(bowler_form) else 0
    }

# --------------------------------
# Prediction
# --------------------------------
if st.button("🎯 Predict Dismissal Matchups"):
    if not batting_xi or not bowling_xi:
        st.warning("Please select both batting and bowling XI.")
    else:
        insights = []
        for batsman in batting_xi:
            for bowler in bowling_xi:
                feats = build_features(batsman, bowler, venue, phase, stats_df)
                input_df = pd.DataFrame([feats])

                # Encode categoricals
                for col in ["batsman", "bowler", "venue", "phase"]:
                    if feats[col] in encoders[col].classes_:
                        input_df[col] = encoders[col].transform([feats[col]])
                    else:
                        input_df[col] = [0]  # fallback for unseen

                prob = model.predict_proba(input_df[features])[0][1] * 100

                if prob >= 70:  # only show strong dismissal chances
                    insights.append((prob, batsman, bowler))

        if insights:
            insights.sort(key=lambda x: x[0], reverse=True)
            st.subheader("🔥 Top Dismissal Matchups (>70% Chance)")
            for prob, batsman, bowler in insights[:10]:
                st.markdown(f"**{batsman}** has a **{prob:.1f}%** chance of being dismissed by **{bowler}** at **{venue}** in **{phase}**.")
        else:
            st.info("No high dismissal matchups (>70%) found for this selection.")

