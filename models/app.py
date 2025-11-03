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

#SVM Matchup Model
# ---------- Step 3: Squad XI Performance Prediction (SVM Model Integration) ----------
import joblib

if 'best_xi' in st.session_state:
    st.markdown("---")
    st.subheader("📊 Squad XI Performance Prediction (SVM Model)")

    # Button to trigger the prediction
    if st.button("🔮 Generate Squad XI Performance Prediction"):
        try:
            # Load model and features
            perf_model = joblib.load("svm_player_performance_model.pkl")
            perf_features = joblib.load("svm_player_performance_features.pkl")
        except Exception as e:
            st.error(f"❌ Could not load SVM performance model: {e}")
            st.stop()

        # Prepare features from GA Best XI
        best_xi_df = st.session_state.best_xi.copy()
        best_xi_df['venue'] = input_venue  # Add venue context

        # One-hot encode 'role' and 'venue' same as training
        cat_encoded = pd.get_dummies(best_xi_df[['role', 'venue']])
        num_cols = ['runs', 'bat_avg', 'bat_sr', 'wickets', 'econ']
        num_encoded = best_xi_df[num_cols]

        X_pred = pd.concat([cat_encoded, num_encoded], axis=1)

        # Align columns with training features
        for col in perf_features:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[perf_features]

        # Run prediction
        preds = perf_model.predict(X_pred)
        best_xi_df['Predicted_Performance'] = preds

        # Display results
        show_df = best_xi_df[['player_name', 'role', 'matches', 'runs', 'bat_avg', 'bat_sr',
                              'wickets', 'econ', 'Predicted_Performance']]
        show_df = format_floats(sort_by_role(format_roles(show_df)))
        st.dataframe(show_df, use_container_width=True)

        # Summary counts
        perf_summary = show_df['Predicted_Performance'].value_counts().reset_index()
        perf_summary.columns = ['Performance Category', 'Count']
        st.subheader("Performance Summary")
        st.dataframe(perf_summary, use_container_width=True)

        st.success("✅ Player performance predictions generated successfully!")
    else:
        st.info("Click the button above to generate Squad XI performance predictions.")




# ----------  Opponent & Apriori ----------
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




#XBoost Matchup Model

# ---------------------------------------------------------------------
# ---------- Step 4: Dismissal Prediction (XGBoost Model Integration) ----------
# ---------------------------------------------------------------------
import joblib
import numpy as np

if 'best_xi' in st.session_state and 'opponent_xi' in st.session_state:
    st.markdown("---")
    st.subheader("🎯 XGBoost-Based Dismissal Matchup Prediction")

    try:
        # Load trained model and encoders
        xgb_model = joblib.load("xgb_dismissal_model.pkl")
        xgb_encoders = joblib.load("xgb_label_encoders.pkl")
        xgb_features = joblib.load("xgb_model_features.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load XGBoost model files: {e}")
        st.stop()

    # Load ball-by-ball dataset
    stats_df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv")
    if "is_wicket" not in stats_df.columns:
        stats_df["is_wicket"] = (~stats_df["dismissal_type"].isna()).astype(int)

    # ---------------------------------------------
    # Use GA and Apriori outputs directly
    # ---------------------------------------------
    best_xi_df = st.session_state.best_xi.copy()
    opponent_xi = st.session_state.opponent_xi
    venue = input_venue  # from Step 1 input

    # Select phase
    phase_list = sorted(stats_df["phase"].dropna().unique())
    phase = st.selectbox("⚡ Select Match Phase", phase_list)

    # Prediction Button
    if st.button("🔮 Generate XGBoost Dismissal Predictions"):
        insights = []

        # Helper to build features
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

        # Compute predictions
        for batsman in best_xi_df["player_name"]:
            for bowler in opponent_xi:
                feats = build_features(batsman, bowler, venue, phase, stats_df)
                input_df = pd.DataFrame([feats])

                # Encode categorical columns
                for col in ["batsman", "bowler", "venue", "phase"]:
                    if feats[col] in xgb_encoders[col].classes_:
                        input_df[col] = xgb_encoders[col].transform([feats[col]])
                    else:
                        input_df[col] = [0]  # fallback for unseen categories

                # Predict dismissal probability
                prob = xgb_model.predict_proba(input_df[xgb_features])[0][1] * 100
                insights.append((prob, batsman, bowler))

        # Display top results
        if insights:
            insights.sort(key=lambda x: x[0], reverse=True)
            st.subheader("🔥 Top Dismissal Matchups (≥60% Chance)")
            for prob, batsman, bowler in insights[:15]:
                st.markdown(
                    f"**{batsman}** has a **{prob:.1f}%** chance of being dismissed by **{bowler}** "
                    f"at **{venue.title()}** during the **{phase.title()}** phase."
                )
        else:
            st.info("No strong dismissal matchups (≥60%) found for this phase.")
