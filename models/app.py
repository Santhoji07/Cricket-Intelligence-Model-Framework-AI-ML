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
st.session_state["input_venue"] = input_venue_display

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

    if st.button("🔮 Generate Squad XI Performance Prediction"):
        try:
            # Load the two role-specific SVM models
            batter_model = joblib.load("svm_batter_model.pkl")
            bowler_model = joblib.load("svm_bowler_model.pkl")
            feature_dict = joblib.load("svm_player_performance_features.pkl")
        except Exception as e:
            st.error(f"❌ Could not load SVM models: {e}")
            st.stop()

        # Get GA best XI
        best_xi_df = st.session_state.best_xi.copy()
        best_xi_df['venue'] = input_venue
        best_xi_df['role'] = best_xi_df['role'].str.lower()

        # Helper: classify into batter or bowler
        def classify_group(r):
            if any(x in r for x in ['bowler', 'spinner']):
                return 'bowler'
            return 'batter'

        best_xi_df['group'] = best_xi_df['role'].apply(classify_group)

        preds_all = []

        # --- Batter predictions ---
        batters = best_xi_df[best_xi_df['group'] == 'batter'].copy()
        if not batters.empty:
            Xb = batters[['runs', 'bat_avg', 'bat_sr']].copy()
            Xb = pd.concat([Xb, pd.get_dummies(batters[['venue']], drop_first=True)], axis=1)

            # Align columns
            batter_feats = feature_dict['batter']
            for col in batter_feats:
                if col not in Xb.columns:
                    Xb[col] = 0
            Xb = Xb[batter_feats]

            batters['Predicted_Performance'] = batter_model.predict(Xb)
            preds_all.append(batters)

        # --- Bowler predictions ---
        bowlers = best_xi_df[best_xi_df['group'] == 'bowler'].copy()
        if not bowlers.empty:
            Xw = bowlers[['wickets', 'econ', 'runs']].copy()
            Xw = pd.concat([Xw, pd.get_dummies(bowlers[['venue']], drop_first=True)], axis=1)

            bowler_feats = feature_dict['bowler']
            for col in bowler_feats:
                if col not in Xw.columns:
                    Xw[col] = 0
            Xw = Xw[bowler_feats]

            bowlers['Predicted_Performance'] = bowler_model.predict(Xw)
            preds_all.append(bowlers)

        # Merge back
        final_df = pd.concat(preds_all).sort_index()

        # Display table
        show_df = final_df[['player_name', 'role', 'matches', 'runs', 'bat_avg', 'bat_sr',
                            'wickets', 'econ', 'Predicted_Performance']]
        show_df = format_floats(sort_by_role(format_roles(show_df)))
        st.dataframe(show_df, use_container_width=True)

        # Summary counts
        summary = show_df['Predicted_Performance'].value_counts().reset_index()
        summary.columns = ['Performance Category', 'Count']
        st.subheader("Performance Summary")
        st.dataframe(summary, use_container_width=True)

        st.success("✅ Squad XI predictions generated successfully!")

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
# ---------- Compact XGBoost Matchup UI ----------
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Config — update paths if your files are elsewhere
MODEL_PATH = "xgb_delivery_model.pkl"
ENC_PATH = "xgb_delivery_label_encoders.pkl"
FEAT_PATH = "xgb_delivery_features.pkl"
ROLES_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv"
BALLS_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv"

# Check session state (must have GA + Apriori run)
if not ('best_xi' in st.session_state and 'opponent_xi' in st.session_state and 'input_venue' in st.session_state):
    st.info("Run the GA Best XI selector and opponent Apriori setup first (so venue, Best XI and Opponent XI are available).")
else:
    st.markdown("---")
    st.subheader("🎯 Compact Dismissal Match-up Viewer (XGBoost)")

    # Load model artifacts (one-time)
    try:
        xgb_model = joblib.load(MODEL_PATH)
        xgb_encoders = joblib.load(ENC_PATH)
        xgb_feature_cols = joblib.load(FEAT_PATH)
    except Exception as e:
        st.error(f"Could not load model artifacts: {e}")
        st.stop()

    # Load supporting data
    stats_df = pd.read_csv(BALLS_FILE)
    if "is_wicket" not in stats_df.columns:
        stats_df["is_wicket"] = (~stats_df.get("dismissal_type", pd.Series([np.nan]*len(stats_df))).isna()).astype(int)

    # roles file (to order and classify players)
    try:
        roles_df = pd.read_csv(ROLES_FILE)
        roles_df['player_name'] = roles_df['player_name'].astype(str).str.strip()
        roles_map = dict(zip(roles_df['player_name'].str.lower(), roles_df['role'].str.lower()))
    except Exception:
        roles_map = {}  # fallback

    # Convenient role display & ordering
    ROLE_ORDER = ['opener','middle_order','wicket_keeper','finisher','spinner','fast_bowler']
    ROLE_LABEL = {
        'opener':'Opener','middle_order':'Middle','wicket_keeper':'WK','finisher':'Finisher',
        'spinner':'Spinner','fast_bowler':'Fast'
    }

    def order_xi(df_or_list):
        """Return ordered list of player names: batters first (openers, middle, wicket-keeper, finisher), then spinners, then fast bowlers."""
        players = []
        if isinstance(df_or_list, pd.DataFrame):
            df = df_or_list.copy()
            # use role column if exists, else lookup roles_map
            if 'role' in df.columns:
                df['role'] = df['role'].fillna('').astype(str).str.lower()
            df['player_name'] = df['player_name'].astype(str).str.strip()
            for r in ROLE_ORDER:
                selected = df[df['role']==r]['player_name'].tolist()
                players.extend(selected)
            # append any unknowns
            rest = [p for p in df['player_name'].tolist() if p not in players]
            players.extend(rest)
        else:
            # df_or_list is a list of names; use roles_map to order
            names = [str(x).strip() for x in df_or_list]
            grouped = {r: [] for r in ROLE_ORDER}
            unknown = []
            for n in names:
                r = roles_map.get(n.lower(), '')
                if r in ROLE_ORDER:
                    grouped[r].append(n)
                else:
                    unknown.append(n)
            for r in ROLE_ORDER:
                players.extend(grouped[r])
            players.extend(unknown)
        # final ensure uniqueness and keep original order for ties
        seen=set(); out=[]
        for p in players:
            if p not in seen:
                out.append(p); seen.add(p)
        return out

    # read session-state
    best_xi_df = st.session_state.best_xi.copy()
    opponent_list = st.session_state.opponent_xi.copy()
    venue = st.session_state.input_venue

    # order XIs
    my_xi_ordered = order_xi(best_xi_df)
    opp_xi_ordered = order_xi(opponent_list)

    # UI layout: two columns for XI lists and controls
    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown("#### 🏏 Your Playing XI")
        # Show compact list: batsmen first then bowlers; show small role badges
        for p in my_xi_ordered:
            r = (best_xi_df.loc[best_xi_df['player_name']==p, 'role'].iloc[0]
                 if p in best_xi_df['player_name'].tolist() and 'role' in best_xi_df.columns else roles_map.get(p.lower(), ''))
            label = ROLE_LABEL.get(r, r.title() if r else '')
            st.write(f"- **{p}** {f'· {label}' if label else ''}")

        st.markdown("#### 🎯 Opponent XI")
        # show opponent ordered compactly; include role if found
        for p in opp_xi_ordered:
            r = roles_map.get(p.lower(), '')
            label = ROLE_LABEL.get(r, r.title() if r else '')
            st.write(f"- {p} {f'· {label}' if label else ''}")

        # controls
        st.markdown("---")
        st.write(f"**Venue:** {venue}")
        phase_list = sorted(stats_df['phase'].dropna().unique())
        phase = st.selectbox("Select Phase", phase_list, index=0)
        if st.button("Compute Matchups"):
            st.session_state['_compute_matchups'] = True
        else:
            # preserve previous unless just triggered
            if '_compute_matchups' not in st.session_state:
                st.session_state['_compute_matchups'] = False

    with col2:
        st.markdown("### 🔍 Inspect Player Matchups")
        # allow selecting which player's matchup to inspect (from either side)
        inspect_side = st.radio("Choose XI", ("Your XI", "Opponent XI"), horizontal=True)
        if inspect_side == "Your XI":
            sel_player = st.selectbox("Select batsman from your XI", my_xi_ordered)
            inspect_vs = opp_xi_ordered  # show opponent bowlers
            inspect_bowlers_list = opp_xi_ordered
            inspector_role = 'batsman'
        else:
            sel_player = st.selectbox("Select batsman from opponent XI", opp_xi_ordered)
            inspect_vs = my_xi_ordered  # show our bowlers
            inspect_bowlers_list = my_xi_ordered
            inspector_role = 'batsman'

        # small description
        st.markdown(f"Showing top bowlers vs **{sel_player}** at **{venue}** during **{phase}**.")

        # only compute when user asked compute (avoids refresh wipes)
        if st.session_state.get('_compute_matchups', False):
            # compute matchups: for selected player vs each bowler in list, call model feature builder and predict
            @st.cache_data(show_spinner=False)
            def compute_probs_for_pair(batsman, bowlers_list, venue, phase):
                rows=[]
                for bowler in bowlers_list:
                    # build features (reuse same logic as training helper)
                    b = str(batsman).lower(); w = str(bowler).lower(); v = str(venue).lower(); p = str(phase).lower()
                    # compute small features from ball-by-ball data
                    sub_bats = stats_df[stats_df['batsman'].str.lower()==b]
                    recent_form = 0.0
                    if 'match_id' in sub_bats.columns and not sub_bats.empty:
                        try:
                            match_runs = sub_bats.groupby('match_id')['runs_scored'].sum().reset_index().sort_values('match_id')
                            recent_form = match_runs['runs_scored'].shift(1).rolling(3, min_periods=1).mean().iloc[-1]
                            if np.isnan(recent_form): recent_form = 0.0
                        except Exception:
                            recent_form = 0.0
                    # bowler recent wickets
                    bw = stats_df[stats_df['bowler'].str.lower()==w]
                    bowler_wickets_last50 = float(bw['is_wicket'].shift(1).tail(50).sum()) if not bw.empty else 0.0
                    # batsman runs vs this bowler
                    hv = stats_df[(stats_df['batsman'].str.lower()==b)&(stats_df['bowler'].str.lower()==w)]
                    batsman_runs_vs_bowler_last50 = float(hv['runs_scored'].shift(1).tail(50).sum()) if not hv.empty else 0.0
                    vb = stats_df[(stats_df['batsman'].str.lower()==b)&(stats_df['phase'].str.lower()==p)]
                    bat_phase_rpb = float(vb['runs_scored'].sum() / (len(vb) if len(vb)>0 else 1))
                    bwp = stats_df[(stats_df['bowler'].str.lower()==w)&(stats_df['phase'].str.lower()==p)]
                    bp_wicket_rate = float(bwp['is_wicket'].sum() / (len(bwp) if len(bwp)>0 else 1))
                    # assemble row consistent with training features
                    row = {
                        'batsman_l': b, 'bowler_l': w, 'venue_l': v, 'phase_l': p,
                        'recent_bat_form': float(recent_form),
                        'bowler_wickets_last50': float(bowler_wickets_last50),
                        'batsman_runs_vs_bowler_last50': float(batsman_runs_vs_bowler_last50),
                        'bat_phase_rpb': float(bat_phase_rpb),
                        'bp_wicket_rate': float(bp_wicket_rate)
                    }
                    # fill missing features expected by model with zeros
                    for f in xgb_feature_cols:
                        if f not in row: row[f] = 0.0
                    df_row = pd.DataFrame([row])[xgb_feature_cols]
                    # encode categoricals
                    for col, le in xgb_encoders.items():
                        if col in df_row.columns:
                            val = str(df_row[col].iloc[0])
                            try:
                                if val in le.classes_:
                                    df_row[col] = le.transform([val])
                                else:
                                    df_row[col] = 0
                            except Exception:
                                try:
                                    df_row[col] = le.transform([val])
                                except Exception:
                                    df_row[col] = 0
                    df_row = df_row.astype(float).fillna(0)
                    # predict
                    try:
                        prob = float(xgb_model.predict_proba(df_row)[0][1]) * 100.0
                    except Exception:
                        prob = np.nan
                    rows.append((bowler, prob))
                # sort descending
                rows = sorted(rows, key=lambda x: (0 if np.isnan(x[1]) else x[1]), reverse=True)
                return rows

            # compute probabilities for selected player only (fast)
            with st.spinner("Calculating probabilities..."):
                pair_probs = compute_probs_for_pair(sel_player, inspect_vs, venue, phase)

            # compact display: top N + full table toggle
            top_n = 6
            st.markdown(f"**Top {top_n} bowlers vs {sel_player}**")
            df_top = pd.DataFrame(pair_probs[:top_n], columns=["Bowler","Dismissal %"])
            df_top["Dismissal %"] = df_top["Dismissal %"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "–")
            st.table(df_top)

            with st.expander("Show full 11×11 matchups for this player"):
                df_full = pd.DataFrame(pair_probs, columns=["Bowler","Dismissal %"])
                df_full["Dismissal %"] = df_full["Dismissal %"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "–")
                st.dataframe(df_full, use_container_width=True)

        else:
            st.info("Click **Compute Matchups** (left) to generate predictions; then select a player to inspect.")

    # end columns
