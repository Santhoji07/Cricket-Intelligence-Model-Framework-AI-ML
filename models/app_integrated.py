# app_integrated.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import math
from pathlib import Path
from collections import Counter, defaultdict

# -------------------------
# CONFIG: update paths here
# -------------------------
PLAYER_STATS_CSV = r"D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv"
PLAYER_ROLES_CSV  = r"D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv"
BALL_BY_BALL_CSV  = r"D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv"

SVM_MODEL_FILE = "svm_dismissal_model.pkl"
SVM_FEATURES_FILE = "svm_dismissal_features.pkl"
XGB_MODEL_FILE = "xgb_dismissal_model.pkl"
XGB_FEATURES_FILE = "xgb_dismissal_features.pkl"

# thresholds
APRIORI_SUPPORT_MIN = 5      # min absolute count for frequent matchup (adjust)
DISMISSAL_PROB_THRESHOLD = 0.70  # show only pairs >= 70% by default
MONTE_CARLO_ITER = 500       # number of simulated matches (reduce to speed up)

# -------------------------
# Utilities & Data Loading
# -------------------------
@st.cache_data
def load_csv_safe(path):
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

player_stats = load_csv_safe(PLAYER_STATS_CSV)
roles_df = load_csv_safe(PLAYER_ROLES_CSV)
ball_by_ball = load_csv_safe(BALL_BY_BALL_CSV)

if player_stats is None:
    st.error(f"Player-stats CSV not found at: {PLAYER_STATS_CSV}")
    st.stop()
if roles_df is None:
    st.error(f"Player-roles CSV not found at: {PLAYER_ROLES_CSV}")
    st.stop()
if ball_by_ball is None:
    st.warning("Ball-by-ball dataset not found — Apriori and some features will be limited.")

# normalize lowercasing columns we use
for df in [player_stats, roles_df]:
    df.columns = [c.lower().strip() for c in df.columns]

# convenience lowercases
if 'player_name' in player_stats.columns:
    player_stats['player_name'] = player_stats['player_name'].astype(str).str.strip()
if 'player_name' in roles_df.columns:
    roles_df['player_name'] = roles_df['player_name'].astype(str).str.strip()
if 'franchise' in roles_df.columns:
    roles_df['franchise'] = roles_df['franchise'].astype(str).str.strip()

# load models if present
def try_load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

svm_model = try_load_model(SVM_MODEL_FILE)
xgb_model = try_load_model(XGB_MODEL_FILE)

def try_load_features(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

svm_features = try_load_features(SVM_FEATURES_FILE)
xgb_features = try_load_features(XGB_FEATURES_FILE)

# -------------------------
# Genetic Algorithm (team selector)
# -------------------------
def build_player_pool_for(franchise, venue):
    # get franchise squad players from roles_df
    squad = roles_df[roles_df['franchise'].str.lower() == franchise.lower()]
    if squad.empty:
        return pd.DataFrame()
    squad_players = set(squad['player_name'].str.lower())

    # filter player_stats by venue and squad players
    df = player_stats.copy()
    df['player_name_l'] = df['player_name'].str.lower()
    df['venue_l'] = df['venue'].astype(str).str.lower()
    pool = df[(df['venue_l'] == venue.lower()) & (df['player_name_l'].isin(squad_players))].copy()

    # fallback thresholds for matches
    for min_matches in [3,2,1,0]:
        sub = pool[pool.get('matches', 0) >= min_matches]
        if len(sub) >= 11:
            return sub.reset_index(drop=True)
    # if nothing satisfies, return whatever pool has
    return pool.reset_index(drop=True)

# GA helper functions (kept simple & deterministic-ish)
def is_valid_team(team_df):
    if team_df is None or len(team_df) != 11:
        return False
    req = {'opener':2, 'middle_order':2, 'finisher':1, 'wicket_keeper':1, 'spinner':2, 'fast_bowler':3}
    counts = team_df['role'].value_counts().to_dict()
    # flexible substitutions NOT implemented here; keep strict to avoid bad teams
    for r,c in req.items():
        if counts.get(r,0) < c:
            return False
    # bowlers exact
    if counts.get('spinner',0) + counts.get('fast_bowler',0) != 5:
        return False
    # overseas constraint
    foreign_count = len(team_df[team_df['indian'].str.lower() != 'yes'])
    if not (2 <= foreign_count <= 4):
        return False
    return True

def fitness(team_df):
    # simple scoring: batting avg + (sr/15) + bowl_wickets*2 - econ
    bat_roles = ['opener','middle_order','finisher','wicket_keeper']
    bowl_roles = ['spinner','fast_bowler']
    bat_score = team_df[team_df['role'].isin(bat_roles)].get('bat_avg',0).sum()
    bat_score += team_df[team_df['role'].isin(bat_roles)].get('bat_sr',0).sum() / 15.0
    bowl_score = team_df[team_df['role'].isin(bowl_roles)].get('wickets',0).sum()*2
    bowl_score -= team_df[team_df['role'].isin(bowl_roles)].get('econ',0).sum()
    penalty = -100 if not is_valid_team(team_df) else 0
    return bat_score + bowl_score + penalty

def generate_random_team_from_pool(player_pool):
    required = {'opener':2,'middle_order':2,'finisher':1,'wicket_keeper':1,'spinner':2,'fast_bowler':3}
    team = pd.DataFrame()
    for role,count in required.items():
        candidates = player_pool[player_pool['role']==role]
        if len(candidates) < count:
            return None
        team = pd.concat([team, candidates.sample(count)])
    return team.reset_index(drop=True)

def crossover(t1, t2, pool):
    required = {'opener':2,'middle_order':2,'finisher':1,'wicket_keeper':1,'spinner':2,'fast_bowler':3}
    child = pd.DataFrame()
    for role,count in required.items():
        candidates = pd.concat([t1[t1['role']==role], t2[t2['role']==role]]).drop_duplicates(subset='player_name')
        if len(candidates) < count:
            backup = pool[(pool['role']==role) & (~pool['player_name'].isin(candidates['player_name']))]
            if len(backup)>0:
                candidates = pd.concat([candidates, backup.sample(min(count-len(candidates), len(backup)))])
        if len(candidates)==0:
            continue
        child = pd.concat([child, candidates.sample(min(count, len(candidates)))])
    # if >11, trim; if <11, fill from pool
    if len(child) > 11:
        child = child.sample(11)
    if len(child) < 11:
        remaining = pool[~pool['player_name'].isin(child['player_name'])]
        need = 11 - len(child)
        if len(remaining) >= need:
            child = pd.concat([child, remaining.sample(need)])
    return child.reset_index(drop=True)

def mutate(team, pool):
    team = team.copy()
    i = random.randrange(len(team))
    role = team.loc[i,'role']
    candidates = pool[(pool['role']==role) & (~pool['player_name'].isin(team['player_name']))]
    if len(candidates)>0:
        team.loc[i] = candidates.sample(1).iloc[0]
    return team.reset_index(drop=True)

def run_genetic_algorithm_for(franchise, venue, generations=30, pop_size=40):
    pool = build_player_pool_for(franchise, venue)
    if pool.empty or len(pool) < 11:
        return None, pool

    # seed initial population
    pop = []
    max_tries = pop_size * 5
    tries = 0
    while len(pop) < pop_size and tries < max_tries:
        t = generate_random_team_from_pool(pool)
        tries += 1
        if t is not None and is_valid_team(t):
            pop.append(t)

    # fallback: allow partially valid teams if still empty
    if not pop:
        for _ in range(pop_size):
            cand = generate_random_team_from_pool(pool)
            if cand is not None:
                pop.append(cand)

    # if even that fails, stop gracefully
    if not pop:
        return None, pool

    # GA main loop
    for gen in range(generations):
        scored = [(t, fitness(t)) for t in pop if t is not None]
        if not scored:
            # if no valid teams left, try to rebuild population
            for _ in range(pop_size):
                t = generate_random_team_from_pool(pool)
                if t is not None:
                    pop.append(t)
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [x[0] for x in scored[:max(2, pop_size // 5)] if x[0] is not None]

        # safety: if top is empty, repopulate using best few
        if not top:
            top = [x[0] for x in scored[:2]]

        new_pop = top.copy()

        while len(new_pop) < pop_size:
            if len(top) >= 2:
                a, b = random.sample(top, 2)
            else:
                a = b = top[0] if top else pop[0]

            child = crossover(a, b, pool)
            if random.random() < 0.2:
                child = mutate(child, pool)

            if len(child) == 11 and is_valid_team(child):
                new_pop.append(child)
            else:
                # fallback: add a random team to maintain diversity
                cand = generate_random_team_from_pool(pool)
                if cand is not None:
                    new_pop.append(cand)
                else:
                    new_pop.append(child)

        pop = new_pop

    best = max(pop, key=fitness) if pop else None
    return (best.reset_index(drop=True), pool) if best is not None else (None, pool)


# -------------------------
# Apriori-like matchup mining
# -------------------------
def mine_top_bowler_batsman_pairs(ball_df, franchise, min_support=APRIORI_SUPPORT_MIN):
    # We compute counts of (batsman, bowler) dismissals at venue/franchise history.
    if ball_df is None:
        return []
    df = ball_df.copy()
    # ensure columns
    for c in ['batsman','bowler','dismissal_type','venue','match_id']:
        if c not in df.columns:
            return []
    df['batsman']=df['batsman'].astype(str).str.strip()
    df['bowler']=df['bowler'].astype(str).str.strip()
    # count dismissals where bowler dismissed batsman
    dismissals = df[~df['dismissal_type'].isna()].copy()
    pair_counts = dismissals.groupby(['batsman','bowler']).size().reset_index(name='count')
    # filter by min_support and sort by count desc
    frequent = pair_counts[pair_counts['count'] >= min_support].sort_values('count',ascending=False)
    # return top list
    results = frequent.to_dict('records')
    return results

# -------------------------
# Build features for model prediction
# -------------------------
def compute_pair_features(batsman, bowler, venue, phase, stats_df):
    # features used in your original models: recent_form, bowler_wickets_venue, and categorical fields
    recent_form = 0.0
    bowler_wickets_venue = 0.0
    try:
        # recent_form: avg runs in last 3 innings (exclude current)
        bdf = stats_df[stats_df['batsman'] == batsman].sort_values('date')
        if not bdf.empty:
            runs_by_match = bdf.groupby('match_id')['runs_scored'].sum().shift(1)
            recent = runs_by_match.dropna().tail(3)
            recent_form = recent.mean() if len(recent)>0 else 0.0
    except Exception:
        recent_form = 0.0
    try:
        # bowler wickets at venue in last 5 balls (approx using ball-by-ball)
        if 'is_wicket' in stats_df.columns:
            b = stats_df[(stats_df['bowler']==bowler) & (stats_df['venue']==venue)].sort_values('date')
            if not b.empty:
                bowler_wickets_venue = b['is_wicket'].shift(1).dropna().tail(5).sum()
    except Exception:
        bowler_wickets_venue = 0.0
    return {
        'batsman': batsman,
        'bowler': bowler,
        'venue': venue,
        'phase': phase if phase is not None else "",
        'recent_form': float(recent_form),
        'bowler_wickets_venue': float(bowler_wickets_venue)
    }

# -------------------------
# Dismissal predict wrapper (handles both SVM/XGB if available)
# -------------------------
def predict_dismissal_probability(feature_dict, model=None, feature_cols=None):
    # feature_dict: dict as returned by compute_pair_features
    # Attempts to produce probability that batsman will be dismissed by bowler
    if model is None:
        return None
    # Construct model_df
    if feature_cols is None:
        # fallback
        cols = ['batsman','bowler','venue','phase','recent_form','bowler_wickets_venue']
    else:
        cols = feature_cols
    model_df = pd.DataFrame([{c: feature_dict.get(c, "") for c in cols}])
    # fill NA numeric zeros
    model_df = model_df.fillna(0)
    try:
        prob = model.predict_proba(model_df)[0][1]
    except Exception:
        # If model doesn't support predict_proba, try decision_function + logistic-like mapping
        try:
            score = model.decision_function(model_df)[0]
            prob = 1.0 / (1.0 + math.exp(-score))
        except Exception:
            prob = None
    return prob

# -------------------------
# Monte Carlo match simulator (simplified; fast)
# -------------------------
def simulate_match_simple(batting_xi, bowling_xi, venue, phase, stats_df, dismissal_model=None, feat_cols=None, iterations=MONTE_CARLO_ITER):
    # returns dict with win probabilities
    if dismissal_model is None:
        # if we don't have models, just compare team batting averages as a naive metric
        bat_avgs = {p: (stats_df[stats_df['batsman']==p]['runs_scored'].mean() or 20) for p in batting_xi}
    else:
        bat_avgs = {}
        for p in batting_xi:
            # approximate using 'recent_form' computed earlier
            feat = compute_pair_features(p, bowling_xi[0], venue, phase, stats_df)
            bat_avgs[p] = feat['recent_form'] if feat['recent_form']>0 else 20.0

    def simulate_innings(bat_xi, bowl_xi):
        total = 0
        wickets = 0
        striker_idx = 0
        next_batsman_idx = 2
        balls = 120
        striker = bat_xi[0] if len(bat_xi)>0 else None
        non_striker = bat_xi[1] if len(bat_xi)>1 else None
        batsmen_order = list(bat_xi)
        batsman_out = set()
        for ball in range(balls):
            over = ball // 6
            bowler = bowl_xi[over % len(bowl_xi)]
            batsman = striker
            # compute dismissal prob for this batsman vs bowler
            feat = compute_pair_features(batsman, bowler, venue, phase, stats_df)
            prob = predict_dismissal_probability(feat, dismissal_model, feat_cols) if dismissal_model else 0.03
            # scale prob - models are per-match/phase; treat as per-ball approx (conservative)
            per_ball_prob = min(max(prob * 0.02, 0.0005), 0.2) if prob is not None else 0.01
            if random.random() < per_ball_prob:
                # wicket
                wickets += 1
                batsman_out.add(batsman)
                total += 0  # no run this ball
                # bring next batsman
                idx = batsmen_order.index(batsman)
                # find next not out
                remaining = [p for p in batsmen_order if p not in batsman_out]
                if len(remaining) <= 1:
                    break
                # new striker is next alive (approx)
                for p in batsmen_order:
                    if p not in batsman_out and p not in [striker, non_striker]:
                        striker = p
                        break
            else:
                # run scoring: sample from Poisson with mean proportional to batsman's recent form/12
                mean_runs = max(2.0, bat_avgs.get(batsman, 20.0) / 6.0)
                runs = np.random.poisson(mean_runs)
                # cap runs per ball to 6 for sanity
                runs = min(runs,6)
                total += runs
                if runs % 2 == 1:
                    striker, non_striker = non_striker, striker
            # end of over swap
            if (ball+1) % 6 == 0:
                striker, non_striker = non_striker, striker
        return total

    team1_wins = 0
    team2_wins = 0
    draws = 0
    for i in range(iterations):
        score1 = simulate_innings(batting_xi, bowling_xi)
        score2 = simulate_innings(bowling_xi, batting_xi)  # swap roles for opposition innings
        if score1 > score2:
            team1_wins += 1
        elif score2 > score1:
            team2_wins += 1
        else:
            draws += 1
    return {
        'team1_win_prob': team1_wins/iterations,
        'team2_win_prob': team2_wins/iterations,
        'draw_prob': draws/iterations,
        'sim_iterations': iterations,
        'avg_score_team1': None,
        'avg_score_team2': None
    }

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide")
st.title("Integrated Hybrid Cricket Pipeline — GA + Apriori + Dismissal Model + Monte Carlo")

col1, col2 = st.columns(2)

with col1:
    franchise = st.selectbox("Select Franchise (to build XI)", sorted(roles_df['franchise'].dropna().unique()))
    venue = st.selectbox("Select Venue", sorted(player_stats['venue'].dropna().unique()))
    run_ga_button = st.button("🔧 Build Best XI (Genetic Algorithm)")

with col2:
    opponent_franchise = st.selectbox("Select Opponent Franchise (for Apriori)", sorted(roles_df['franchise'].dropna().unique()), index=0, key="opp_fr")
    run_apriori_button = st.button("🔎 Find Frequent Bowler-Batsman Matchups")
    run_sim_button = st.button("▶️ Monte Carlo simulate match")

# GA: build playing XI
best_xi = None
player_pool_used = None
if run_ga_button:
    with st.spinner("Running Genetic Algorithm — building best XI..."):
        best_xi, player_pool_used = run_genetic_algorithm_for(franchise, venue, generations=25, pop_size=40)
    if best_xi is None:
        st.error("Could not build Playing XI for selected franchise & venue. Check pool size.")
    else:
        st.success("Best XI (generated):")
        st.table(best_xi[['player_name','role','bat_avg','bat_sr','wickets','econ','indian']])

# Apriori pairing mining
apriori_pairs = []
if run_apriori_button:
    if ball_by_ball is None:
        st.error("Ball-by-ball data not loaded; cannot mine matchups.")
    else:
        with st.spinner("Mining frequent bowler-batsman dismissals..."):
            apriori_pairs = mine_top_bowler_batsman_pairs(ball_by_ball, opponent_franchise, min_support=APRIORI_SUPPORT_MIN)
        if not apriori_pairs:
            st.info("No strong frequent pairings found (increase dataset or lower support threshold).")
        else:
            st.subheader("Top bowler->batsman dismissal pairs (count >= support)")
            top = pd.DataFrame(apriori_pairs).head(20)
            st.dataframe(top)

# Allow user to input opponent XI manually (or auto from opponent franchise)
st.subheader("Opponent XI (provide 11 players)")
opp_candidates = roles_df[roles_df['franchise'].str.lower()==opponent_franchise.lower()]['player_name'].unique()
opponent_xi = st.multiselect("Opponent XI (pick up to 11)", options=sorted(opponent_candidates) if (opponent_candidates:= list(opp_candidates)) else [], help="If list empty, type names manually", max_selections=11)

if len(opponent_xi) < 11:
    st.info("You may provide fewer than 11 opponent players (Monte Carlo will cycle bowlers).")

# Dismissal predictions using model(s)
st.subheader("Dismissal predictions (batsman vs bowler)")

# choose which model to use
model_choice = st.radio("Choose model for dismissal probability", ("XGBoost (if available)", "SVM (if available)", "None (no model)"))
model_obj = None
model_feats = None
if model_choice.startswith("XGBoost"):
    if xgb_model is None:
        st.warning("XGBoost model not available; fallback to SVM if present.")
        model_obj = svm_model
        model_feats = svm_features
    else:
        model_obj = xgb_model
        model_feats = xgb_features
elif model_choice.startswith("SVM"):
    if svm_model is None:
        st.warning("SVM model not available; fallback to XGBoost if present.")
        model_obj = xgb_model
        model_feats = xgb_features
    else:
        model_obj = svm_model
        model_feats = svm_features
else:
    model_obj = None
    model_feats = None

# if best_xi is present, list pairs between best_xi and opponent_xi (or select top pairings from Apriori)
if best_xi is not None and opponent_xi:
    # generate features & predictions for each batsman in best_xi vs each bowler in opponent
    pairs = []
    for batsman in best_xi['player_name'].tolist():
        for bowler in opponent_xi:
            feat = compute_pair_features(batsman, bowler, venue, phase=None, stats_df=ball_by_ball if ball_by_ball is not None else player_stats)
            prob = predict_dismissal_probability(feat, model_obj, model_feats)
            pairs.append((batsman, bowler, prob if prob is not None else np.nan))
    # show high-prob pairs above threshold
    dfp = pd.DataFrame(pairs, columns=['batsman','bowler','prob'])
    dfp['prob_pct'] = dfp['prob']*100
    dfp = dfp.sort_values('prob_pct', ascending=False)
    st.markdown(f"### Predicted dismissal probabilities (showing >= {int(DISMISSAL_PROB_THRESHOLD*100)}%)")
    high = dfp[dfp['prob'] >= DISMISSAL_PROB_THRESHOLD]
    if not high.empty:
        st.dataframe(high[['batsman','bowler','prob_pct']].reset_index(drop=True))
    else:
        st.info("No predicted dismissal pair >= threshold. You can lower threshold or check models/data.")
else:
    st.info("To predict dismissal probabilities: first generate GA Best XI and provide Opponent XI.")

# Monte Carlo simulation
if run_sim_button:
    if best_xi is None:
        st.error("No GA Best XI available — run GA first.")
    elif not opponent_xi:
        st.error("Please enter opponent XI before running simulation.")
    else:
        with st.spinner("Running Monte Carlo simulations (this may take time)..."):
            bat_xi = best_xi['player_name'].tolist()
            bowl_xi = opponent_xi
            sim_res = simulate_match_simple(bat_xi, bowl_xi, venue, phase=None, stats_df=ball_by_ball if ball_by_ball is not None else player_stats, dismissal_model=model_obj, feat_cols=model_feats, iterations=MONTE_CARLO_ITER)
        st.subheader("Monte Carlo Simulation Results")
        st.write(f"Sim iterations: {sim_res['sim_iterations']}")
        st.metric("Team (franchise) win probability", f"{sim_res['team1_win_prob']*100:.1f}%")
        st.metric("Opponent win probability", f"{sim_res['team2_win_prob']*100:.1f}%")
        st.metric("Draw probability", f"{sim_res['draw_prob']*100:.1f}%")
        st.info("Monte Carlo is simplified. For a more accurate simulation we should incorporate ball-level dismissal probs, bowling quotas, batting order handling, and per-player distributions.")

st.sidebar.header("Notes / Tips")
st.sidebar.markdown("""
- Put your `.pkl` models in the same folder (svm_dismissal_model.pkl, xgb_dismissal_model.pkl).
- Adjust `MONTE_CARLO_ITER` and thresholds at the top of the script to trade accuracy vs speed.
- This integrated app is a starting point: you can replace the simple Monte Carlo with your full match simulator later.
""")
