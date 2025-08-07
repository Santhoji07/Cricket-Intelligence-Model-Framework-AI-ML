import streamlit as st
import pandas as pd
import random

# -------------------------------------
# 1. Load Data
# -------------------------------------
@st.cache_data
def load_player_stats():
    df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv")
    df.fillna(0, inplace=True)
    df.columns = [c.lower().strip() for c in df.columns]
    return df

@st.cache_data
def load_roles():
    df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    return df

player_stats = load_player_stats()
roles_df = load_roles()

# -------------------------------------
# 2. Select Franchise & Venue
# -------------------------------------
st.title("🏏 Dream XI Generator (GA-based)")
venue = st.selectbox("Select Venue", sorted(player_stats["venue"].dropna().unique()))
team_list = sorted(roles_df["franchise"].dropna().unique())
franchise = st.selectbox("Select Franchise", team_list)

# -------------------------------------
# 3. Build Player Pool
# -------------------------------------
@st.cache_data
def get_player_pool(venue, franchise):
    team_players = roles_df[roles_df["franchise"] == franchise]
    df = player_stats[player_stats["player_name"].isin(team_players["player_name"])]
    df = df[df["venue"] == venue]

    for min_matches in [3, 2, 1, 0]:
        filtered = df[df["matches"] >= min_matches]
        if len(filtered) >= 11:
            break

    merged = pd.merge(filtered, team_players, on="player_name")
    merged.fillna(0, inplace=True)
    merged.columns = [col.lower().strip() for col in merged.columns]

    # ✅ Debug Output
    st.write("📊 Columns after merge:", merged.columns.tolist())
    st.write("🧍 Rows after merge:", merged.shape[0])

    if "role" not in merged.columns:
        st.error("❌ 'role' column missing after merging. Likely due to mismatch in 'player_name'.")
        st.stop()

    return merged



player_pool = get_player_pool(venue, franchise)

if len(player_pool) < 11:
    st.warning("Not enough data for this team at the venue.")
    st.stop()

st.write(f"🧪 Player pool size: {len(player_pool)}")

# -------------------------------------
# 4. Helper Functions
# -------------------------------------
def is_valid_team(team):
    if team is None or len(team) != 11:
        return False

    roles = team["role"].value_counts().to_dict()
    required = {'opener':2, 'middle_order':2, 'finisher':1, 'wicket_keeper':1, 'spinner':2, 'fast_bowler':3}

    for role, count in required.items():
        available = roles.get(role, 0)
        if available < count:
            return False

    total_bowlers = roles.get("spinner", 0) + roles.get("fast_bowler", 0)
    if total_bowlers != 5:
        return False

    overseas = team[team["indian"].str.lower() != "yes"]
    if not (2 <= len(overseas) <= 4):
        return False

    return True

def fitness(team):
    bat_roles = ["opener", "middle_order", "finisher", "wicket_keeper"]
    bowl_roles = ["spinner", "fast_bowler"]

    bat_score = team[team["role"].isin(bat_roles)]["bat_avg"].sum()
    bat_score += team[team["role"].isin(bat_roles)]["bat_sr"].sum() / 15

    bowl_score = team[team["role"].isin(bowl_roles)]["wickets"].sum() * 2
    bowl_score -= team[team["role"].isin(bowl_roles)]["econ"].sum()

    penalty = -100 if not is_valid_team(team) else 0
    return bat_score + bowl_score + penalty

def generate_random_team():
    required = {'opener':2, 'middle_order':2, 'finisher':1, 'wicket_keeper':1, 'spinner':2, 'fast_bowler':3}
    team = pd.DataFrame()

    for role, count in required.items():
        pool = player_pool[player_pool["role"] == role]
        if len(pool) < count:
            return None
        team = pd.concat([team, pool.sample(count)])

    return team.reset_index(drop=True)

def crossover(t1, t2):
    roles = {'opener':2, 'middle_order':2, 'finisher':1, 'wicket_keeper':1, 'spinner':2, 'fast_bowler':3}
    child = pd.DataFrame()

    for role, count in roles.items():
        pool = pd.concat([t1[t1["role"] == role], t2[t2["role"] == role]])
        pool = pool.drop_duplicates(subset="player_name")
        if len(pool) < count:
            backup = player_pool[(player_pool["role"] == role) & (~player_pool["player_name"].isin(pool["player_name"]))]
            pool = pd.concat([pool, backup.sample(min(count - len(pool), len(backup)))])
        child = pd.concat([child, pool.sample(min(count, len(pool)))])
    return child.reset_index(drop=True)

def mutate(team):
    team = team.copy()
    idx = random.randint(0, len(team) - 1)
    role = team.loc[idx, "role"]
    pool = player_pool[(player_pool["role"] == role) & (~player_pool["player_name"].isin(team["player_name"]))]
    if not pool.empty:
        team.loc[idx] = pool.sample(1).iloc[0]
    return team

# -------------------------------------
# 5. Genetic Algorithm
# -------------------------------------
def run_ga(generations=20, population_size=30):
    population = [generate_random_team() for _ in range(population_size)]
    population = [p for p in population if p is not None]

    for _ in range(generations):
        scored = [(team, fitness(team)) for team in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [x[0] for x in scored[:10] if x[0] is not None]

        new_gen = top.copy()
        while len(new_gen) < population_size:
            p1, p2 = random.sample(top, 2)
            child = crossover(p1, p2)
            if random.random() < 0.2:
                child = mutate(child)
            if is_valid_team(child):
                new_gen.append(child)

        population = new_gen

    best = max(population, key=fitness)
    return best

# -------------------------------------
# 6. Generate Button
# -------------------------------------
if st.button("⚡ Generate Best Playing XI"):
    with st.spinner("Running Genetic Algorithm..."):
        best_team = run_ga()

    if best_team is not None:
        role_order = ['opener', 'middle_order', 'wicket_keeper', 'finisher', 'spinner', 'fast_bowler']
        best_team["role_order"] = best_team["role"].apply(lambda r: role_order.index(r))
        best_team = best_team.sort_values("role_order").drop(columns="role_order").reset_index(drop=True)

        st.subheader("✅ Best XI")
        st.dataframe(best_team[["player_name", "role", "bat_avg", "bat_sr", "wickets", "econ", "indian"]])

        st.success(f"🏆 Fitness Score: {round(fitness(best_team), 2)}")

        bench = player_pool[~player_pool["player_name"].isin(best_team["player_name"])]
        st.subheader("🪑 Bench")
        st.dataframe(bench[["player_name", "role", "bat_avg", "bat_sr", "wickets", "econ"]])
    else:
        st.error("❌ Could not find a valid team.")
