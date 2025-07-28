import streamlit as st
import pandas as pd
import random

# -----------------------------
# 🔹 LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    stats = pd.read_csv("D:\AI ML Cricket Project CIM model\CIM\data\player_stats_venue.csv")   # 📂 Must have: player_name, venue, matches, bat_avg, bat_sr, wickets, econ, role, indian
    roles = pd.read_csv("D:\AI ML Cricket Project CIM model\CIM\data\player_roles.csv")         # 📂 Must have: player_name, franchise
    return stats, roles

stats_df, roles_df = load_data()


# 👇 Add here
roles_df = roles_df.dropna(subset=['franchise'])
roles_df['franchise'] = roles_df['franchise'].astype(str)

# ✅ This will avoid sorting issues


# -----------------------------
# 🔹 HELPER: BUILD PLAYER POOL
# -----------------------------
def build_player_pool(venue, franchise):
    """Filter players by franchise & venue, with fallback thresholds."""
    squad = roles_df[roles_df['franchise'].str.lower() == franchise.lower()]
    pool = stats_df[(stats_df['venue'].str.lower() == venue.lower()) &
                    (stats_df['player_name'].isin(squad['player_name']))]

    # ✅ No merge for 'indian' — it’s already in stats_df

    # Start strict (>=3 matches) and relax if needed
    thresholds = [3, 2, 1, 0]
    final_pool = pd.DataFrame()
    for t in thresholds:
        subset = pool[pool['matches'] >= t]
        if len(subset) >= 11:
            final_pool = subset
            break
    if final_pool.empty:
        final_pool = pool  # fallback worst-case

    return final_pool.reset_index(drop=True)

# -----------------------------
# 🔹 TEAM VALIDATION
# -----------------------------
def is_valid_team(team):
    if team is None or len(team) != 11:
        return False

    role_counts = team['role'].value_counts().to_dict()
    required_roles = {'opener': 2, 'middle_order': 2, 'finisher': 1,
                      'wicket_keeper': 1, 'spinner': 2, 'fast_bowler': 3}

    # ✅ Check roles & patch flexibility
    for role, required in required_roles.items():
        available = role_counts.get(role, 0)
        if available < required:
            # Flexible patching
            if role == 'opener' and role_counts.get('middle_order', 0) >= required - available:
                role_counts['middle_order'] -= required - available
            elif role == 'middle_order' and role_counts.get('finisher', 0) >= required - available:
                role_counts['finisher'] -= required - available
            elif role == 'finisher' and role_counts.get('middle_order', 0) >= required - available:
                role_counts['middle_order'] -= required - available
            elif role == 'spinner' and role_counts.get('fast_bowler', 0) >= required - available:
                role_counts['fast_bowler'] -= required - available
            elif role == 'fast_bowler' and role_counts.get('spinner', 0) >= required - available:
                role_counts['spinner'] -= required - available
            else:
                return False

    # ✅ spinner + fast bowler must total 5
    if role_counts.get('spinner', 0) + role_counts.get('fast_bowler', 0) != 5:
        return False

    # ✅ overseas players check (now from player_stats_venue.csv)
    foreign_players = team[team['indian'].str.lower() != 'yes']
    if not (2 <= len(foreign_players) <= 4):
        return False

    return True

# -----------------------------
# 🔹 FITNESS FUNCTION
# -----------------------------
def fitness(team):
    bat_roles = ['opener', 'middle_order', 'finisher', 'wicket_keeper']
    bat_score = team[team['role'].isin(bat_roles)]['bat_avg'].sum() + \
                (team[team['role'].isin(bat_roles)]['bat_sr'].sum() / 15)

    bowl_roles = ['spinner', 'fast_bowler']
    bowl_score = team[team['role'].isin(bowl_roles)]['wickets'].sum() * 2 - \
                 team[team['role'].isin(bowl_roles)]['econ'].sum()

    penalty = 100 if not is_valid_team(team) else 0
    return bat_score + bowl_score - penalty

# -----------------------------
# 🔹 BUILD RANDOM TEAM
# -----------------------------
def generate_random_team(player_pool):
    required_roles = {'opener': 2, 'middle_order': 2, 'finisher': 1,
                      'wicket_keeper': 1, 'spinner': 2, 'fast_bowler': 3}

    team = pd.DataFrame()
    for role, count in required_roles.items():
        candidates = player_pool[player_pool['role'] == role]
        if len(candidates) >= count:
            selected = candidates.sample(count)
            team = pd.concat([team, selected])
        else:
            # fallback: take whatever exists
            selected = candidates.sample(len(candidates)) if not candidates.empty else pd.DataFrame()
            team = pd.concat([team, selected])

    # ✅ ensure exactly 11 players
    if len(team) > 11:
        team = team.sample(11)

    return team.reset_index(drop=True)

# -----------------------------
# 🔹 CROSSOVER & MUTATION
# -----------------------------
def crossover(team1, team2, player_pool):
    required_roles = {'opener': 2, 'middle_order': 2, 'finisher': 1,
                      'wicket_keeper': 1, 'spinner': 2, 'fast_bowler': 3}
    new_team = pd.DataFrame()

    for role, count in required_roles.items():
        candidates = pd.concat([team1[team1['role'] == role], team2[team2['role'] == role]]) \
                        .drop_duplicates(subset='player_name')
        if len(candidates) < count:
            backup = player_pool[(player_pool['role'] == role) &
                                 (~player_pool['player_name'].isin(candidates['player_name']))]
            needed = count - len(candidates)
            candidates = pd.concat([candidates, backup.sample(min(needed, len(backup)))])
        selected = candidates.sample(min(count, len(candidates)))
        new_team = pd.concat([new_team, selected])

    return new_team.reset_index(drop=True)

def mutate(team, player_pool):
    team = team.sample(frac=1).reset_index(drop=True)
    for i in range(len(team)):
        role = team.iloc[i]['role']
        replacements = player_pool[(player_pool['role'] == role) &
                                   (~player_pool['player_name'].isin(team['player_name']))]
        if not replacements.empty:
            team.iloc[i] = replacements.sample(1).iloc[0]
            break
    return team.reset_index(drop=True)

# -----------------------------
# 🔹 GA MAIN LOOP
# -----------------------------
def run_genetic_algorithm(player_pool, generations=50, population_size=50):
    population = [generate_random_team(player_pool) for _ in range(population_size)]

    for _ in range(generations):
        scored = [(team, fitness(team)) for team in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        top_teams = [x[0] for x in scored[:10] if x[0] is not None]
        if len(top_teams) < 2:
            break

        new_gen = top_teams.copy()
        while len(new_gen) < population_size:
            t1, t2 = random.sample(top_teams, 2)
            child = crossover(t1, t2, player_pool)
            if random.random() < 0.2:
                child = mutate(child, player_pool)
            if is_valid_team(child):
                new_gen.append(child)
        population = new_gen

    return max(population, key=fitness)

# -----------------------------
# 🔹 STREAMLIT UI
# -----------------------------
st.title("🏏 Dream XI Generator (Venue + Franchise Aware)")

venue_list = sorted(stats_df['venue'].unique())
team_list = sorted(roles_df['franchise'].unique())

venue = st.selectbox("📍 Select Venue", venue_list)
franchise = st.selectbox("🏢 Select Franchise", team_list)

if st.button("⚡ Generate Best XI"):
    pool = build_player_pool(venue, franchise)

    if len(pool) < 11:
        st.error("❌ Not enough players to build a team for this venue.")
    else:
        best_team = run_genetic_algorithm(pool)

        # 📊 Show Best XI
        role_order = ['opener', 'middle_order', 'wicket_keeper', 'finisher', 'spinner', 'fast_bowler']
        best_team = best_team.assign(order_idx=best_team['role'].apply(lambda r: role_order.index(r)))
        best_team = best_team.sort_values(by='order_idx').drop(columns='order_idx').reset_index(drop=True)

        st.subheader("✅ Best Playing XI")
        st.dataframe(best_team[['player_name', 'role', 'bat_avg', 'bat_sr', 'wickets', 'econ', 'indian']])

        st.success(f"🏆 Total Fitness Score: {round(fitness(best_team), 2)}")

        # 📊 Bench players
        bench = pool[~pool['player_name'].isin(best_team['player_name'])]
        st.subheader("📋 Bench Players")
        st.dataframe(bench[['player_name', 'role', 'matches']])

        # 📊 Excluded players
        excluded = roles_df[(roles_df['franchise'].str.lower() == franchise.lower()) &
                            (~roles_df['player_name'].isin(pool['player_name']))]
        st.subheader("🚫 Excluded Players (No Venue Data)")
        st.dataframe(excluded)
