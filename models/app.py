import streamlit as st
import pandas as pd
import random

# -----------------------------
# 🔹 Load Player Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("D:\AI ML Cricket Project CIM model\CIM\data\player_stats_venue.csv")  # 📂 Make sure your CSV is in same folder

player_pool = load_data()

# -----------------------------
# 🔹 GA Helper Functions
# -----------------------------
def is_valid_team(team):
    if team is None or len(team) != 11:
        return False

    role_counts = team['role'].value_counts().to_dict()
    required_roles = {
        'opener': 2,
        'middle_order': 2,
        'finisher': 1,
        'wicket_keeper': 1,
        'spinner': 2,
        'fast_bowler': 3
    }

    for role, required in required_roles.items():
        available = role_counts.get(role, 0)
        if available < required:
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

    if role_counts.get('wicket_keeper', 0) < 1:
        return False

    # ✅ spinner + fast bowler must total 5
    if role_counts.get('spinner', 0) + role_counts.get('fast_bowler', 0) != 5:
        return False

    foreign_players = team[team['indian'].str.lower() != 'yes']
    if not (2 <= len(foreign_players) <= 4):
        return False

    return True


def fitness(team):
    bat_roles = ['opener', 'middle_order', 'finisher', 'wicket_keeper']
    bat_score = team[team['role'].isin(bat_roles)]['bat_avg'].sum() + \
                (team[team['role'].isin(bat_roles)]['bat_sr'].sum() / 15)

    bowl_roles = ['spinner', 'fast_bowler']
    bowl_score = team[team['role'].isin(bowl_roles)]['wickets'].sum() * 2 - \
                 team[team['role'].isin(bowl_roles)]['econ'].sum()

    penalty = 100 if not is_valid_team(team) else 0
    return bat_score + bowl_score - penalty


def generate_random_team():
    required_roles = {
        'opener': 2,
        'middle_order': 2,
        'finisher': 1,
        'wicket_keeper': 1,
        'spinner': 2,
        'fast_bowler': 3
    }

    team = pd.DataFrame()
    for role, count in required_roles.items():
        candidates = player_pool[player_pool['role'] == role]
        if len(candidates) < count:
            return None
        selected = candidates.sample(count)
        team = pd.concat([team, selected])

    # ✅ Ensure at least 3 fast bowlers
    fast_bowlers = team[team['role'] == 'fast_bowler']
    if len(fast_bowlers) < 3:
        needed = 3 - len(fast_bowlers)
        extras = player_pool[(player_pool['role'] == 'fast_bowler') & (~player_pool['player_name'].isin(team['player_name']))]
        if len(extras) >= needed:
            to_remove = team[~team['role'].isin(['wicket_keeper'])].sample(needed)
            team = team.drop(to_remove.index)
            team = pd.concat([team, extras.sample(needed)])

    return team.reset_index(drop=True)


def crossover(team1, team2):
    required_roles = {
        'opener': 2,
        'middle_order': 2,
        'finisher': 1,
        'wicket_keeper': 1,
        'spinner': 2,
        'fast_bowler': 3
    }

    new_team = pd.DataFrame()
    for role, count in required_roles.items():
        candidates = pd.concat([team1[team1['role'] == role], team2[team2['role'] == role]]).drop_duplicates(subset='player_name')

        if len(candidates) < count:
            backup = player_pool[(player_pool['role'] == role) & (~player_pool['player_name'].isin(candidates['player_name']))]
            needed = count - len(candidates)
            candidates = pd.concat([candidates, backup.sample(min(needed, len(backup)))])

        selected = candidates.sample(min(count, len(candidates)))
        new_team = pd.concat([new_team, selected])

    return new_team.reset_index(drop=True)


def mutate(team):
    team = team.sample(frac=1).reset_index(drop=True)
    for i in range(len(team)):
        role = team.iloc[i]['role']
        current_names = team['player_name'].tolist()

        replacements = player_pool[(player_pool['role'] == role) & (~player_pool['player_name'].isin(current_names))]
        if not replacements.empty:
            new_player = replacements.sample(1).iloc[0]
            team.iloc[i] = new_player
            break
    return team.reset_index(drop=True)


def run_genetic_algorithm(generations=50, population_size=50):
    population = [generate_random_team() for _ in range(population_size)]
    population = [team for team in population if team is not None]

    if not population:
        return None

    for _ in range(generations):
        scored = [(team, fitness(team)) for team in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_teams = [x[0] for x in scored[:10] if x[0] is not None]

        if len(top_teams) < 2:
            return None

        new_gen = top_teams.copy()
        while len(new_gen) < population_size:
            t1, t2 = random.sample(top_teams, 2)
            child = crossover(t1, t2)
            if random.random() < 0.2:
                child = mutate(child)
            if is_valid_team(child):
                new_gen.append(child)
        population = new_gen

    return max(population, key=fitness)


# -----------------------------
# 🔹 Streamlit UI
# -----------------------------
st.title("🏏 Dream Team Generator (Genetic Algorithm)")

st.write("Upload a **players.csv** file with columns: `player_name`, `role`, `bat_avg`, `bat_sr`, `wickets`, `econ`, `indian`")

if st.button("⚡ Generate Best XI"):
    best_team = run_genetic_algorithm()
    if best_team is not None:
        role_order = ['opener', 'middle_order', 'wicket_keeper', 'finisher', 'spinner', 'fast_bowler']
        best_team = best_team.assign(role_order_index=best_team['role'].apply(lambda r: role_order.index(r)))
        best_team = best_team.sort_values(by='role_order_index').drop(columns='role_order_index').reset_index(drop=True)

        st.subheader("✅ Best Playing XI")
        st.dataframe(best_team)

        st.success(f"🏆 Total Fitness Score: {round(fitness(best_team), 2)}")
    else:
        st.error("❌ Could not generate a valid team. Check your dataset.")


