import pandas as pd
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv")
df.columns = [col.strip().lower() for col in df.columns]
df = df.rename(columns={
    'player': 'player_name',
    'avg': 'bat_avg',
    'sr': 'bat_sr'
})
df = df.dropna(subset=['player_name'])
df.fillna(0, inplace=True)

input_venue = input("Enter venue name: ").strip().lower()
input_team = input("Enter team name: ").strip().lower()

roles_df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv")
roles_df['player'] = roles_df['player'].str.strip()
roles_df['team'] = roles_df['team'].str.strip().str.lower()

squad_names = roles_df[roles_df['team'] == input_team]['player'].str.lower().tolist()
df['player_name_lower'] = df['player_name'].str.lower()
df['venue_lower'] = df['venue'].str.lower()

player_pool = pd.DataFrame()
for min_matches in [3, 2, 1, 0]:
    pool = df[(df['venue_lower'] == input_venue) & (df['player_name_lower'].isin(squad_names)) & (df['matches'] >= min_matches)].copy()
    if len(pool) >= 11:
        print(f" Player pool created with min_matches >= {min_matches}")
        player_pool = pool
        break

if player_pool.empty:
    print(" No players found from this team at this venue.")
    exit()

pool_names_lower = set(player_pool['player_name_lower'].tolist())
squad_not_in_pool = set(squad_names) - pool_names_lower
print(f"\n Players from Squad NOT in Player Pool (filtered out by venue or matches):")
print(", ".join(sorted([p.title() for p in squad_not_in_pool])) if squad_not_in_pool else "None")

print("\n🧾 Player pool summary:")
print(player_pool[['player_name', 'role', 'indian']])
print("\n📊 Role counts:")
print(player_pool['role'].value_counts())
print("\n🌏 Foreign player count:", len(player_pool[player_pool['indian'].str.lower() != 'yes']))
print("🇮🇳 Indian player count:", len(player_pool[player_pool['indian'].str.lower() == 'yes']))

print("\n🔍 Role availability check:")
role_needs = {'opener': 2, 'middle_order': 2, 'finisher': 1, 'wicket_keeper': 1, 'spinner': 2, 'fast_bowler': 3}
for role, required in role_needs.items():
    found = len(player_pool[player_pool['role'] == role])
    print(f"{role}: required={required}, found={found}")


def apply_role_substitutions(team):
    required_roles = {'opener': 2, 'middle_order': 2, 'finisher': 1, 'wicket_keeper': 1, 'spinner': 2, 'fast_bowler': 3}
    role_counts = team['role'].value_counts().to_dict()
    modified_team = team.copy()

    def replace_role(short_role, fallback_role):
        shortfall = required_roles[short_role] - role_counts.get(short_role, 0)
        if shortfall > 0 and role_counts.get(fallback_role, 0) > 0:
            for i in modified_team.index:
                if modified_team.at[i, 'role'] == fallback_role and shortfall > 0:
                    modified_team.at[i, 'role'] = short_role
                    shortfall -= 1

    replace_role('opener', 'middle_order')
    replace_role('middle_order', 'finisher')
    replace_role('finisher', 'middle_order')
    replace_role('spinner', 'fast_bowler')
    replace_role('fast_bowler', 'spinner')
    return modified_team


def is_valid_team(team):
    if team is None or len(team) != 11:
        return False

    role_counts = team['role'].value_counts().to_dict()
    required_roles = {'opener': 2, 'middle_order': 2, 'finisher': 1, 'wicket_keeper': 1, 'spinner': 2, 'fast_bowler': 3}

    try:
        for role, min_count in required_roles.items():
            if role_counts.get(role, 0) < min_count:
                return False
    except:
        return False

    foreign_players = team[team['indian'].str.lower() != 'yes']
    if not (2 <= len(foreign_players) <= 4):
        return False

    return True


def fitness(team):
    bat_roles = ['opener', 'middle_order', 'finisher', 'wicket_keeper']
    bat_score = team[team['role'].isin(bat_roles)]['bat_avg'].sum() + (team[team['role'].isin(bat_roles)]['bat_sr'].sum() / 15)
    bowl_roles = ['spinner', 'fast_bowler']
    bowl_score = team[team['role'].isin(bowl_roles)]['wickets'].sum() * 2 - team[team['role'].isin(bowl_roles)]['econ'].sum()
    penalty = 100 if not is_valid_team(team) else 0
    return bat_score + bowl_score - penalty


def generate_random_team():
    required_roles = {'opener': 2, 'middle_order': 2, 'finisher': 1, 'wicket_keeper': 1, 'spinner': 2, 'fast_bowler': 3}
    team = pd.DataFrame()

    for role, count in required_roles.items():
        candidates = player_pool[player_pool['role'] == role]
        if len(candidates) < count:
            team = pd.DataFrame()
            break
        selected = candidates.sample(count)
        team = pd.concat([team, selected])

    if team.empty:
        selected = player_pool.sample(11)
        team = selected

    fast_bowlers = team[team['role'] == 'fast_bowler']
    if len(fast_bowlers) < 3:
        needed = 3 - len(fast_bowlers)
        extras = player_pool[(player_pool['role'] == 'fast_bowler') & (~player_pool['player_name'].isin(team['player_name']))]
        if len(extras) >= needed:
            to_remove = team[~team['role'].isin(['wicket_keeper'])].sample(needed)
            team = team.drop(to_remove.index)
            team = pd.concat([team, extras.sample(needed)])

    current_foreign = team[team['indian'].str.lower() != 'yes']
    if len(current_foreign) < 4:
        needed = 4 - len(current_foreign)
        available_foreign = player_pool[(player_pool['indian'].str.lower() != 'yes') & (~player_pool['player_name'].isin(team['player_name']))].sample(frac=1)
        for _, row in available_foreign.iterrows():
            replaceable = team[(team['indian'].str.lower() == 'yes') & (team['role'] == row['role'])]
            if not replaceable.empty:
                team = team.drop(replaceable.iloc[0].name)
                team = pd.concat([team, pd.DataFrame([row])])
                needed -= 1
                if needed == 0:
                    break

    if len(team) == 11:
        return team.reset_index(drop=True)
    return None


def crossover(team1, team2):
    required_roles = {'opener': 2, 'middle_order': 2, 'finisher': 1, 'wicket_keeper': 1, 'spinner': 2, 'fast_bowler': 3}
    new_team = pd.DataFrame()

    for role, count in required_roles.items():
        candidates = pd.concat([team1[team1['role'] == role], team2[team2['role'] == role]]).drop_duplicates(subset='player_name')
        if len(candidates) < count:
            backup = player_pool[(player_pool['role'] == role) & (~player_pool['player_name'].isin(candidates['player_name']))]
            needed = count - len(candidates)
            candidates = pd.concat([candidates, backup.sample(min(needed, len(backup)))])
        selected = candidates.sample(min(count, len(candidates)))
        new_team = pd.concat([new_team, selected])

    fast_bowlers = new_team[new_team['role'] == 'fast_bowler']
    if len(fast_bowlers) < 3:
        needed = 3 - len(fast_bowlers)
        extras = player_pool[(player_pool['role'] == 'fast_bowler') & (~player_pool['player_name'].isin(new_team['player_name']))]
        if len(extras) >= needed:
            to_remove = new_team[~new_team['role'].isin(['wicket_keeper'])].sample(needed)
            new_team = new_team.drop(to_remove.index)
            new_team = pd.concat([new_team, extras.sample(needed)])

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


population_size = 50
generations = 50
population = [generate_random_team() for _ in range(population_size)]
population = [team for team in population if team is not None]

for gen in range(generations):
    scored = [(team, fitness(team)) for team in population]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_teams = [x[0] for x in scored[:10] if x[0] is not None]

    if len(top_teams) < 2:
        if len(scored) >= 2:
            top_teams = [x[0] for x in scored[:2]]
        elif len(scored) == 1:
            top_teams = [scored[0][0], scored[0][0]]
        else:
            print("No valid teams to continue evolution.")
            exit()

    new_gen = top_teams.copy()
    while len(new_gen) < population_size:
        t1, t2 = random.sample(top_teams, 2)
        child = crossover(t1, t2)
        if random.random() < 0.2:
            child = mutate(child)
        if is_valid_team(child):
            new_gen.append(child)
    population = new_gen

best_team = max(population, key=fitness)
best_team = apply_role_substitutions(best_team)
role_order = ['opener', 'middle_order', 'wicket_keeper', 'finisher', 'spinner', 'fast_bowler']
best_team_full_details = best_team.assign(role_order_index=best_team['role'].apply(lambda r: role_order.index(r))) \
                               .sort_values(by='role_order_index') \
                               .drop(columns='role_order_index') \
                               .reset_index(drop=True)
print(" Best Playing XI - Full Details:\n")
display_cols = [col for col in best_team_full_details.columns if not col.endswith('_lower')]
print(best_team_full_details[display_cols])
print("\n Total Fitness Score:", round(fitness(best_team), 2))
selected_players = set(best_team['player_name'].str.lower())
left_out_from_pool = set(player_pool['player_name_lower']) - selected_players
print(f"\n🪑 Players Left Out from Player Pool (Not in Final XI):")
print(", ".join(sorted([p.title() for p in left_out_from_pool])) if left_out_from_pool else "None")
