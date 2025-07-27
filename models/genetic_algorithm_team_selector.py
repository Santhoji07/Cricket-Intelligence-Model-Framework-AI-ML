import pandas as pd
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv")

# Standardize column names
df.columns = [col.strip().lower() for col in df.columns]
df = df.rename(columns={
    'player': 'player_name',
    'avg': 'bat_avg',
    'sr': 'bat_sr'
})

# Clean missing data
df = df.dropna(subset=['player_name'])
df.fillna(0, inplace=True)

# --- New Input ---
input_venue = input("Enter venue name: ").strip().lower()
input_team = input("Enter team name: ").strip().lower()

# --- Load team-to-player mapping ---
roles_df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv")
roles_df['player'] = roles_df['player'].str.strip()
roles_df['team'] = roles_df['team'].str.strip().str.lower()

# --- Get squad players from selected team ---
squad_names = roles_df[roles_df['team'] == input_team]['player'].str.lower().tolist()

# Filter player_pool based on venue AND squad
df['player_name_lower'] = df['player_name'].str.lower()
df['venue_lower'] = df['venue'].str.lower()

player_pool = pd.DataFrame()
for min_matches in [3, 2, 1, 0]:
    pool = df[
        (df['venue_lower'] == input_venue) &
        (df['player_name_lower'].isin(squad_names)) &
        (df['matches'] >= min_matches)
    ].copy()
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
role_needs = {
    'opener': 2,
    'middle_order': 2,
    'finisher': 1,
    'wicket_keeper': 1,
    'spinner': 2,
    'fast_bowler': 3
}
for role, required in role_needs.items():
    found = len(player_pool[player_pool['role'] == role])
    print(f"{role}: required={required}, found={found}")



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

    # Enforce spinner + fast_bowler total must be exactly 5
    spinner_count = role_counts.get('spinner', 0)
    fast_bowler_count = role_counts.get('fast_bowler', 0)
    if spinner_count + fast_bowler_count != 5:
        return False

    foreign_players = team[team['indian'].str.lower() != 'yes']
    if not (2 <= len(foreign_players) <= 4):
        return False

    return True

# Fitness function
def fitness(team):
    # Batting roles
    bat_roles = ['opener', 'middle_order', 'finisher', 'wicket_keeper']
    bat_score = team[team['role'].isin(bat_roles)]['bat_avg'].sum() \
              + (team[team['role'].isin(bat_roles)]['bat_sr'].sum() / 15)

    # Bowling roles
    bowl_roles = ['spinner', 'fast_bowler']
    bowl_score = team[team['role'].isin(bowl_roles)]['wickets'].sum() * 2 \
               - team[team['role'].isin(bowl_roles)]['econ'].sum()

    penalty = 100 if not is_valid_team(team) else 0
    return bat_score + bowl_score - penalty



# Generate valid random team

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
            team = pd.DataFrame()  # reset, try fallback
            break
        selected = candidates.sample(count)
        team = pd.concat([team, selected])

    # Fallback: fill randomly if not enough role-based players
    if team.empty:
        selected = player_pool.sample(11)
        team = selected

    # Enforce at least 3 fast bowlers
    fast_bowlers = team[team['role'] == 'fast_bowler']
    if len(fast_bowlers) < 3:
        needed = 3 - len(fast_bowlers)
        extras = player_pool[
            (player_pool['role'] == 'fast_bowler') &
            (~player_pool['player_name'].isin(team['player_name']))
        ]
        if len(extras) >= needed:
            to_remove = team[~team['role'].isin(['wicket_keeper'])].sample(needed)
            team = team.drop(to_remove.index)
            team = pd.concat([team, extras.sample(needed)])

            

    # Try to include 4 foreign players if they exist with valid roles
    current_foreign = team[team['indian'].str.lower() != 'yes']
    if len(current_foreign) < 4:
        needed = 4 - len(current_foreign)
        available_foreign = player_pool[
            (player_pool['indian'].str.lower() != 'yes') &
            (~player_pool['player_name'].isin(team['player_name']))
        ]
        
        available_foreign = available_foreign.sort_values(
            by=['wickets', 'bat_avg'], ascending=False
        )

        
        
        
  # shuffle

        for _, row in available_foreign.iterrows():
            replaceable = team[
                (team['indian'].str.lower() == 'yes') &
                (team['role'] == row['role'])
            ]
            if not replaceable.empty:
                team = team.drop(replaceable.iloc[0].name)
                team = pd.concat([team, pd.DataFrame([row])])
                needed -= 1
                if needed == 0:
                    break

    # Final check
    if len(team) == 11:
        return team.reset_index(drop=True)
    return None





# Crossover: Mix top 5 from parent1 + rest from parent2, refill to 11
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
        candidates = pd.concat([
            team1[team1['role'] == role],
            team2[team2['role'] == role]
        ]).drop_duplicates(subset='player_name')

        # If not enough candidates from parents, use from full player pool
        if len(candidates) < count:
            backup = player_pool[(player_pool['role'] == role) & (~player_pool['player_name'].isin(candidates['player_name']))]
            needed = count - len(candidates)
            candidates = pd.concat([candidates, backup.sample(min(needed, len(backup)))])
        
        selected = candidates.sample(min(count, len(candidates)))
        new_team = pd.concat([new_team, selected])

            # Ensure at least 3 fast bowlers after crossover
    fast_bowlers = new_team[new_team['role'] == 'fast_bowler']
    if len(fast_bowlers) < 3:
        needed = 3 - len(fast_bowlers)
        extras = player_pool[
            (player_pool['role'] == 'fast_bowler') &
            (~player_pool['player_name'].isin(new_team['player_name']))
        ]
        if len(extras) >= needed:
            to_remove = new_team[~new_team['role'].isin(['wicket_keeper'])].sample(needed)
            new_team = new_team.drop(to_remove.index)
            new_team = pd.concat([new_team, extras.sample(needed)])


    return new_team.reset_index(drop=True)



# Mutation: Replace one player randomly
def mutate(team):
    team = team.sample(frac=1).reset_index(drop=True)

    for i in range(len(team)):
        role = team.iloc[i]['role']
        current_names = team['player_name'].tolist()

        replacements = player_pool[
            (player_pool['role'] == role) & 
            (~player_pool['player_name'].isin(current_names))
        ]

        if not replacements.empty:
            new_player = replacements.sample(1).iloc[0]
            team.iloc[i] = new_player
            break  # mutate only once

    return team.reset_index(drop=True)



# --- Genetic Algorithm Execution ---

population_size = 50
generations = 50
population = [generate_random_team() for _ in range(population_size)]
population = [team for team in population if team is not None]

for gen in range(generations):
    scored = [(team, fitness(team)) for team in population]
    scored.sort(key=lambda x: x[1], reverse=True)
    #top_teams = [x[0] for x in scored[:10]]

    top_teams = [x[0] for x in scored[:10] if x[0] is not None]

# Fallbacks if not enough top teams
if len(top_teams) < 2:
    if len(scored) >= 2:
        top_teams = [x[0] for x in scored[:2]]  # take from population
    elif len(scored) == 1:
        top_teams = [scored[0][0], scored[0][0]]  # duplicate one team
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

# Final best team output
best_team = max(population, key=fitness)
# Define preferred batting/bowling order
role_order = ['opener', 'middle_order', 'wicket_keeper', 'finisher', 'spinner', 'fast_bowler']

# Sort based on role order
best_team_full_details = (
    best_team.assign(role_order_index=best_team['role'].apply(lambda r: role_order.index(r)))
             .sort_values(by='role_order_index')
             .drop(columns='role_order_index')
             .reset_index(drop=True)
)


print(" Best Playing XI - Full Details:\n")
display_cols = [col for col in best_team_full_details.columns if not col.endswith('_lower')]
print(best_team_full_details[display_cols])

print("\n Total Fitness Score:", round(fitness(best_team), 2))

# Players from player pool not selected in Playing XI
selected_players = set(best_team['player_name'].str.lower())
left_out_from_pool = player_pool[~player_pool['player_name_lower'].isin(selected_players)]

print(f"\n🪑 Players Left Out from Player Pool (Not in Final XI): {len(left_out_from_pool)} players\n")
if not left_out_from_pool.empty:
    display_cols = [col for col in left_out_from_pool.columns if not col.endswith('_lower')]
    print(left_out_from_pool[display_cols].sort_values(by="role").reset_index(drop=True))
else:
    print("None")


# Optional: save to CSV
# best_team_full_details.to_csv("best_playing_xi.csv", index=False)


#constrains issue'''

'''import pandas as pd
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv")

# Standardize column names
df.columns = [col.strip().lower() for col in df.columns]
df = df.rename(columns={
    'player': 'player_name',
    'avg': 'bat_avg',
    'sr': 'bat_sr'
})

# Clean missing data
df = df.dropna(subset=['player_name'])
df.fillna(0, inplace=True)

# --- New Input ---
input_venue = input("Enter venue name: ").strip().lower()
input_team = input("Enter team name: ").strip().lower()

# --- Load team-to-player mapping ---
roles_df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv")
roles_df['player'] = roles_df['player'].str.strip()
roles_df['team'] = roles_df['team'].str.strip().str.lower()

# --- Get squad players from selected team ---
squad_names = roles_df[roles_df['team'] == input_team]['player'].str.lower().tolist()

# Filter player_pool based on venue AND squad
df['player_name_lower'] = df['player_name'].str.lower()
df['venue_lower'] = df['venue'].str.lower()

player_pool = pd.DataFrame()
for min_matches in [3, 2, 1, 0]:
    pool = df[
        (df['venue_lower'] == input_venue) &
        (df['player_name_lower'].isin(squad_names)) &
        (df['matches'] >= min_matches)
    ].copy()
    if len(pool) >= 11:
        print(f"✅ Player pool created with min_matches >= {min_matches}")
        player_pool = pool
        break

if player_pool.empty:
    print("❌ No players found from this team at this venue.")
    exit()

pool_names_lower = set(player_pool['player_name_lower'].tolist())
squad_not_in_pool = set(squad_names) - pool_names_lower

print(f"\n⚠️ Players from Squad NOT in Player Pool (filtered out by venue or matches):")
print(", ".join(sorted([p.title() for p in squad_not_in_pool])) if squad_not_in_pool else "None")

print("\n🧾 Player pool summary:")
print(player_pool[['player_name', 'role', 'indian']])
print("\n📊 Role counts:")
print(player_pool['role'].value_counts())
print("\n🌏 Foreign player count:", len(player_pool[player_pool['indian'].str.lower() != 'yes']))
print("🇮🇳 Indian player count:", len(player_pool[player_pool['indian'].str.lower() == 'yes']))

print("\n🔍 Role availability check:")
role_needs = {
    'opener': 2,
    'middle_order': 2,
    'finisher': 1,
    'wicket_keeper': 1,
    'spinner': 2,
    'fast_bowler': 3
}
for role, required in role_needs.items():
    found = len(player_pool[player_pool['role'] == role])
    print(f"{role}: required={required}, found={found}")


# ✅ --- VALIDATION FUNCTION ---
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

    # Substitution rules
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

    # Ensure one wicket keeper
    if role_counts.get('wicket_keeper', 0) < 1:
        return False

    # Spinner + fast bowler total must be exactly 5
    spinner_count = role_counts.get('spinner', 0)
    fast_bowler_count = role_counts.get('fast_bowler', 0)
    if spinner_count + fast_bowler_count != 5:
        return False

    # Foreign player rule
    foreign_players = team[team['indian'].str.lower() != 'yes']
    if not (2 <= len(foreign_players) <= 4):
        return False

    return True


# ✅ --- FITNESS FUNCTION ---
def fitness(team):
    bat_roles = ['opener', 'middle_order', 'finisher', 'wicket_keeper']
    bat_score = team[team['role'].isin(bat_roles)]['bat_avg'].sum() \
              + (team[team['role'].isin(bat_roles)]['bat_sr'].sum() / 15)

    bowl_roles = ['spinner', 'fast_bowler']
    bowl_score = team[team['role'].isin(bowl_roles)]['wickets'].sum() * 2 \
               - team[team['role'].isin(bowl_roles)]['econ'].sum()

    penalty = 100 if not is_valid_team(team) else 0
    return bat_score + bowl_score - penalty


# ✅ --- RANDOM TEAM GENERATOR ---
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
            team = pd.DataFrame()  # reset if role shortage
            break
        selected = candidates.sample(count)
        team = pd.concat([team, selected])

    # Fallback: pick any 11 players if role allocation failed
    if team.empty:
        team = player_pool.sample(11)

    return team.reset_index(drop=True)


# ✅ --- CROSSOVER FUNCTION ---
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
        candidates = pd.concat([
            team1[team1['role'] == role],
            team2[team2['role'] == role]
        ]).drop_duplicates(subset='player_name')

        # Fill gaps with pool players
        if len(candidates) < count:
            backup = player_pool[(player_pool['role'] == role) & (~player_pool['player_name'].isin(candidates['player_name']))]
            candidates = pd.concat([candidates, backup.sample(min(count - len(candidates), len(backup)))])

        selected = candidates.sample(min(count, len(candidates)))
        new_team = pd.concat([new_team, selected])

    return new_team.reset_index(drop=True)


# ✅ --- MUTATION FUNCTION ---
def mutate(team):
    team = team.sample(frac=1).reset_index(drop=True)

    for i in range(len(team)):
        role = team.iloc[i]['role']
        current_names = team['player_name'].tolist()

        replacements = player_pool[
            (player_pool['role'] == role) &
            (~player_pool['player_name'].isin(current_names))
        ]

        if not replacements.empty:
            new_player = replacements.sample(1).iloc[0]
            team.iloc[i] = new_player
            break

    return team.reset_index(drop=True)


# ✅ --- GENETIC ALGORITHM EXECUTION ---
population_size = 50
generations = 50
population = [generate_random_team() for _ in range(population_size)]
population = [team for team in population if team is not None]

for gen in range(generations):
    scored = [(team, fitness(team)) for team in population]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_teams = [x[0] for x in scored[:10] if x[0] is not None]

    if len(top_teams) < 2:
        print("⚠️ Not enough valid teams to continue evolution.")
        break

    new_gen = top_teams.copy()
    while len(new_gen) < population_size:
        t1, t2 = random.sample(top_teams, 2)
        child = crossover(t1, t2)
        if random.random() < 0.2:
            child = mutate(child)
        if is_valid_team(child):
            new_gen.append(child)

    population = new_gen

# ✅ --- FINAL BEST TEAM OUTPUT ---
best_team = max(population, key=fitness)

# Preferred batting/bowling order for display
role_order = ['opener', 'middle_order', 'wicket_keeper', 'finisher', 'spinner', 'fast_bowler']
best_team['role_order'] = best_team['role'].apply(lambda r: role_order.index(r))
best_team = best_team.sort_values(by='role_order')

print("\n🏏 ✅ FINAL BEST XI for", input_team.upper(), "at", input_venue.title(), "🏟️")
print(best_team[['player_name', 'role', 'bat_avg', 'bat_sr', 'wickets', 'econ', 'indian']].to_string(index=False))'''
