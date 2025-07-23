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

# Choose the most common venue

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

# Try multiple match thresholds: >=3, >=2, >=1, >=0
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

    # Players from the squad not in the player pool
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


# --- Helper Functions ---

# Validate team constraints strictly

def is_valid_team(team):
    if team is None or len(team) != 11:
        return False

    role_counts = team['role'].value_counts().to_dict()

    # Original role requirements
    required_roles = {
        'opener': 2,
        'middle_order': 2,
        'finisher': 1,
        'wicket_keeper': 1,
        'spinner': 2,
        'fast_bowler': 3
    }

    # Clone to modify for substitutions
    role_counts_flexible = role_counts.copy()

    # Substitute logic
    # If opener < 2, try using middle_order
    if role_counts_flexible.get('opener', 0) < 2:
        short = 2 - role_counts_flexible.get('opener', 0)
        role_counts_flexible['middle_order'] = role_counts_flexible.get('middle_order', 0) - short

    # If middle_order < 2, try using finisher
    if role_counts_flexible.get('middle_order', 0) < 2:
        short = 2 - role_counts_flexible.get('middle_order', 0)
        role_counts_flexible['finisher'] = role_counts_flexible.get('finisher', 0) - short

    # If finisher < 1, try using middle_order
    if role_counts_flexible.get('finisher', 0) < 1:
        short = 1 - role_counts_flexible.get('finisher', 0)
        role_counts_flexible['middle_order'] = role_counts_flexible.get('middle_order', 0) - short

    # If spinner < 2, try using fast_bowler
    if role_counts_flexible.get('spinner', 0) < 2:
        short = 2 - role_counts_flexible.get('spinner', 0)
        role_counts_flexible['fast_bowler'] = role_counts_flexible.get('fast_bowler', 0) - short

    # If fast_bowler < 3, try using spinner
    if role_counts_flexible.get('fast_bowler', 0) < 3:
        short = 3 - role_counts_flexible.get('fast_bowler', 0)
        role_counts_flexible['spinner'] = role_counts_flexible.get('spinner', 0) - short

    # Final check after substitutions
    try:
        if role_counts_flexible.get('wicket_keeper', 0) < 1:
            return False
        if role_counts_flexible.get('opener', 0) < 2:
            return False
        if role_counts_flexible.get('middle_order', 0) < 0:
            return False
        if role_counts_flexible.get('finisher', 0) < 0:
            return False
        if role_counts_flexible.get('spinner', 0) < 0:
            return False
        if role_counts_flexible.get('fast_bowler', 0) < 0:
            return False
    except:
        return False

    # Relaxed: allow 2–4 foreign players
    foreign_players = team[team['indian'].str.lower() != 'yes']
    if not (2 <= len(foreign_players) <= 4):
        return False

    return True'''

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

    # Select top-performing players for each role regardless of nationality
    for role, count in required_roles.items():
        candidates = player_pool[player_pool['role'] == role].copy()

        # Sort based on performance: batting roles use bat_avg + bat_sr, bowling uses wickets + econ
        if role in ['opener', 'middle_order', 'finisher', 'wicket_keeper']:
            candidates['performance'] = candidates['bat_avg'] + candidates['bat_sr'] / 10
        else:
            candidates['performance'] = candidates['wickets'] * 2 - candidates['econ']

        top_candidates = candidates.sort_values(by='performance', ascending=False).head(count)

        # If not enough candidates, fallback to random
        if len(top_candidates) < count:
            missing = count - len(top_candidates)
            filler = candidates[~candidates['player_name'].isin(top_candidates['player_name'])].sample(missing)
            top_candidates = pd.concat([top_candidates, filler])

        team = pd.concat([team, top_candidates])

    team = team.drop_duplicates(subset='player_name')

    # Enforce 4 foreign players if possible with role balance
    current_foreign = team[team['indian'].str.lower() != 'yes']
    if len(current_foreign) < 4:
        needed = 4 - len(current_foreign)
        foreign_pool = player_pool[
            (player_pool['indian'].str.lower() != 'yes') &
            (~player_pool['player_name'].isin(team['player_name']))
        ].copy()

        foreign_pool['performance'] = np.where(
            foreign_pool['role'].isin(['opener', 'middle_order', 'finisher', 'wicket_keeper']),
            foreign_pool['bat_avg'] + foreign_pool['bat_sr'] / 10,
            foreign_pool['wickets'] * 2 - foreign_pool['econ']
        )

        foreign_pool = foreign_pool.sort_values(by='performance', ascending=False)

        for _, row in foreign_pool.iterrows():
            # Replace a weaker Indian player with same role
            replaceable = team[
                (team['indian'].str.lower() == 'yes') &
                (team['role'] == row['role'])
            ].sort_values(by='bat_avg' if row['role'] in ['opener', 'middle_order', 'finisher', 'wicket_keeper'] else 'wickets')

            if not replaceable.empty:
                team = team.drop(replaceable.iloc[0].name)
                team = pd.concat([team, pd.DataFrame([row])])
                needed -= 1
            if needed == 0:
                break

    # Final validation
    team = team.sample(frac=1).reset_index(drop=True)  # shuffle
    return team if is_valid_team(team) else None






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


#constrains issue 