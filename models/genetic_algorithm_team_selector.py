
import pandas as pd
import random

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
'''target_venue = df['venue'].mode()[0]
player_pool = df[(df['venue'] == target_venue) & (df['matches'] >= 2)].copy()'''
# --- Accept Input ---
input_venue = input("Enter venue name: ").strip().lower()

input_squad = input("Enter comma-separated player names in squad: ")
squad_names = [name.strip().lower() for name in input_squad.split(",")]

# Filter player_pool based on venue AND squad
df['player_name_lower'] = df['player_name'].str.lower()
df['venue_lower'] = df['venue'].str.lower()

player_pool = df[
    (df['venue_lower'] == input_venue) &
    (df['matches'] >= 0) &
    (df['player_name_lower'].isin(squad_names))
].copy()

if player_pool.empty:
    print("No valid players found for this venue and squad.")
    exit()

# --- Helper Functions ---

# Validate team constraints strictly
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

    for role, count in required_roles.items():
        if role_counts.get(role, 0) != count:
            return False

    #  Foreign player constraint: at least 4 foreign
    foreign_players = team[team['indian'].str.lower() != 'yes']
    if len(foreign_players) != 4:
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
    for _ in range(1000):
        team = player_pool.sample(11)
        if is_valid_team(team):
            return team
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
    top_teams = [x[0] for x in scored[:10]]

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
print(best_team_full_details)
print("\n Total Fitness Score:", round(fitness(best_team), 2))

# Optional: save to CSV
# best_team_full_details.to_csv("best_playing_xi.csv", index=False)
