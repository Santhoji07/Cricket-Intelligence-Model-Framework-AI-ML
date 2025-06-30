
import pandas as pd
import random

# Load dataset
df = pd.read_csv("D:/Cricket Project CIM model/CIM/data/player_stats_venue.csv")

# Standardize and rename columns
df.columns = [col.strip().lower() for col in df.columns]
df = df.rename(columns={
    'player': 'player_name',
    'avg': 'bat_avg',
    'sr': 'bat_sr'
})

# Clean data
df = df.dropna(subset=['player_name'])
df.fillna(0, inplace=True)

# Choose the most common venue
target_venue = df['venue'].mode()[0]
player_pool = df[df['venue'] == target_venue].copy()

# --- Helper Functions ---

# Validate team constraints
def is_valid_team(team):
    if len(team) != 11:
        return False
    roles = team['role'].value_counts()
    indians = team[team['indian'].str.lower() == 'yes']
    wks = team[team['role'].str.lower().str.contains("wicket")]
    bowlers = team[team['role'].isin(['fast_bowler', 'spinner'])]
    return len(indians) >= 7 and len(wks) >= 1 and len(bowlers) >= 5

# Define fitness function
def fitness(team):
    batting = team['bat_avg'].sum() + team['bat_sr'].sum() / 10
    bowling = team['wickets'].sum() * 2 - team['econ'].sum()
    penalty = 100 if not is_valid_team(team) else 0
    return batting + bowling - penalty

# Generate a random team
def generate_random_team():
    for _ in range(1000):
        team = player_pool.sample(11)
        if is_valid_team(team):
            return team
    return None

# Crossover and mutation
def crossover(team1, team2):
    combined = pd.concat([team1.iloc[:5], team2.iloc[5:]]).drop_duplicates()
    while len(combined) < 11:
        candidate = player_pool[~player_pool['player_name'].isin(combined['player_name'])].sample(1)
        combined = pd.concat([combined, candidate])
    return combined

def mutate(team):
    team = team.sample(frac=1).reset_index(drop=True)
    new_player = player_pool[~player_pool['player_name'].isin(team['player_name'])].sample(1)
    team.iloc[0] = new_player.iloc[0]
    return team

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

# Final best team with all attributes
best_team = max(population, key=fitness)

# Reset index and display all columns
best_team_full_details = best_team.reset_index(drop=True)
print("🏏 Best Playing XI - Full Details:\n")
print(best_team_full_details)

# Optional: Save to CSV
# best_team_full_details.to_csv("best_playing_xi_full_details.csv", index=False)

