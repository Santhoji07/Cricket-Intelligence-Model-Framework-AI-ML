import pandas as pd
import random
import numpy as np

random.seed(42)
np.random.seed(42)

class CricketTeamGA:
    def __init__(self, stats_file, roles_file):
        # Load datasets
        self.df_stats = pd.read_csv(stats_file)
        self.df_roles = pd.read_csv(roles_file)

        # Standardize columns in stats dataset
        self.df_stats.columns = [col.strip().lower() for col in self.df_stats.columns]

        # Rename batting average and strike rate if needed
        self.df_stats = self.df_stats.rename(columns={
            'player_name': 'player_name',  # keep as is
            'avg': 'bat_avg',
            'sr': 'bat_sr'
        })

        # Drop rows with missing player_name and fill NA
        self.df_stats = self.df_stats.dropna(subset=['player_name'])
        self.df_stats.fillna(0, inplace=True)

        # Standardize roles dataset
        self.df_roles['player_name'] = self.df_roles['player_name'].str.strip()
        self.df_roles['franchise'] = self.df_roles['franchise'].str.strip().str.lower()

        # Define required roles and constraints
        self.required_roles = {
            'opener': 2,
            'middle_order': 2,
            'finisher': 1,
            'wicket_keeper': 1,
            'spinner': 2,
            'fast_bowler': 3
        }
        self.population_size = 50
        self.generations = 50
        self.mutation_rate = 0.2

        self.player_pool = pd.DataFrame()

    def prepare_player_pool(self, input_team, input_venue):
        # Lowercase for matching
        squad_names = self.df_roles[self.df_roles['franchise'] == input_team]['player_name'].str.lower().tolist()
        self.df_stats['player_name_lower'] = self.df_stats['player_name'].str.lower()
        self.df_stats['venue_lower'] = self.df_stats['venue'].str.lower()

        # Filter player pool by venue, squad, and minimum matches
        player_pool = pd.DataFrame()
        for min_matches in [3, 2, 1, 0]:
            pool = self.df_stats[
                (self.df_stats['venue_lower'] == input_venue) &
                (self.df_stats['player_name_lower'].isin(squad_names)) &
                (self.df_stats['matches'] >= min_matches)
            ].copy()
            if len(pool) >= 11:
                player_pool = pool
                break

        if player_pool.empty:
            raise ValueError(f"No sufficient players found for franchise '{input_team}' at venue '{input_venue}'")

        self.player_pool = player_pool.copy()
        self.player_pool['role'] = self.player_pool['role'].fillna('unknown').str.lower()
        self.player_pool['indian'] = self.player_pool['indian'].fillna('yes').str.lower()
        self.player_pool.reset_index(drop=True, inplace=True)

    def is_valid_team(self, team):
        if team is None or len(team) != 11:
            return False

        role_counts = team['role'].value_counts().to_dict()
        required_roles = self.required_roles.copy()

        # Flexible role substitutions for balancing
        for role, required in required_roles.items():
            available = role_counts.get(role, 0)
            if available < required:
                if role == 'opener' and role_counts.get('middle_order', 0) >= required - available:
                    role_counts['middle_order'] -= (required - available)
                elif role == 'middle_order' and role_counts.get('finisher', 0) >= required - available:
                    role_counts['finisher'] -= (required - available)
                elif role == 'finisher' and role_counts.get('middle_order', 0) >= required - available:
                    role_counts['middle_order'] -= (required - available)
                elif role == 'spinner' and role_counts.get('fast_bowler', 0) >= required - available:
                    role_counts['fast_bowler'] -= (required - available)
                elif role == 'fast_bowler' and role_counts.get('spinner', 0) >= required - available:
                    role_counts['spinner'] -= (required - available)
                else:
                    return False

        if role_counts.get('wicket_keeper', 0) < 1:
            return False

        # Exactly 5 total of spinners and fast bowlers
        spinner_count = role_counts.get('spinner', 0)
        fast_bowler_count = role_counts.get('fast_bowler', 0)
        if spinner_count + fast_bowler_count != 5:
            return False

        foreign_players = team[team['indian'] != 'yes']
        if not (2 <= len(foreign_players) <= 4):
            return False

        return True

    def fitness(self, team):
        bat_roles = ['opener', 'middle_order', 'finisher', 'wicket_keeper']
        bowl_roles = ['spinner', 'fast_bowler']

        bat_score = team[team['role'].isin(bat_roles)]['bat_avg'].sum() + \
                    team[team['role'].isin(bat_roles)]['bat_sr'].sum() / 15
        bowl_score = team[team['role'].isin(bowl_roles)]['wickets'].sum() * 2 - \
                     team[team['role'].isin(bowl_roles)]['econ'].sum()
        penalty = 100 if not self.is_valid_team(team) else 0

        return bat_score + bowl_score - penalty

    def generate_random_team(self):
        team = pd.DataFrame()

        for role, count in self.required_roles.items():
            candidates = self.player_pool[self.player_pool['role'] == role]
            if len(candidates) < count:
                return None
            selected = candidates.sample(count)
            team = pd.concat([team, selected])

        fast_bowlers = team[team['role'] == 'fast_bowler']
        if len(fast_bowlers) < 3:
            needed = 3 - len(fast_bowlers)
            extras = self.player_pool[
                (self.player_pool['role'] == 'fast_bowler') &
                (~self.player_pool['player_name'].isin(team['player_name']))
            ]
            if len(extras) >= needed:
                to_remove = team[~team['role'].isin(['wicket_keeper'])].sample(needed)
                team = team.drop(to_remove.index)
                team = pd.concat([team, extras.sample(needed)])

        current_foreign = team[team['indian'] != 'yes']
        if len(current_foreign) < 4:
            needed = 4 - len(current_foreign)
            available_foreign = self.player_pool[
                (self.player_pool['indian'] != 'yes') &
                (~self.player_pool['player_name'].isin(team['player_name']))
            ].sort_values(by=['wickets', 'bat_avg'], ascending=False)

            for _, foreign_player in available_foreign.iterrows():
                replaceable = team[
                    (team['indian'] == 'yes') & (team['role'] == foreign_player['role'])
                ]
                if not replaceable.empty:
                    team = team.drop(replaceable.iloc[0].name)
                    team = pd.concat([team, pd.DataFrame([foreign_player])])
                    needed -= 1
                    if needed == 0:
                        break

        if len(team) == 11:
            return team.reset_index(drop=True)
        return None

    def crossover(self, team1, team2):
        new_team = pd.DataFrame()
        for role, count in self.required_roles.items():
            candidates = pd.concat([
                team1[team1['role'] == role],
                team2[team2['role'] == role]
            ]).drop_duplicates('player_name')
            if len(candidates) < count:
                backup = self.player_pool[
                    (self.player_pool['role'] == role) &
                    (~self.player_pool['player_name'].isin(candidates['player_name']))
                ]
                candidates = pd.concat([candidates, backup.sample(min(count - len(candidates), len(backup)))])
            selected = candidates.sample(min(count, len(candidates)))
            new_team = pd.concat([new_team, selected])

        fast_bowlers = new_team[new_team['role'] == 'fast_bowler']
        if len(fast_bowlers) < 3:
            needed = 3 - len(fast_bowlers)
            extras = self.player_pool[
                (self.player_pool['role'] == 'fast_bowler') &
                (~self.player_pool['player_name'].isin(new_team['player_name']))
            ]
            if len(extras) >= needed:
                to_remove = new_team[~new_team['role'].isin(['wicket_keeper'])].sample(needed)
                new_team = new_team.drop(to_remove.index)
                new_team = pd.concat([new_team, extras.sample(needed)])

        return new_team.reset_index(drop=True)

    def mutate(self, team):
        team = team.sample(frac=1).reset_index(drop=True)
        for i in range(len(team)):
            role = team.iloc[i]['role']
            current_names = team['player_name'].tolist()
            replacements = self.player_pool[
                (self.player_pool['role'] == role) &
                (~self.player_pool['player_name'].isin(current_names))
            ]
            if not replacements.empty:
                new_player = replacements.sample(1).iloc[0]
                team.iloc[i] = new_player
                break  # only mutate once
        return team.reset_index(drop=True)

    def run_ga(self, input_team, input_venue):
        self.prepare_player_pool(input_team, input_venue)
        population = [self.generate_random_team() for _ in range(self.population_size)]
        population = [t for t in population if t is not None]

        for gen in range(self.generations):
            scored = [(t, self.fitness(t)) for t in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_teams = [t for t, _ in scored[:10] if t is not None]

            if len(top_teams) < 2:
                if len(scored) >= 2:
                    top_teams = [t for t, _ in scored[:2]]
                elif len(scored) == 1:
                    top_teams = [scored[0][0], scored[0][0]]
                else:
                    raise ValueError("No valid teams to continue evolution.")

            new_gen = top_teams.copy()
            while len(new_gen) < self.population_size:
                t1, t2 = random.sample(top_teams, 2)
                child = self.crossover(t1, t2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                if self.is_valid_team(child):
                    new_gen.append(child)

            population = new_gen

        best_team = max(population, key=self.fitness)
        return best_team
