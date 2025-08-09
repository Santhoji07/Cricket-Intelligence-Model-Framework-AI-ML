import pandas as pd
import random
import numpy as np

random.seed(42)
np.random.seed(42)

class CricketTeamGA:
    def __init__(self, stats_file, roles_file):
        self.df_stats = pd.read_csv(stats_file)
        self.df_roles = pd.read_csv(roles_file)

        # Standardize column names
        self.df_stats.columns = [c.strip().lower() for c in self.df_stats.columns]
        self.df_stats = self.df_stats.rename(columns={'avg': 'bat_avg', 'sr': 'bat_sr'})
        self.df_stats.dropna(subset=['player_name'], inplace=True)
        self.df_stats.fillna(0, inplace=True)

        self.df_roles['player_name'] = self.df_roles['player_name'].str.strip()
        self.df_roles['franchise'] = self.df_roles['franchise'].str.strip().str.lower()

        self.required_roles = {
            'opener': 2,
            'middle_order': 2,
            'finisher': 1,
            'wicket_keeper': 1,
            'spinner': 2,
            'fast_bowler': 3
        }
        self.population_size = 20
        self.generations = 20
        self.mutation_rate = 0.2

    def _aggregate_pool(self, df):
        if df.empty:
            return df
        return df.groupby('player_name', as_index=False).agg({
            'role': 'first',
            'matches': 'sum',
            'runs': 'sum',
            'bat_avg': 'mean',
            'bat_sr': 'mean',
            'wickets': 'sum',
            'econ': 'mean',
            'indian': 'first'
        })

    def prepare_player_pool(self, input_team, input_venue, allow_fallback=True):
        self.franchise_list = self.df_roles[self.df_roles['franchise'] == input_team]
        squad_names = self.franchise_list['player_name'].str.lower().tolist()

        self.df_stats['player_name_lower'] = self.df_stats['player_name'].str.lower()
        self.df_stats['venue_lower'] = self.df_stats['venue'].str.lower()

        player_pool = pd.DataFrame()
        self.min_matches_used = None
        for min_matches in [3, 2, 1, 0]:
            pool = self.df_stats[
                (self.df_stats['venue_lower'] == input_venue) &
                (self.df_stats['player_name_lower'].isin(squad_names)) &
                (self.df_stats['matches'] >= min_matches)
            ].copy()
            if len(pool) >= 11:
                player_pool = pool
                self.min_matches_used = min_matches
                break

        if not player_pool.empty:
            player_pool = self._aggregate_pool(player_pool)

        if (player_pool.empty or len(player_pool) < 11) and allow_fallback:
            self.prepare_player_pool_full_squad(input_team)
            return

        pool_names_lower = set(player_pool['player_name'].str.lower())
        self.squad_not_in_pool = sorted(set(squad_names) - pool_names_lower)

        self.player_pool = player_pool.copy()
        self.player_pool['role'] = self.player_pool['role'].fillna('unknown').str.lower()
        self.player_pool['indian'] = self.player_pool['indian'].fillna('yes').str.lower()
        self.role_counts = self.player_pool['role'].value_counts()

    def prepare_player_pool_full_squad(self, input_team):
        self.franchise_list = self.df_roles[self.df_roles['franchise'] == input_team]
        squad_names = self.franchise_list['player_name'].str.lower().tolist()
        self.df_stats['player_name_lower'] = self.df_stats['player_name'].str.lower()

        # Filter only players with >=5 matches in fallback mode
        squad_stats = self.df_stats[
            self.df_stats['player_name_lower'].isin(squad_names) &
            (self.df_stats['matches'] >= 5)
        ].copy()

        aggregated = self._aggregate_pool(squad_stats)

        self.player_pool = aggregated
        self.min_matches_used = "Squad fallback (aggregated)"
        self.role_counts = self.player_pool['role'].value_counts()
        self.squad_not_in_pool = []

    def is_valid_team(self, team):
        if team is None or len(team) != 11:
            return False
        rc = team['role'].value_counts().to_dict()
        if rc.get('spinner', 0) + rc.get('fast_bowler', 0) != 5:
            return False
        foreign_players = team[team['indian'] != 'yes']
        return 2 <= len(foreign_players) <= 4

    def fitness(self, team):
        bat_roles = ['opener', 'middle_order', 'finisher', 'wicket_keeper']
        bowl_roles = ['spinner', 'fast_bowler']
        bat_score = team[team['role'].isin(bat_roles)]['bat_avg'].sum() + \
                    team[team['role'].isin(bat_roles)]['bat_sr'].sum() / 15
        bowl_score = team[team['role'].isin(bowl_roles)]['wickets'].sum() * 2 - \
                     team[team['role'].isin(bowl_roles)]['econ'].sum()
        return bat_score + bowl_score - (0 if self.is_valid_team(team) else 100)

    def _pick_candidates(self, role, count, fallback, used):
        """Helper to pick specialists for bowling roles."""
        candidates = self.player_pool[(self.player_pool['role'] == role) & (~self.player_pool['player_name'].isin(used))]
        if role in ['fast_bowler', 'spinner']:
            sort_cols, ascending_order = ['wickets', 'econ'], [False, True]
        else:
            sort_cols, ascending_order = ['bat_avg', 'bat_sr', 'wickets'], [False, False, False]

        if len(candidates) >= count:
            chosen = candidates.sort_values(by=sort_cols, ascending=ascending_order).head(count)
        else:
            chosen = candidates
            shortfall = count - len(candidates)
            if fallback:
                fb_candidates = self.player_pool[(self.player_pool['role'] == fallback) & (~self.player_pool['player_name'].isin(used))]
                fb_candidates = fb_candidates.sort_values(by=sort_cols, ascending=ascending_order)
                chosen = pd.concat([chosen, fb_candidates.head(shortfall)])
        return chosen

    def generate_random_team(self):
        team = pd.DataFrame()
        used = set()
        for role, count, fallback in [
            ('opener', 2, 'middle_order'),
            ('middle_order', 2, 'finisher'),
            ('wicket_keeper', 1, 'middle_order'),
            ('finisher', 1, 'middle_order'),
            ('fast_bowler', 3, 'spinner'),
            ('spinner', 2, 'fast_bowler')
        ]:
            picks = self._pick_candidates(role, count, fallback, used)
            team = pd.concat([team, picks])
            used.update(picks['player_name'])

        if len(team) < 11:
            fillers = self.player_pool[~self.player_pool['player_name'].isin(used)].sort_values(
                by=['matches', 'wickets', 'bat_avg'], ascending=False).head(11 - len(team))
            team = pd.concat([team, fillers])

        team = team.drop_duplicates('player_name').head(11).reset_index(drop=True)
        return team if self.is_valid_team(team) else None

    def crossover(self, t1, t2):
        team = pd.DataFrame()
        used = set()
        for role, count, fallback in [
            ('opener', 2, 'middle_order'),
            ('middle_order', 2, 'finisher'),
            ('wicket_keeper', 1, 'middle_order'),
            ('finisher', 1, 'middle_order'),
            ('fast_bowler', 3, 'spinner'),
            ('spinner', 2, 'fast_bowler')
        ]:
            candidates = pd.concat([
                t1[(t1['role'] == role) & (~t1['player_name'].isin(used))],
                t2[(t2['role'] == role) & (~t2['player_name'].isin(used))]
            ]).drop_duplicates('player_name')

            if role in ['fast_bowler', 'spinner']:
                candidates = candidates.sort_values(by=['wickets', 'econ'], ascending=[False, True])
            else:
                candidates = candidates.sort_values(by=['bat_avg', 'bat_sr', 'wickets'], ascending=[False, False, False])

            if len(candidates) >= count:
                chosen = candidates.head(count)
            else:
                chosen = candidates
                shortfall = count - len(candidates)
                if fallback:
                    fb = pd.concat([
                        t1[(t1['role'] == fallback) & (~t1['player_name'].isin(used))],
                        t2[(t2['role'] == fallback) & (~t2['player_name'].isin(used))]
                    ]).drop_duplicates('player_name')
                    fb = fb.sort_values(by=['wickets', 'econ'] if role in ['fast_bowler', 'spinner'] else ['bat_avg', 'bat_sr', 'wickets'],
                                        ascending=[False, True] if role in ['fast_bowler', 'spinner'] else [False, False, False])
                    chosen = pd.concat([chosen, fb.head(shortfall)])
            team = pd.concat([team, chosen])
            used.update(chosen['player_name'])

        if len(team) < 11:
            fillers = self.player_pool[~self.player_pool['player_name'].isin(used)].sort_values(
                by=['matches', 'wickets', 'bat_avg'], ascending=False).head(11 - len(team))
            team = pd.concat([team, fillers])

        return team.drop_duplicates('player_name').head(11).reset_index(drop=True)

    def mutate(self, team):
        idx = random.randint(0, len(team) - 1)
        role = team.iloc[idx]['role']
        current_names = team['player_name'].tolist()
        replacements = self.player_pool[(self.player_pool['role'] == role) & (~self.player_pool['player_name'].isin(current_names))]
        if role in ['fast_bowler', 'spinner']:
            replacements = replacements.sort_values(by=['wickets', 'econ'], ascending=[False, True])
        else:
            replacements = replacements.sort_values(by=['bat_avg', 'bat_sr', 'wickets'], ascending=[False, False, False])
        if not replacements.empty:
            team.iloc[idx] = replacements.head(1).iloc[0]
        return team.reset_index(drop=True)

    def select_best_statistical_xi(self):
        pool = self.player_pool.copy()
        selected = pd.DataFrame()
        used = set()
        for role, count, fallback in [
            ('opener', 2, 'middle_order'),
            ('middle_order', 2, 'finisher'),
            ('wicket_keeper', 1, 'middle_order'),
            ('finisher', 1, 'middle_order'),
            ('fast_bowler', 3, 'spinner'),
            ('spinner', 2, 'fast_bowler')
        ]:
            if role in ['fast_bowler', 'spinner']:
                sort_cols, asc_order = ['wickets', 'econ'], [False, True]
            else:
                sort_cols, asc_order = ['bat_avg', 'bat_sr', 'wickets'], [False, False, False]
            candidates = pool[(pool['role'] == role) & (~pool['player_name'].isin(used))]
            if len(candidates) >= count:
                picks = candidates.sort_values(by=sort_cols, ascending=asc_order).head(count)
            else:
                picks = candidates
                shortfall = count - len(picks)
                fb_candidates = pool[(pool['role'] == fallback) & (~pool['player_name'].isin(used))]
                fb_candidates = fb_candidates.sort_values(by=sort_cols, ascending=asc_order)
                picks = pd.concat([picks, fb_candidates.head(shortfall)])
            selected = pd.concat([selected, picks])
            used.update(picks['player_name'])

        if len(selected) < 11:
            fillers = pool[~pool['player_name'].isin(used)].sort_values(
                by=['matches', 'wickets', 'bat_avg'], ascending=False).head(11 - len(selected))
            selected = pd.concat([selected, fillers])

        return selected.drop_duplicates('player_name').head(11).reset_index(drop=True)

    def run_ga(self, input_team, input_venue):
        self.prepare_player_pool(input_team, input_venue)
        population = [self.generate_random_team() for _ in range(self.population_size)]
        population = [t for t in population if t is not None]

        # Fallback branch now runs GA on aggregated pool
        if not population:
            self.prepare_player_pool_full_squad(input_team)
            population = [self.generate_random_team() for _ in range(self.population_size)]
            population = [t for t in population if t is not None]
            if not population:
                return self.select_best_statistical_xi()

        for _ in range(self.generations):
            scored = [(t, self.fitness(t)) for t in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = [t for t, _ in scored[:10] if t is not None]
            if len(top) < 2:
                return self.select_best_statistical_xi()
            new_gen = top.copy()
            while len(new_gen) < self.population_size:
                p1, p2 = random.sample(top, 2)
                child = self.crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                if self.is_valid_team(child):
                    new_gen.append(child)
            population = new_gen

        return max(population, key=self.fitness)
