import streamlit as st
import pandas as pd
from ga_team_selector import CricketTeamGA

STATS_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv"
ROLES_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv"

ROLE_DISPLAY = {
    'opener': 'Opener',
    'middle_order': 'Middle order',
    'wicket_keeper': 'Wicket-Keeper',
    'finisher': 'Finisher',
    'spinner': 'Spinner',
    'fast_bowler': 'Fast Bowler'
}
ROLE_ORDER = ['Opener', 'Middle order', 'Wicket-Keeper', 'Finisher', 'Spinner', 'Fast Bowler']

def format_roles(df):
    df = df.copy()
    if 'role' in df.columns:
        df['role'] = df['role'].map(ROLE_DISPLAY).fillna(df['role'])
    return df

def sort_by_role(df):
    df = df.copy()
    if 'role' in df.columns:
        df['role_order_index'] = df['role'].map(lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 99)
        df = df.sort_values(by='role_order_index').drop(columns=['role_order_index'])
    return df

def format_floats(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=['float', 'float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].apply(
            lambda x: int(x) if pd.notnull(x) and float(x).is_integer()
            else round(x, 2) if pd.notnull(x)
            else x
        )
    return df

def clean_table(df):
    cols_to_drop = [c for c in df.columns if 'lower' in c or 'venue' in c or c == 'franchise']
    return df.drop(columns=cols_to_drop, errors='ignore')

def show_table(df, start_index=1):
    df = df.copy()
    df.index = range(start_index, start_index + len(df))
    st.dataframe(df)

st.title("Cricket Intelligence Model - Best XI Selector")

df_roles_raw = pd.read_csv(ROLES_FILE)
df_stats_raw = pd.read_csv(STATS_FILE)

franchise_display = sorted(df_roles_raw['franchise'].dropna().unique())
venue_display = sorted(df_stats_raw['venue'].dropna().unique())

input_team_display = st.selectbox("Select Franchise", franchise_display)
input_venue_display = st.selectbox("Select Venue", venue_display)

input_team = input_team_display.strip().lower()
input_venue = input_venue_display.strip().lower()

if st.button("Select Best XI"):
    try:
        ga_model = CricketTeamGA(STATS_FILE, ROLES_FILE)
        best_team = ga_model.run_ga(input_team, input_venue)

        st.subheader(f"{input_team_display} Squad List")
        squad_df = clean_table(ga_model.franchise_list.copy())
        squad_df = format_floats(sort_by_role(format_roles(squad_df)))
        show_table(squad_df, start_index=1)

        st.subheader(f"Player Pool Used (min_matches ≥ {ga_model.min_matches_used})")
        pool_df = clean_table(ga_model.player_pool.copy())
        pool_df = format_floats(sort_by_role(format_roles(pool_df)))
        show_table(pool_df, start_index=1)

        st.subheader("Role Counts")
        role_counts_data = [
            {"Role": ROLE_DISPLAY[r], "Count": int(ga_model.role_counts.get(r, 0))}
            for r in ['opener','middle_order','wicket_keeper','finisher','spinner','fast_bowler']
        ]
        role_counts_df = pd.DataFrame(role_counts_data)
        show_table(role_counts_df, start_index=1)

        best_team_df = clean_table(best_team.copy())
        best_team_df = format_floats(sort_by_role(format_roles(best_team_df)))
        st.subheader(f'Selected Best Playing XI for {input_venue_display}')
        show_table(best_team_df, start_index=1)

        leftover_df = ga_model.player_pool[~ga_model.player_pool['player_name'].isin(best_team['player_name'])].copy()
        if not leftover_df.empty:
            # Aggregate any leftovers & sort
            leftover_df = leftover_df.groupby('player_name', as_index=False).agg({
                'role': 'first',
                'matches': 'sum',
                'runs': 'sum',
                'bat_avg': 'mean',
                'bat_sr': 'mean',
                'wickets': 'sum',
                'econ': 'mean',
                'indian': 'first'
            })

        leftover_df = clean_table(leftover_df)
        leftover_df = format_floats(sort_by_role(format_roles(leftover_df)))
        st.subheader("Players Left Out from Player Pool")
        show_table(leftover_df, start_index=len(best_team_df)+1)

        st.success(f"Total Fitness Score: {ga_model.fitness(best_team):.2f}")

        if "fallback" in str(ga_model.min_matches_used).lower():
            st.info("Not enough venue stats, showing best squad XI (aggregated stats) instead.")

    except Exception as e:
        st.error(str(e))
