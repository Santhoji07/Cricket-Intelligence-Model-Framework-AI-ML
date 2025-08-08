import streamlit as st
import pandas as pd
from ga_team_selector import CricketTeamGA

# Update these paths to your actual CSV locations
STATS_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv"
ROLES_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv"

def main():
    st.title("Cricket Intelligence Model - Best XI Selector")

    # Load for dropdown options
    df_roles = pd.read_csv(ROLES_FILE)
    df_stats = pd.read_csv(STATS_FILE)

    df_roles['franchise'] = df_roles['franchise'].str.strip().str.lower()
    df_stats['venue'] = df_stats['venue'].str.strip().str.lower()

    franchises = sorted(df_roles['franchise'].dropna().unique())
    venues = sorted(df_stats['venue'].dropna().unique())

    input_team = st.selectbox("Select Franchise", options=franchises)
    input_venue = st.selectbox("Select Venue", options=venues)

    ga_model = CricketTeamGA(STATS_FILE, ROLES_FILE)

    if st.button("Select Best XI"):
        try:
            best_team = ga_model.run_ga(input_team.strip().lower(), input_venue.strip().lower())

            role_order = ['opener', 'middle_order', 'wicket_keeper', 'finisher', 'spinner', 'fast_bowler']
            best_team = best_team.assign(
                role_order_index=best_team['role'].apply(lambda r: role_order.index(r))
            ).sort_values(by='role_order_index').reset_index(drop=True).drop(columns=['role_order_index'])

            st.subheader("Selected Best Playing XI:")
            st.dataframe(best_team[['player_name', 'role', 'bat_avg', 'bat_sr', 'wickets', 'econ', 'indian']])

            total_fitness = ga_model.fitness(best_team)
            st.write(f"**Total Fitness Score:** {round(total_fitness, 2)}")

        except Exception as e:
            st.error(f"Error selecting team:\n{str(e)}\n\nCheck that sufficient players exist for this franchise and venue.")

if __name__ == "__main__":
    main()
