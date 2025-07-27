import streamlit as st
import pandas as pd
import numpy as np
import random

# 🎯 Your existing backend logic here (just wrapped in functions)
from models import generate_best_xi  # assume we refactor your script into this function

# 🎨 Streamlit UI
st.title("🏏 Cricket Intelligent Model - Best XI Selector")

# --- Load datasets ---
df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv")
roles_df = pd.read_csv("D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv")

# Prepare dropdown lists
venues = sorted(df['venue'].dropna().unique())
teams = sorted(roles_df['team'].dropna().unique())

# --- User inputs ---
selected_venue = st.selectbox("🏟 Select Venue", venues)
selected_team = st.selectbox("🛡 Select Team", teams)

if st.button("🔍 Generate Best XI"):
    with st.spinner("Generating the best team..."):
        best_team, left_out, score = generate_best_xi(selected_venue, selected_team)

        st.subheader("✅ Best Playing XI")
        st.dataframe(best_team)

        st.metric(label="🏆 Team Fitness Score", value=round(score, 2))

        st.subheader("🪑 Players Left Out")
        st.dataframe(left_out)
