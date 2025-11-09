# montecarlo_simulator.py
import pandas as pd
import numpy as np
import json

def run_montecarlo_simulation(ga_file, svm_file, apriori_file, xgb_file, n_simulations=1000):
    # Load datasets
    ga_df = pd.read_csv(ga_file)
    svm_df = pd.read_csv(svm_file) if svm_file.endswith('.csv') else None
    apriori_df = pd.read_csv(apriori_file)
    xgb_df = pd.read_csv(xgb_file)

    # Safety checks
    if ga_df.empty or xgb_df.empty:
        return {"error": "Missing or empty input data."}

    # Base parameters
    players = ga_df['player_name'].tolist()
    win_counts = 0
    simulation_results = []

    for _ in range(n_simulations):
        score = 0.0
        # Random performance factors
        for player in players:
            base_runs = ga_df.loc[ga_df['player_name']==player, 'runs'].mean()
            base_wkts = ga_df.loc[ga_df['player_name']==player, 'wickets'].mean()
            # Add performance noise and matchup adjustment
            perf_factor = np.random.normal(1.0, 0.2)
            matchup_boost = xgb_df['prob_wicket'].mean() if 'prob_wicket' in xgb_df.columns else 0.05
            score += base_runs * perf_factor - base_wkts * matchup_boost * 10
        # Simple win threshold
        if score > np.percentile(np.random.normal(300, 50, 1000), 50):
            win_counts += 1
        simulation_results.append(score)

    win_probability = (win_counts / n_simulations) * 100
    summary = {
        "win_probability": round(win_probability, 2),
        "avg_team_score": round(np.mean(simulation_results), 2),
        "score_std": round(np.std(simulation_results), 2),
        "n_simulations": n_simulations
    }

    return summary
