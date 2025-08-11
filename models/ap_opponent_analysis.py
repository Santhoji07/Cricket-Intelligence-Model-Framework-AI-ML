# ap_opponent_analysis.py
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

def run_apriori_matchups(my_xi, opponent_xi, ball_df):
    # Filter only my bowlers vs opponent batsmen
    my_bowlers = my_xi[my_xi['role'].isin(['fast_bowler', 'spinner'])]['player_name'].str.lower().tolist()
    opponent_bats = [p.lower() for p in opponent_xi]

    wk = ball_df[
        (ball_df['bowler'].str.lower().isin(my_bowlers)) &
        (ball_df['batsman'].str.lower().isin(opponent_bats)) &
        (ball_df['is_wicket'] == 1)
    ].copy()

    # Keep only bowler-induced dismissals
    wk = wk[wk['dismissal_type'].isin(['caught', 'bowled', 'lbw', 'stumped', 'hit wicket'])]
    if wk.empty:
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

    # Add prefixes for items
    wk['bowler'] = 'bowler:' + wk['bowler'].str.lower()
    wk['batsman'] = 'batsman:' + wk['batsman'].str.lower()
    wk['phase'] = 'phase:' + wk['phase'].astype(str)
    wk['venue'] = 'venue:' + wk['venue'].str.lower()
    wk['dismissal_type'] = 'dismissal:' + wk['dismissal_type'].str.lower()

    # Build transactions
    transactions = wk[['bowler', 'batsman', 'phase', 'venue', 'dismissal_type']].values.tolist()
    all_items = sorted(set(i for t in transactions for i in t))

    encoded_df = pd.DataFrame(0, index=np.arange(len(transactions)), columns=all_items)
    for idx, items in enumerate(transactions):
        encoded_df.loc[idx, items] = 1

    # One-hot encoded DataFrame -> convert to bool type
    encoded_df = encoded_df.astype(bool)


    # Run Apriori
    freq_items = apriori(encoded_df, min_support=0.01, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)
    rules = rules[rules['lift'] > 1.2]

    # Filter to bowler → batsman rules
    def is_valid_rule(row):
        ants = ' '.join(row['antecedents'])
        cons = ' '.join(row['consequents'])
        return ('bowler:' in ants) and ('batsman:' in cons)

    rules = rules[rules.apply(is_valid_rule, axis=1)]
    rules['antecedents'] = rules['antecedents'].apply(lambda s: ', '.join(s))
    rules['consequents'] = rules['consequents'].apply(lambda s: ', '.join(s))

    return rules[['antecedents','consequents','support','confidence','lift']].sort_values(
        ['lift', 'confidence'], ascending=False
    )

