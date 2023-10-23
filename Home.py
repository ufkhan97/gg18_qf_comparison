import pandas as pd
import numpy as np
import utils


GG18_ROUNDS_QUERY_ID = 8
GG18_LONG_VOTES_QUERY_ID = 19
GG18_LONG_STAMPS_QUERY_ID = 20

print("Loading Round Data...")
df_rounds = pd.read_csv('df_rounds.csv')#utils.load_data(GG18_ROUNDS_QUERY_ID)
df_rounds['min_donation_threshold_amount'] = df_rounds['min_donation_threshold_amount'].fillna(0)

print("Loading Vote Data...")
df_votes = pd.read_csv('df_votes.csv')#utils.load_data(GG18_LONG_VOTES_QUERY_ID)
df_votes['sum_amountusd'] = df_votes['sum_amountusd'].fillna(0)
df_votes['score'] = df_votes['score'].fillna(0)
df_votes['voter'] = df_votes['voter'].str.lower()

print("Loading Stamp Data...")
df_stamps = utils.load_data(GG18_LONG_STAMPS_QUERY_ID)
df_stamps = df_stamps[df_stamps['address'].isin(df_votes['voter'])]
df_stamps['address'] = df_stamps['address'].str.lower()

print("Loading Completed. Processing data...")
def process_round_votes(round_votes,  min_donation_threshold):
    round_votes = round_votes[round_votes['passing_passport'] == True]
    round_votes = round_votes[round_votes['sum_amountusd'] >= min_donation_threshold]
    pivot_votes = round_votes.pivot_table(index='voter', columns='project_name', values='sum_amountusd', fill_value=0)
    return pivot_votes

rounds_to_test = ['Web3 Social', 'Climate Round']
total_rounds = len(rounds_to_test)
result = []
for i, round in enumerate(rounds_to_test):
    print(f"Processing round {i+1} of {total_rounds}: {round}")

    min_donation_threshold = df_rounds.loc[df_rounds['round_name'] == round, 'min_donation_threshold_amount'].iloc[0]
    matching_cap = df_rounds.loc[df_rounds['round_name'] == round, 'matching_cap_amount'].iloc[0]
    matching_pool = df_rounds.loc[df_rounds['round_name'] == round, 'matching_funds_available'].iloc[0]
    round_votes = df_votes[df_votes['round_name'] == round]
    pivot_votes = process_round_votes(round_votes,  min_donation_threshold)
    round_stamps = df_stamps[df_stamps['address'].isin(round_votes['voter'])]
    round_votes = round_votes[round_votes['voter'].isin(df_stamps['address'])]
    round_stamps['provider_exists'] = 1
    pivot_stamps = round_stamps.pivot_table(index='address', columns='provider', values='provider_exists', fill_value=0, aggfunc='max')
    qf = utils.run_qf_algos(matching_cap/100, pivot_votes, pivot_stamps)
    df_qf = pd.DataFrame(qf).reset_index()
    df_qf.rename(columns={'index': 'project'}, inplace=True)
    df_qf.set_index('project', inplace=True)
    df_qf = df_qf.stack().reset_index()
    df_qf.columns = ['project', 'algorithm', 'match_percent']
    df_qf['match_amount'] = df_qf['match_percent'] * matching_pool
    df_qf['round_id'] = df_rounds['round_id'].iloc[0]
    df_qf = df_qf[['round_id', 'round_name', 'algorithm', 'project', 'match_percent']]
    result.append(df_qf)

print("Processing completed.")
df_qf = pd.concat(result)
df_qf.to_csv('qf_results.csv', index=False)