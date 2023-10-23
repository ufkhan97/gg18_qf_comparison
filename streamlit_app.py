import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import plotly.graph_objs as go
import plotly.express as px
import utils


GG18_ROUNDS_QUERY_ID = 8
GG18_LONG_VOTES_QUERY_ID = 19
GG18_LONG_STAMPS_QUERY_ID = 20

st.set_page_config(
    page_title="GG18 QF",
    page_icon="📊",
    layout="wide",
)
st.title('📊 GG18 QF')

# Load in data
data_load_state = st.text('Loading data...')
#df_rounds = utils.load_data(GG18_ROUNDS_QUERY_ID)
#df_votes = utils.load_data(GG18_LONG_VOTES_QUERY_ID)
data_load_state.text('Loading data... done!')

df_rounds = pd.read_csv('df_rounds.csv')
df_votes = pd.read_csv('df_votes.csv')
df_votes['sum_amountusd'] = df_votes['sum_amountusd'].fillna(0)
df_votes['score'] = df_votes['score'].fillna(0)
df_rounds['min_donation_threshold_amount'] = df_rounds['min_donation_threshold_amount'].fillna(0)

# Filter by round and choose if you want to stamps
round_option = st.selectbox('Select Round', list(df_votes['round_name'].unique()), 4)
include_stamps = st.checkbox('Include Stamp Clustering Methods (slow)', value=False)


df_votes = df_votes[df_votes['round_name'] == round_option]
df_rounds = df_rounds[df_rounds['round_name'] == round_option]


# Calculate metrics
num_votes = len(df_votes)
sum_amountUSD = df_votes['sum_amountusd'].sum()
num_unique_voters = df_votes['voter'].nunique()
num_unique_projects = df_votes['project_name'].nunique()
matching_pool = df_rounds['matching_funds_available'].iloc[0]
matching_cap = df_rounds['matching_cap_amount'].iloc[0]
min_donation_threshold = df_rounds['min_donation_threshold_amount'].iloc[0]
unique_voters_passing_passport = df_votes[df_votes['passing_passport'] == True]['voter'].nunique()
total_unique_voters = df_votes['voter'].nunique()
percent_passing_passport = unique_voters_passing_passport / total_unique_voters

if round_option == 'Climate Round':
    df_shell = pd.read_csv('climate_shell.csv')
    matching_pool = 100000
     #filter df_votes to only include projects that respond Yes in the shell column
    df_votes = df_votes[df_votes['project_name'].isin(df_shell[df_shell['shell'] == 'Yes']['title'])]
# Display metrics
col1, col2 = st.columns(2)
col1.metric("Crowdfunded", '${:,.2f}'.format(sum_amountUSD))
col2.metric("Projects", '{:,}'.format(num_unique_projects))
col1.metric("Voters", '{:,}'.format(num_unique_voters))
col2.metric("Votes", '{:,}'.format(num_votes))
col1.metric("Matching Pool", '${:,.2f}'.format(matching_pool))
col2.metric("Matching Cap", '{}%'.format(matching_cap))
col1.metric("Percent Users Passing Passport", '{:.2%}'.format(percent_passing_passport))
#col2.metric("Stamps", '{:,}'.format(num_stamps))
col1.metric("Min Donation", '${:,.2f}'.format(min_donation_threshold))

# Filter and create pivot tables
df_votes = df_votes[df_votes['passing_passport'] == True]
df_votes = df_votes[df_votes['sum_amountusd'] >= min_donation_threshold]

pivot_stamps = None
if include_stamps:
    df_stamps = utils.load_data(GG18_LONG_STAMPS_QUERY_ID)
    df_stamps = df_stamps[df_stamps['address'].isin(df_votes['voter'])]
    num_stamps = len(df_stamps)
    df_stamps = df_stamps[df_stamps['address'].isin(df_votes['voter'])]
    df_stamps['provider_exists'] = 1
    df_votes['voter'] = df_votes[df_votes['voter'].isin(df_stamps['address'])]['voter']
    pivot_stamps = df_stamps.pivot_table(index='address', columns='provider', values='provider_exists', fill_value=0, aggfunc='max')

pivot_votes = df_votes.pivot_table(index='voter', columns='project_name', values='sum_amountusd', fill_value=0)

st.subheader('QF ALGORITHMS')
qf = utils.run_qf_algos(matching_cap/100, pivot_votes, pivot_stamps)
st.write(qf)

df_qf = pd.DataFrame(qf).reset_index()

df_qf = df_qf.rename(columns={'index': 'project'})
# add grantAddress based on project name using df_votes
#df_qf = pd.merge(df_qf, df_votes[['project_name', 'grantAddress']], on='project', how='left')
df_qf = df_qf.set_index(['project'])#, 'grantAddress'])
df_qf = df_qf.stack()
df_qf = pd.DataFrame(df_qf).reset_index()
df_qf.columns = ['project',  'algorithm', 'match_percent']
df_qf['round_id'] = df_rounds['round_id'].iloc[0]
df_qf['round_name'] = round_option
df_qf['match_amount'] = df_qf['match_percent'] * matching_pool
# desired db schema: round_id, round_name, algorithim, project, match_amount
df_qf = df_qf[['round_id', 'round_name', 'algorithm', 'project',  'match_percent', 'match_amount']]

st.subheader('Download Matching Results')
algo_option = st.selectbox('Select Algorithm', list(df_qf['algorithm'].unique()), 0)
df_qf_filtered = df_qf[df_qf['algorithm'] == algo_option]
st.write(df_qf_filtered) ## THIS ONE FOR THE DATABASE
# add download button
st.download_button('Download CSV', df_qf_filtered.to_csv(index=False), 'qf_results.csv', )


