import streamlit as st
import pandas as pd
import numpy as np
import requests
from itertools import combinations
from math import log
from math import sqrt
from math import floor
from functools import reduce
import time

@st.cache_resource(ttl=36000)
def load_data(QUESTION_ID):
    # Set up headers with the session token
    headers = {
        "Content-Type": "application/json",
        "X-Metabase-Session": st.secrets["metabase_token"]
    }

    # Execute a specific saved question (query) using its ID
    query_url = f"https://regendata.xyz/api/card/{QUESTION_ID}/query/json"
    response = requests.post(query_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
    else:
        st.write("Error:", response.status_code, response.text)
        data = []

    # Load in df
    result = pd.DataFrame(data)
    return result

def standard_donation(donation_df):
  # just do a normal vote (nothing quadratic)
  projects = donation_df.columns
  funding = {p: donation_df[p].sum() for p in projects}
  return funding

def standard_qf(donation_df):
  projects = donation_df.columns
  funding = {p: donation_df[p].apply(lambda x: sqrt(x)).sum() ** 2 for p in projects}
  return funding

def pairwise(donation_df, M=0.01):

  projects = donation_df.columns
  donors = donation_df.index

  # start off with funding = sum of individual donations, then add the pairwise matching amounts
  funding = {p: donation_df[p].sum() for p in projects}
  sqrt_donation_df = donation_df.apply(lambda col: np.sqrt(col))

  # The next line of code creates a matrix containing each pairwise coefficient k_i,j
  # Unpacking it...
  # The dot product is a matrix multiplication that will give us a matrix where entry i,j is the dot product of
  # i's square-rooted donation vector with j's square-rooted donation vector.
  # Next, even though M is technically a scalar, pandas will automatically interpret the syntax "M + <matrix>"
  # by assuming that M here refers to a matrix with M in every entry, and the same dimensions as the actual matrix
  # on the other side of the +.
  # Same goes for "M / <matrix>".
  # The result is a matrix, "k_matrix", where entry i,j is the k_i,j described in the original pairwise matching blog post
  k_matrix = M / (M + sqrt_donation_df.dot(sqrt_donation_df.transpose()))

  proj_sets = {d : set([p for p in projects if donation_df.loc[d, p] > 0]) for d in donors}

  for  wallet1, wallet2 in combinations(donors,2):
    for p in proj_sets[wallet1].intersection(proj_sets[wallet2]):
      funding[p] += sqrt_donation_df.loc[wallet1, p] * sqrt_donation_df.loc[wallet2, p] * k_matrix.loc[wallet1, wallet2]

  return funding

def donation_profile_pairwise(donation_df):
  # pairwise qf where we attenuate based on how many shared projects users donated to

  projects = donation_df.columns
  donors = donation_df.index

  # start off with funding = sum of individual donations, then add the pairwise matching amounts
  funding = {p: donation_df[p].sum() for p in projects}

  # for each user, make a set of the projects they donated to. We'll reference these sets multiple times.
  proj_sets = {d : set([p for p in projects if donation_df.loc[d, p] > 0]) for d in donors}

  for d1, d2 in combinations(donors,2):

    shared_projects = proj_sets[d1].intersection(proj_sets[d2])
    if len(shared_projects) == 0: continue

    d1_total_projects = len(proj_sets[d1])
    d2_total_projects = len(proj_sets[d2])

    # calculate the number of projects unique to d1 and d2
    d1_unique_projects = len( proj_sets[d1].difference(proj_sets[d2]) )
    d2_unique_projects = len( proj_sets[d2].difference(proj_sets[d1]) )

    # coefficient = total number of unique projects donated to / total number of projects donated to
    # the more unique pojects donated to, the higher the coefficient.
    # if d1 and d2 did not donate to any of the same projects, the coefficient is 1. If they donated to the exact same set of projects, the coefficient is 0.
    coefficient = (d1_unique_projects + d2_unique_projects) / (d1_total_projects + d2_total_projects)

    for p in shared_projects:
      funding[p] += coefficient * sqrt(donation_df.loc[d1,p]) * sqrt(donation_df.loc[d2,p])

  return funding

def donation_profile_clustermatch(donation_df):
  # run cluster match, using donation profiles as the clusters
  # i.e., everyone who donated to the same set of projects gets put under the same square root.

  # we'll store donation profiles as binary strings.
  # i.e. say there are four projects total. if an agent donated to project 0, project 1, and project 3, they will be put in cluster "1101".
  # here the indices 0,1,2,3 refer to the ordering in the input list of projects.

  projects = donation_df.columns

  clusters = {} # a dictionary that will map clusters to the total donation amounts coming from those clusters.

  # build up the cluster donation amounts
  for (wallet, donations) in donation_df.iterrows():

    # figure out what cluster the current user is in
    c = ''.join('1' if donations[p] > 0 else '0' for p in projects)

    # now update that cluster's donation amounts (or initialize new donation amounts if this is the first donor from that cluster)
    if c in clusters.keys():
      for p in projects:
        clusters[c][p] += donations[p]
    else:
      clusters[c] = {p: donations[p] for p in projects}

  # now do QF on the clustered donations.
  funding = {p: sum(sqrt(clusters[c][p]) for c in clusters.keys()) ** 2 for p in projects}

  return funding

def stamp_profile_pairwise(donation_df, stamp_df):
  # pairwise match where we attenuate based on how many stamps users have in common

  projects = donation_df.columns
  donors = donation_df.index
  stamp_owners = stamp_df.index
  stamps = stamp_df.columns

  # start off with funding = sum of individual donations, then add the pairwise matching amounts
  funding = {p: donation_df[p].sum() for p in projects}

  # for each user, make a set of 1) the stamps they have and 2) the projects they donated to. We'll reference these sets multiple times.
  proj_sets = {d : set([p for p in projects if donation_df.loc[d, p] > 0]) for d in donors}
  stamp_sets = {d : set([s for s in stamps if stamp_df.loc[d, s] == 1]) for d in stamp_owners}

  for d1, d2 in combinations(donors,2):

    shared_projects = proj_sets[d1].intersection(proj_sets[d2])
    if len(shared_projects) == 0: continue

    d1_total_stamps = len(stamp_sets[d1])
    d2_total_stamps = len(stamp_sets[d2])

    if d1_total_stamps == 0 or d2_total_stamps == 0: continue # if a user has 0 stamps, exclude them from generating any matching funds

    # calculate the number of stamps unique to d1 and d2
    d1_unique_stamps = len( stamp_sets[d1].difference(stamp_sets[d2]) )
    d2_unique_stamps = len( stamp_sets[d2].difference(stamp_sets[d1]) )

    # coefficient = total number of unique stamps / total number of stamps
    # the more unique stamps, the higher the coefficient.
    # if d1 and d2 have no stamps in common, the coefficient is 1. If they share the exact same stamps, the coefficient is 0.
    coefficient = (d1_unique_stamps + d2_unique_stamps) / (d1_total_stamps + d2_total_stamps)

    for p in shared_projects:
      funding[p] += coefficient * sqrt(donation_df.loc[d1,p]) * sqrt(donation_df.loc[d2,p])

  return funding

def stamp_clustermatch(donation_df, stamp_df):
  # run cluster match, clustering on stamps.

  projects = donation_df.columns
  stamps = stamp_df.columns

  clusters = {s: {p: 0 for p in projects} for s in stamps} # a dictionary that will map clusters to the total donation amounts coming from those clusters.

  # build up the cluster donation amounts
  for (wallet, donations) in donation_df.iterrows():

    users_total_stamp_weight = stamp_df.loc[wallet].sum()

    for s in stamps:
      stamp_weight = stamp_df.loc[wallet, s]
      if stamp_weight > 0:
        for p in projects:
          clusters[s][p] += donations[p] * (stamp_weight / users_total_stamp_weight)

  # now do QF on the clustered donation amounts.
  funding = {p: sum(sqrt(clusters[s][p]) for s in stamps) ** 2 for p in projects}

  return funding

def CO_clustermatch(donation_df, stamp_df):
  # run CO-CM on a set of funding amounts, using stamps as the clusters
  # in this implementation, I try to improve on what we wrote down in the whitepaper.
  # a version that exactly follows the whitepaper is below.

  projects = donation_df.columns
  stamps = stamp_df.columns
  donors_in_clusters = stamp_df.index
  donors = donation_df.index

  # first, get a list of who's in each cluster
  cluster_members = {s: stamp_df.index[stamp_df[s] > 0].tolist() for s in stamps}

  # CO-CM requires us to compute a function K(i,h) for all pairs of agents (i) and clusters (h). In our paper, K(i,h) is either sqrt(c_i) or c_i.
  # (c_i is i's contribution amount to the project up for funding)
  # we're going to try something a little more sophisticated:
  # if i is in h, K(i,h) = sqrt(c_i)
  # if i is not in h, and i is not in any groups with anybody else who's in h, K(i,h) = c_i
  # if i is not in h, but i is in groups with folks who are in h:
  #     let X be the set of people that i is in a group with
  #     let Y be the subset of X who are in h
  #     return (|Y| / |X|) * sqrt(c_i) + (1 - |Y| / |X|) * c_i
  #     so, if most of the people i is in groups with are in h, we return something closer to sqrt(c_i)
  #     if very few of the people i is in a group with are in h, we return something closer to c_i.

  K = {p: {wallet: {s: 0 for s in stamps} for (wallet, stamp_profile) in stamp_df.iterrows()} for p in projects}

  # grab contributions and squares of contributions up front
  contribution = {d: {p: donation_df.loc[d, p] for p in projects} for d in donors}
  sqrt_contribution = {d: {p: sqrt(contribution[d][p]) for p in projects} for d in donors}

  # friendship_matrix is a matrix whose rows and columns are both wallets,
  # and a value greater than 0 at index i,j means that wallets i and j are in at least one group together.
  # using a matrix will make it much faster to understand who agents are in groups with, since matrix stuff is really optimized under the hood.
  friendship_matrix = stamp_df.dot(stamp_df.transpose())

  for (d, stamp_profile) in stamp_df.iterrows():
    # get a set of all people the current user (d) is friends with
    total_friends = set(friendship_matrix.index[friendship_matrix[d] > 0])
    for s in stamps:
      # figure out how "socially close" the current wallet is to cluster c
      # if the current wallet is in c, we're very socially close
      # use a variable called "balance" to denote how close we'll be to c_i vs sqrt(c_i)
      if stamp_profile[s] > 0:
        balance = 1
      elif len(total_friends) == 0:
        balance = 0
      # otherwise, out of all the people the current user is in a group with, how many of *them* are in c?
      else:
        friends_in_s = total_friends.intersection(set(cluster_members[s]))
        balance = len(friends_in_s) / len(total_friends)
      for p in projects:
        K[p][d][s] = (balance * sqrt_contribution[d][p]) + ((1-balance) * contribution[d][p])

  K_dfs = {p: pd.DataFrame(K[p]).transpose() for p in projects}
  normalized_stamp_df = stamp_df.apply(lambda row: row / row.sum(), axis=1)

  vec = stamp_df.apply(lambda row: row.sum(), axis=1)

  normalized_stamp_df.fillna(value=0, inplace=True)

  # initialize funding to be sum of donations, then add in pairwise amounts
  funding = {p: donation_df[p].sum() for p in projects}


  for p in projects:
    K_df = K_dfs[p]
    for (s1, s2) in combinations(stamps, 2):
      s1_sum = K_df[s2].dot(normalized_stamp_df[s1])
      s2_sum = K_df[s1].dot(normalized_stamp_df[s2])
      if np.isnan(s1_sum) or np.isnan(s2_sum):
        raise NotImplementedError('User has 0 stamps/ cluster memberships')
      funding[p] += sqrt(s1_sum * s2_sum)
  return funding

def CO_clustermatch_simple(donation_df, stamp_df):
  # run CO-CM on a set of funding amounts, using stamps as the clusters
  # unlike the above function (CO_clustermatch), follow the formula in the whitepaper exactly -- don't get fancy with K(i,h)

  projects = donation_df.columns
  stamps = stamp_df.columns
  donors_in_clusters = stamp_df.index
  donors = donation_df.index

  # first, get a list of who's in each cluster
  cluster_members = {s: stamp_df.index[stamp_df[s] > 0].tolist() for s in stamps}

  # CO-CM requires us to compute a function K(i,h) for all pairs of agents (i) and clusters (h). In our paper, K(i,h) is either sqrt(c_i) or c_i.

  K = {p: {wallet: {s: 0 for s in stamps} for (wallet, stamp_profile) in stamp_df.iterrows()} for p in projects}

  # grab contributions and squares of contributions up front
  contribution = {d: {p: donation_df.loc[d, p] for p in projects} for d in donors}
  sqrt_contribution = {d: {p: sqrt(contribution[d][p]) for p in projects} for d in donors}

  # friendship_matrix is a matrix whose rows and columns are both wallets,
  # and a value greater than 0 at index i,j means that wallets i and j are in at least one group together.
  # using a matrix will make it much faster to understand who agents are in groups with, since matrix stuff is really optimized under the hood.
  friendship_matrix = stamp_df.dot(stamp_df.transpose())

  # Pre-compute intersections
  cluster_members_sets = {s: set(cluster_members[s]) for s in stamps}  
  for (d, stamp_profile) in stamp_df.iterrows():
      # get a set of all people the current user (d) is friends with
      total_friends = set(friendship_matrix.index[friendship_matrix[d] > 0])
      total_friends_intersections = {s: len(total_friends & cluster_members_sets[s]) for s in stamps}  
      for p in projects:
          sqrt_contribution_dp = sqrt_contribution[d][p] if d in sqrt_contribution else None
          contribution_dp = contribution[d][p]  
          for s in stamps:
              # if d is in s or any of d's friends are in s, we should square root the contribution
              should_square_root = stamp_profile[s] > 0 or total_friends_intersections[s] > 0
              if should_square_root:
                  if sqrt_contribution_dp is not None:
                      K[p][d][s] = sqrt_contribution_dp
              else:
                  K[p][d][s] = contribution_dp

  K_dfs = {p: pd.DataFrame(K[p]).transpose() for p in projects}
  normalized_stamp_df = stamp_df.apply(lambda row: row / row.sum(), axis=1)

  # initialize funding to be sum of donations, then add in pairwise amounts
  funding = {p: donation_df[p].sum() for p in projects}

  for p in projects:
    K_df = K_dfs[p]
    for (s1, s2) in combinations(stamps, 2):
      s1_sum = K_df[s2].dot(normalized_stamp_df[s1])
      s2_sum = K_df[s1].dot(normalized_stamp_df[s2])
      if np.isnan(s1_sum) or np.isnan(s2_sum):
        raise NotImplementedError('User has 0 stamps/ cluster memberships')
      funding[p] += sqrt(s1_sum * s2_sum)

  return funding

@st.cache_resource(ttl=36000)
def run_qf_algos(donation_df, stamp_df=None):
    all_functions = [standard_donation, standard_qf,  CO_clustermatch_simple, CO_clustermatch, stamp_clustermatch, donation_profile_clustermatch, pairwise, stamp_profile_pairwise, donation_profile_pairwise ]
    requires_stamps = [CO_clustermatch_simple, CO_clustermatch, stamp_clustermatch, stamp_profile_pairwise]
    descriptions = {standard_donation: 'User donations only (nothing quadratic)',
                    standard_qf: 'Normal QF',
                    CO_clustermatch_simple: 'CO-CM (following the whitepaper)',
                    CO_clustermatch: 'CO-CM rev2',
                    stamp_clustermatch: 'Cluster match (clustering on stamps)',
                    donation_profile_clustermatch: 'Cluster match (clustering on donation profiles)',
                    pairwise: 'Pairwise match (following original blog post)',
                    stamp_profile_pairwise: 'Pairwise match (attenuating using stamp profiles)',
                    donation_profile_pairwise: 'Pairwise match (attenuating using donation profiles)'}
    
    results = dict()
    for f in all_functions:
        st.write(f'starting {f.__name__}')
        start = time.time()
        if f in requires_stamps:
            if stamp_df is not None:
                results[f] = f(donation_df,stamp_df)
        else:
            results[f] = f(donation_df)
        end = time.time()
        st.write(f"Function '{f.__name__}' execution time: {(end - start):.2f} seconds")

    results_eng_descriptions = {descriptions[alg] : results[alg] for alg in results.keys()}

    projects = donation_df.columns
    total_money = {x: sum(results_eng_descriptions[x][p] for p in projects) for x in results_eng_descriptions.keys()}
    results_normalized = {x: {p: results_eng_descriptions[x][p]/total_money[x] for p in projects} for x in results_eng_descriptions.keys()}
    res_norm_df = pd.DataFrame(results_normalized)
    return res_norm_df

