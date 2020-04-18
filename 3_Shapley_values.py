
# Shapley values
# ====================
# Shapley values are used in machine learning for model interpretation by estimating how much each feature
# contributes to the prediction.
#
# The concept comes from cooperative games in game theory. In this type of games there is
# a set of players and a value function that assigns a value for any possible subset of players ("coalition").
# The idea is to add each player sequentially to form a coalition. Then, we calculate his added value for each
# sequence and we average across all sequences. The result is the player Shapley value.
#
# For more information about the model, please see the book "The Model Thinker" by Scott E. Page.
#
# Algorithm:
# - Run over all possible player permutations.
# - For each permutation check the marginal contribution for adding this player.
# - Find the expected marginal contributions for each player.
#
# For more information about the use of Shapley values in ML, visit this great website:
# https://christophm.github.io/interpretable-ml-book/shapley.html
#
# Code by Amir Dudai (@amirdud)
# April 2020
#
# Please let me know if you find any corrections, or have any suggestions.
#
# ====================


from collections import defaultdict
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


# Define
def shapley_values(dic_legend,dic_v):
    # Calculation of Shapley values
    #
    # input:
    # dic_legend - define the participating players {1: "name_1"; 2: "name_2",...}
    # dic_v - keys should be sorted numbers, where the number of digits in each
    #              key increases: 1,2,3,12,13,23, etc.
    #
    # output:
    # shapley_vals - dictionary of Shapley values for each player

    # find all permutations
    objects_list = np.array(list(dic_legend.keys()))
    perms_tuple = list(itertools.permutations(objects_list))
    perms_list = [list(tup) for tup in perms_tuple]

    dic_marginal = defaultdict(list)

    # run over all possible permutations
    for perm in perms_list:

        # for each permutation check the marginal contribution for adding this player
        for i, element in enumerate(perm):

            before_element = perm[:i]
            till_element = perm[:i+1]

            # if nothing before
            if not before_element:
                val = dic_v[element]
                dic_marginal[element].append(val)

            else:
                till_element_key = int(''.join([str(x) for x in sorted(till_element)]))
                before_element_key = int(''.join([str(x) for x in sorted(before_element)]))

                till_val = dic_v[till_element_key]
                before_val = dic_v[before_element_key]
                val = till_val - before_val
                dic_marginal[element].append(val)

    shapley_vals = {k: np.array(v).mean() for (k, v) in dic_marginal.items()}

    # convert to numpy array
    shapley_vals_np = np.fromiter(shapley_vals.values(), dtype=float)
    keys_dic_v = np.fromiter(dic_v.keys(), dtype=int)

    # check result
    assert np.sum(shapley_vals_np) == dic_v[keys_dic_v[-1]], \
        'The Shapley values do not sum up to the last value defined'

    return shapley_vals

# Example 1 (Speakers):
# ======================
# 0 - spanish speaker
# 1 - french speaker
# 2 - spanish & french speaker
dic_legend = {1: 'spanish',
              2: 'french' ,
              3: 'bilingual'}

# values dictionary:
dic_v = {1: 0,
         2: 0,
         3: 1200,
         12: 1200,
         13: 1200,
         23: 1200,
         123: 1200}
# run
shapley_vals = shapley_values(dic_legend,dic_v)

# plot shapley values
names = list(dic_legend.values())
shaps = list(shapley_vals.values())

fig,(ax0,ax1) = plt.subplots(1,2,figsize=(10,4))
ax0.bar(range(len(shapley_vals)),shaps,tick_label=names,color='w',edgecolor='k',linewidth=1)
ax0.set_xlabel('Player')
ax0.set_ylabel('Shapley Value')
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.set_title('Speakers Example')

# Example 2 (crew team):
# ======================
# define dic_legend
dic_legend = {1: 'rower1',
              2: 'rower2' ,
              3: 'rower3',
              4: 'rower4',
              5: 'rower5',
              6: 'coxwain'}

# define dic_v:
helper_list = np.fromiter(dic_legend.keys(), dtype=int).tolist()

dic_v = {}
for i in range(1, len(helper_list)+1):
    for tup in itertools.combinations(helper_list, i):
        key = int(''.join([str(i) for i in sorted(tup)]))
        dic_v[key]=0

dic_v[12345] = 2
dic_v[12346] = 10
dic_v[12356] = 10
dic_v[12456] = 10
dic_v[13456] = 10
dic_v[23456] = 10
dic_v[123456] = 10

# run
shapley_vals = shapley_values(dic_legend,dic_v)

# plot shapley values
names = list(dic_legend.values())
shaps = list(shapley_vals.values())

ax1.bar(range(len(shapley_vals)),shaps,tick_label=names,color='w',edgecolor='k',linewidth=1)
ax1.set_xlabel('Player')
ax1.set_ylabel('Shapley Value')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('Crew Team Example')

