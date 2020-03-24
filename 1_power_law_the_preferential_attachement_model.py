#
# "The preferential attachment model" for generating a power-law distribution
# ============================================================================
# In this model, the larger the group is, it attracts more agents to join it ("the rich get richer").
# For more information about the model, please see the book "The Model Thinker" by Scott E. Page.
#
# Algorithm:
# In this model, each person arrives one after the other.
# - The first one to arrive forms a group.
# - Each subsequent person:
#   * forms a new group with probability p (should be small).
#   * joins an existing group with probability 1-p.
#     Then, the group for which the person joins is determined probabilistically according
#     to the relative size of the existing groups.
#
# Code by Amir Dudai (@amirdud)
# March 2020
#
#
# Please let me know if you find any corrections, or have any suggestions.
#
# ===========================================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
matplotlib.use('Qt5Agg')
import scipy.stats as ss
import copy

# define function
def pref_attach_model(n,p):
    # Simulation of the preferential attachment model
    #
    # input:
    # n - number of agents in the simulation
    # p - probability to form a new group each iteration
    #
    # output:
    # gs_sims - list of groups formed in each iteration
    # gs_p_sims - list of probabilities to join a group in each iteration

    gs_sims = []
    gs_p_sims = []
    gs = []
    for i in range(n):
        gs_size = np.array([len(c) for c in gs])
        gs_p = gs_size / i
        gs_p.tolist()
        gs_p_sims.append(gs_p)

        n_gs = len(gs)
        gs_ind = range(n_gs)

        # first visit
        if i == 0:
            g_curr = []
            g_curr.append(i)

            # add new group
            gs.append(g_curr)

        # all others
        else:
            rnd1 = np.random.rand(1)
            # form new group
            if rnd1 < p:
                g_curr = []
                g_curr.append(i)

                # add new group
                gs.append(g_curr)

            # join group
            else:
                rv = ss.rv_discrete(values=(gs_ind, gs_p))
                rv_ind = rv.rvs(size=1)[0]
                gs[rv_ind].append(i)

        gs_sim = copy.deepcopy(gs)
        gs_sims.append(gs_sim)

    # align indexing of gs_p_sims with gs_sims
    gs_p_sims.pop(0)
    gs_size = np.array([len(c) for c in gs])
    gs_p = gs_size / (n - 1)
    gs_p.tolist()
    gs_p_sims.append(gs_p)

    return gs_sims, gs_p_sims

# run
n = 5000
p = 0.02
gs_sims, gs_p_sims = pref_attach_model(n,p)

# show
gs_p_sorted = np.sort(gs_p_sims[-1])[::-1]
x = np.arange(1,gs_p_sorted.size+1,1)

fig,(ax_anim,ax1) = plt.subplots(1,2,figsize=(10,6))
plt.suptitle('The Preferential Attachment Model')

ax1.loglog(x,gs_p_sorted,'o')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel('Sorted Groups')
ax1.set_ylabel('Probability')

ax_anim.set_xlim(0, 20)
ax_anim.set_ylim(0,1)
ax_anim.set_xlabel('Sorted Groups')
ax_anim.set_ylabel('Probability')
ax_anim.spines['top'].set_visible(False)
ax_anim.spines['right'].set_visible(False)

y, = ax_anim.plot([],[], 'k-',lw=4, label='simulation')

def init():
    y.set_data(np.array([0]),gs_p_sims[0])
    return y

def update_data(i):
    y.set_data(np.arange(len(gs_sims[i])),np.sort(gs_p_sims[i])[::-1])
    return y

anim = FuncAnimation(fig, update_data, init_func=init,frames=1200, interval=2, blit=False)

# save
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\amird\\Anaconda3\\pkgs\\ffmpeg-4.2-ha925a31_0\\Library\\bin\\ffmpeg.exe'
mywriter = animation.FFMpegWriter(fps = 80)
anim.save('1_preferential_attachment_model.mp4',writer=mywriter)