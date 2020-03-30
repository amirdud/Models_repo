#
# "The self-organized criticality model" for generating a power-law distribution
# ===============================================================================
# In this group of models, the system reaches a critical state in which its components are distributed
# in a power-law distribution.
# The specific model we use here is "The forest fire model":
# Trees grow on a 2D grid. With some probability a tree catches fire and spreads it to all
# its neighboring trees. The distribution of the size of the components (connected trees)
# right before a fire strikes arranges itself to satisfy a power-law distribution.
#
# For more information about the model, please see the book "The Model Thinker" by Scott E. Page.
#
# Algorithm:
# In this model, there is an NxN 2-dimensional grid.
# - Each time point, a location on the grid is randomly picked:
#   * if empty, a tree grows with probability g (should be large).
#   * if not, the tree in this location catches fire with probability 1-g, and spreads through all
#     neighboring trees.
#
# Code by Amir Dudai (@amirdud)
# March 2020
#
# Please let me know if you find any corrections, or have any suggestions.
#
# ===============================================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import scipy.stats as ss
from scipy.ndimage.measurements import label
import copy
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

ffmpeg_PATH = 'C:\\Users\\amird\\Anaconda3\\pkgs\\ffmpeg-4.2-ha925a31_0\\Library\\bin\\ffmpeg.exe'

# functions
def add_tuple(x,y):
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])

    return tuple(z)

def set_fire(grd,loc,actions):
    # if there is a tree
    if grd[loc] == 1:
        grd[loc] = 0

        # go to neighbors
        for act in actions:
            next_loc = add_tuple(loc,actions[act])
            set_fire(grd,next_loc,actions)

# define function
def forest_fire_model(N,g,n_sims,p_init):
    # Simulation of the forest fire model
    #
    # input:
    # N - size of grid (N x N)
    # g - probability to grow a tree in an empty location
    # n_sims - number of time points in the simulation
    # p_init - the probability to have a tree in each location in the initial grid
    #
    # output:
    # grd_show_sims - list of grids in all time points
    # loc_show_sims - list of picked locations in all time points
    # is_tree_sims - list of whether there is or isn't a tree in the picked location
    # will_tree_sims - list of whether after g was determined there should be a tree or shouldn't
    #                  in the picked location. if the location was empty, 1 means a tree should grow
    #                  and 0 means it shouldn't. if there was a tree in this location, 1 means a tree
    #                  should stay there, and 0 means it should catch fire.
    # comp_size_sims - list of all component sizes in the forest
    # comp_labels_sims - list of labels of all components in the forest

    # initialize grid
    grd = np.pad(np.array(np.random.rand(N, N) < p_init, dtype=np.int), 1)

    # define neighbors
    filter = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # define the connection kernel structure
    actions = {'left': (-1, 0), 'right': (1, 0), 'down': (0, -1), 'up': (0, 1)}

    loc_sims = []
    grd_sims = []
    grd_show_sims = []
    loc_show_sims = []
    is_tree_sims = []
    will_tree_sims = []
    comp_size_sims = []
    comp_labels_sims = []

    # first grid
    grd_sim = copy.deepcopy(grd)
    grd_sims.append(grd_sim)
    grd_show_sims.append(grd_sim[1:-1, 1:-1])

    for i in range(n_sims):
        # draw a location
        (r, c) = np.random.randint(1, N + 1, size=2)  # for a 12x12 grid, allow indices: 1-10 (without margins: 0,11)

        rnd = np.random.rand(1)

        # if there is no tree
        if grd[(r, c)] == 0:
            is_tree_sims.append(0)

            # grow a tree
            if rnd < g:
                grd[(r, c)] = 1
                will_tree_sims.append(1)
            else:
                will_tree_sims.append(0)

        # if there is already a tree
        else:
            is_tree_sims.append(1)

            # forest fire
            if rnd > g:
                set_fire(grd, (r, c), actions)
                will_tree_sims.append(0)
            else:
                will_tree_sims.append(1)

        # get size of components
        labeled, n_comps = label(grd, filter)
        un_labels, un_counts = np.unique(labeled, return_counts=True)
        un_counts = un_counts[un_labels > 0]
        un_labels = un_labels[un_labels > 0]
        comp_size_sims.append(un_counts)
        comp_labels_sims.append(un_labels)

        grd_sim = copy.deepcopy(grd)
        grd_sims.append(grd_sim)
        grd_show_sims.append(grd_sim[1:-1, 1:-1])

        loc_sims.append((r, c))
        loc_show_sims.append((r - 1, c - 1))

    return grd_show_sims, loc_show_sims, is_tree_sims, will_tree_sims, comp_size_sims, comp_labels_sims

# run
N = 20
n_sims = 200
g = 0.98
p_init = 0.4

grd_show_sims, loc_show_sims, is_tree_sims, will_tree_sims, \
comp_size_sims,comp_labels_sims = forest_fire_model(N,g,n_sims,p_init)

# show
fig,(ax_grd,ax_graph) = plt.subplots(1,2,figsize=(12,6))
plt.suptitle('the self-organized criticality: forest-fire model')

ax_grd.set_xticks(np.arange(0,N+5,5))
ax_grd.set_yticks(np.arange(0,N+5,5))
ax_grd.set_xticklabels(np.arange(0,N+5,5))
ax_grd.set_yticklabels(np.arange(0,N+5,5))

ax_graph.set_xlim(0,100/2)
ax_graph.set_ylim(0,N*N/2)
ax_graph.set_xlabel('Components')
ax_graph.set_ylabel('Component Size')
ax_graph.spines['top'].set_visible(False)
ax_graph.spines['right'].set_visible(False)

im = ax_grd.imshow(grd_show_sims[0].T,extent=[0, N, 0, N],origin='lower',cmap='YlGn')
will_point_r, = ax_grd.plot([], [], 'wo',markersize=8,markeredgecolor='k',markerfacecolor='r')
will_point_g, = ax_grd.plot([], [], 'wo',markersize=8,markeredgecolor='k',markerfacecolor='g')
is_point_r, = ax_grd.plot([], [], 'wo',markersize=4,markeredgecolor='k',markerfacecolor='r')
is_point_g, = ax_grd.plot([], [], 'wo',markersize=4,markeredgecolor='k',markerfacecolor='g')
y, = ax_graph.plot([], [], 'k-',lw=4)

def init():
    # im.set_data(grd_sims[0].T)
    im.set_data(grd_show_sims[0].T)
    if is_tree_sims[0]==0:
        is_point_r.set_data([loc_show_sims[0][0]+0.5], [loc_show_sims[0][1] + 0.5])
        is_point_g.set_data([], [])
    else:
        is_point_r.set_data([], [])
        is_point_g.set_data([loc_show_sims[0][0] + 0.5], [loc_show_sims[0][1] + 0.5])

    if will_tree_sims[0]==0:
        will_point_r.set_data([loc_show_sims[0][0] + 0.5], [loc_show_sims[0][1] + 0.5])
        will_point_g.set_data([], [])
    else:
        will_point_r.set_data([], [])
        will_point_g.set_data([loc_show_sims[0][0] + 0.5], [loc_show_sims[0][1] + 0.5])

    y.set_data(comp_labels_sims[0], np.sort(comp_size_sims[0])[::-1])

    return [im,is_point_r,is_point_g,will_point_r,will_point_g]

def update_data(i):
    im.set_data(grd_show_sims[i].T)

    if is_tree_sims[i] == 0:
        is_point_r.set_data([loc_show_sims[i][0]  + 0.5], [loc_show_sims[i][1] + 0.5])
        is_point_g.set_data([], [])
    else:
        is_point_r.set_data([], [])
        is_point_g.set_data([loc_show_sims[i][0]  + 0.5], [loc_show_sims[i][1] + 0.5])

    if will_tree_sims[i] == 0:
        will_point_r.set_data([loc_show_sims[i][0]  + 0.5], [loc_show_sims[i][1] + 0.5])
        will_point_g.set_data([], [])
    else:
        will_point_r.set_data([], [])
        will_point_g.set_data([loc_show_sims[i][0]  + 0.5], [loc_show_sims[i][1] + 0.5])

    y.set_data(comp_labels_sims[i], np.sort(comp_size_sims[i])[::-1])

    return [im,is_point_r,is_point_g,will_point_r,will_point_g]

# Note: The face color of the circle shows whether there is or isn't a tree in the picked location.
#       The contour shows if a tree should be there or not in the next step.
anim = FuncAnimation(fig, update_data, init_func=init,frames=200, interval=50, blit=False)

# save
plt.rcParams['animation.ffmpeg_path'] = ffmpeg_PATH
mywriter = animation.FFMpegWriter()
anim.save('2_self_organized_criticality_model.mp4',writer=mywriter)