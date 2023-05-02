# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Purpose: 3D histograms for BMA parameters from M-H MCMC Algorithm
#
# Created by Tao Huang, July, 2022
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import numpy as np  
import matplotlib.pyplot as plt

# function to create a folder to store the results if it does not exist

def ResultsFolder(Folder):
    if os.path.exists(Folder) == False:
        os.mkdir(Folder)

Folder1 = './Results_MCMC/'
ResultsFolder(Folder1)

### Generateng 3D histogram distribution of BMA weights and variances

# read results from the MCMC sampling and the EM algorithm
MCMC_weight = np.genfromtxt('BMA_MCMC_weights.csv', delimiter=',',skip_header=1)
MCMC_sigma = np.genfromtxt('BMA_MCMC_standard_deviation.csv', delimiter=',',skip_header=1)

# figure for BMA weights
fig_w = plt.figure(figsize=(16,9))
ax_w = fig_w.add_subplot(projection='3d')

# figure for BMA sd
fig_sd = plt.figure(figsize=(16,9))
ax_sd = fig_sd.add_subplot(projection='3d')

colors = ['r','orange','y','g','c','b','purple','grey','coral','lime']

yticks = [i for i in range(1,11)]

n=-1

for c, k in zip(colors, yticks):
    n+=1
    # Generate the histogram for each BMA weights and sd
    weights = MCMC_weight[:,n]
    hist_w = np.histogram(weights,bins=10,density=True)

    sd = MCMC_sigma[:,n]
    hist_sd = np.histogram(sd,bins=10,density=True)

    # center of a bin
    xs_w = hist_w[1][:-1] + np.diff(hist_w[1])/2
    xs_sd = hist_sd[1][:-1] + np.diff(hist_sd[1])/2

    # density
    ys_w = hist_w[0]
    ys_sd = hist_sd[0]

    # Plot the bar graph given by xs and ys on the plane y=k
    ax_w.bar(xs_w, ys_w, zs=k, width=np.mean(np.diff(hist_w[1])),zdir='y', color=c, edgecolor='w')
    ax_sd.bar(xs_sd, ys_sd, zs=k, width=np.mean(np.diff(hist_sd[1])),zdir='y', color=c, edgecolor='w')

# set the view angle
ax_w.view_init(20,-20)
ax_sd.view_init(20,-20)
    
# keep the grid, default is True
#ax_w.grid(True)

ax_w.set_xlabel('BMA weight',labelpad=20, fontsize=20)
ax_w.set_ylabel('Model member',labelpad=20,fontsize=20)
ax_w.set_zlabel('Posterior density',labelpad=10,fontsize=20)

ax_sd.set_xlabel('BMA standard deviation',labelpad=20, fontsize=20)
ax_sd.set_ylabel('Model member',labelpad=20,fontsize=20)
ax_sd.set_zlabel('Posterior density',labelpad=10,fontsize=20)

# fontsize of label tick
ax_w.tick_params(labelsize=16)
ax_sd.tick_params(labelsize=16)

# On the y axis let's only label the discrete values that we have data for.
ax_w.set_yticks(yticks)
ax_sd.set_yticks(yticks)

fig_w.tight_layout(pad=0.5)
fig_sd.tight_layout(pad=0.5)

fig_w.savefig(Folder1+'BMA weights.jpg',dpi=300)
fig_sd.savefig(Folder1+'BMA sigmas.jpg',dpi=300)

plt.close()

print("DONE!")
