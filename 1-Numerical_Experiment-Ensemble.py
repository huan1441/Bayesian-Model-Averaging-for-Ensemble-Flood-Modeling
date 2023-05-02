# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Purpose: Generate ensemble of HEC-RAS simulations for water stage
#          by adding multiple white-noise errors into the observed data
#
# Created by Tao Huang, June 2022
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import numpy as np  
from scipy.stats import norm
#import copy
#import matplotlib.pyplot as plt

# function to create a folder to store the results if it does not exist

def ResultsFolder(Folder):
    if os.path.exists(Folder) == False:
        os.mkdir(Folder)

### a function that yields a number of samples (ratios) drawn from a prior distribution of the channel Manning's n and upstream streamflow
def MCMC_ensemble(obs,sd):
    # obs is the time-series observed data and sd is a list of different standard deviations

    # length of time-series data
    T = len(obs)

    # number of candidate models
    K = len(sd)

    MCMC_ensemble = np.full([T,K],np.nan)

    for i in range(K):
        MCMC_error = norm.rvs(loc=0,scale=sd[i],size=T)

        MCMC_ensemble[:,i] = obs + MCMC_error

    header_K = np.array(list(range(1,K+1)))

    MCMC_prediction = np.row_stack((header_K,MCMC_ensemble))

    np.savetxt('./Results_MCMC/PredictionData_Example.csv', MCMC_prediction, fmt='%f',delimiter = ',')


### Main program
Folder1 = './Results_MCMC/'
ResultsFolder(Folder1)
    
# read obervations
MCMC_obs = np.genfromtxt('./ObservedData_Example.csv', delimiter=',')

# standard deviation of 10 candidate models
sd_list = [0.2,0.2,0.4,0.4,0.6,0.6,0.8,0.8,1,1]

MCMC_ensemble(MCMC_obs,sd_list)

print("DONE!")

