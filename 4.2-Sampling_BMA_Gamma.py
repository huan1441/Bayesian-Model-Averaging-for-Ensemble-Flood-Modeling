# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Purpose: Sampling water stage predictions based on BMA parameters (Gamma PDF)
#
# Created by Tao Huang, July, 2022
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import numpy as np
from scipy.stats import gamma
#from scipy.stats import norm
#import matplotlib.pyplot as plt

# function to create a folder to store the results if it does not exist

def ResultsFolder(Folder):
    if os.path.exists(Folder) == False:
        os.mkdir(Folder)

Folder1 = './Results_BMA/'
ResultsFolder(Folder1)


### Generateng BMA probabilistic ensemble predictions

# original ensemble of all model predictions
f = np.genfromtxt('PredictionData_Example.csv', delimiter=',',skip_header=True)

# number of obervations
T = f.shape[0]

# number of models
K = f.shape[1]


# probability based on BMA weights
#prob_raw = np.genfromtxt('EM_weights.csv', delimiter=',')[-1]
prob_raw = np.mean(np.genfromtxt('BMA_MCMC_weights.csv', delimiter=',',skip_header=1),axis=0)

# obtain No. list of the models with higher weights (> 0.0001)
HWlist = []
for i in range(K):
    if prob_raw[i] > 0.0001:
        HWlist.append(i)

# number of models with higher weights
K_high = len(HWlist)
High_Model_No = list(range(K_high))
prob_high = prob_raw[HWlist]

# normalize the probility, the sum of which is equal to 1
prob = np.full([K_high],np.nan)
for i in range(K_high):
    prob[i] = prob_high[i]/np.sum(prob_high)

# standard deviation of the models with higher weights
#sigma_raw = np.genfromtxt('EM_standard_deviation.csv', delimiter=',')[-1]
sigma_raw = np.mean(np.genfromtxt('BMA_MCMC_standard_deviation.csv', delimiter=',',skip_header=1),axis=0)
sigma = sigma_raw[HWlist]

# Sampling size
Size = 1000

# predictions of models with higher weights
f_high = f[:,HWlist]

# Empty arrarys to store the sampling BMA predictions and the confidence interval
BMA_sample = np.full([T,Size],np.nan)
BMA_mean = np.full([T,1],np.nan)
Upper_BMA_mean = np.full([T,1],np.nan)
Lower_BMA_mean = np.full([T,1],np.nan)

for i in range(T):
    # Generate the integer values (with number of "Size") of High_Model_No. from all the model numbers based on higher BMA weights
    sample_model_no = np.random.choice(High_Model_No, size=Size, replace=True, p=prob)

    for j in range(Size):

        BMA_sample[i,j] = gamma.rvs(a=np.square(f_high[i,sample_model_no[j]]/sigma[sample_model_no[j]]),
                                    loc=0,
                                    scale=np.square(sigma[sample_model_no[j]])/f_high[i,sample_model_no[j]],
                                    size=1)

# Obtain (1-alpha)*100% confidence interval by taking the corresponding quantiles
alpha = 0.10

for i in range(T):
    BMA_mean[i] = np.mean(BMA_sample[i,:])
    Upper_BMA_mean[i] = np.quantile(BMA_sample[i,:],1-alpha/2)
    Lower_BMA_mean[i] = np.quantile(BMA_sample[i,:],alpha/2)

# Output the process and final results
header_T = np.array(list(range(1,T+1))).T

        
BMA_sample = np.column_stack((header_T,BMA_sample))

BMA_result = np.column_stack((header_T,BMA_mean,Upper_BMA_mean,Lower_BMA_mean))

np.savetxt('./Results_BMA/BMA_sample.csv', BMA_sample, fmt='%f',delimiter = ',')
np.savetxt('./Results_BMA/BMA_sample_mean.csv', BMA_result, fmt='%f',header="Time,BMA_mean,Upper_bound,Lower_bound",delimiter = ',',comments='')

print("DONE!")
