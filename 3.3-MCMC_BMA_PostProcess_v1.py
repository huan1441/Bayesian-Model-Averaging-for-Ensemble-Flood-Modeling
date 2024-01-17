# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Purpose: Post-precessing for parameters from M-H MCMC Algorithm for BMA
#
# Created by Tao Huang, January, 2022
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import numpy as np  
#from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# function to create a folder to store the results if it does not exist

def ResultsFolder(Folder):
    if os.path.exists(Folder) == False:
        os.mkdir(Folder)

Folder1 = './Results_MCMC/'
ResultsFolder(Folder1)

### Generateng the distribution of BMA weights and variances

# read results from the MCMC sampling and the EM algorithm
MCMC_weight = np.genfromtxt('BMA_MCMC_weights.csv', delimiter=',',skip_header=1)
MCMC_sigma = np.genfromtxt('BMA_MCMC_standard_deviation.csv', delimiter=',',skip_header=1)

EM_weight = np.genfromtxt('EM_weights.csv', delimiter=',')[-1]
EM_sigma = np.genfromtxt('EM_standard_deviation.csv', delimiter=',')[-1]

# number of models
K = MCMC_weight.shape[1]

# Empty arrarys to store the sampling BMA parameters and the confidence interval
BMA_weight_mean = np.full([K,1],np.nan)
Upper_BMA_weight = np.full([K,1],np.nan)
Lower_BMA_weight = np.full([K,1],np.nan)

BMA_sigma_mean = np.full([K,1],np.nan)
Upper_BMA_sigma = np.full([K,1],np.nan)
Lower_BMA_sigma = np.full([K,1],np.nan)

# Obtain (1-alpha)*100% confidence interval by taking the corresponding quantiles
alpha = 0.10

for i in range(K):
    # confidence interval for BMA weights
    BMA_weight_mean[i] = np.mean(MCMC_weight[:,i])
    Upper_BMA_weight[i] = np.quantile(MCMC_weight[:,i],1-alpha/2)
    Lower_BMA_weight[i] = np.quantile(MCMC_weight[:,i],alpha/2)

    # confidence interval for BMA sigmas
    BMA_sigma_mean[i] = np.mean(MCMC_sigma[:,i])
    Upper_BMA_sigma[i] = np.quantile(MCMC_sigma[:,i],1-alpha/2)
    Lower_BMA_sigma[i] = np.quantile(MCMC_sigma[:,i],alpha/2)

# Output the process and final results
header_K = np.array(list(range(1,K+1))).T

BMA_weight_result = np.column_stack((header_K,BMA_weight_mean,Upper_BMA_weight,Lower_BMA_weight))
BMA_sigma_result = np.column_stack((header_K,BMA_sigma_mean,Upper_BMA_sigma,Lower_BMA_sigma))

np.savetxt('./Results_MCMC/BMA_MCMC_weight_ci.csv', BMA_weight_result, fmt='%f',header="Model,BMA_weight_mean,Upper_bound,Lower_bound",delimiter = ',',comments='')
np.savetxt('./Results_MCMC/BMA_MCMC_sigma_ci.csv', BMA_sigma_result, fmt='%f',header="Model,BMA_sigma_mean,Upper_bound,Lower_bound",delimiter = ',',comments='')

### graphical analysis
MCMC_Data = MCMC_weight[:,0]
EM_Data = EM_weight[0]
#MCMC_Data = MCMC_sigma[:,8]
#EM_Data = EM_sigma[8]

# histogram
# Placing the plots in the plane
plot1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
plot2 = plt.subplot2grid((2, 2), (1, 0))
plot3 = plt.subplot2grid((2, 2), (1, 1))

# Trace plot
plot1.plot(MCMC_Data,color='g')
plot1.set_title('Trace Plot')
plot1.set_xlabel("Number of Markov chains")
plot1.set_ylabel("BMA weight")
#plot1.set_ylabel("BMA $\\sigma$")

# Autocorrelation
plot_acf(MCMC_Data,ax=plot2,marker=None,auto_ylims=True)
plot2.set_title('Autocorrelation')
plot2.set_xlabel("Lag")
plot2.set_ylabel("ACF")

# Histogram
plot3.hist(MCMC_Data,label='MCMC',bins=10,density=True,edgecolor='w')
plot3.plot(EM_Data,-0.01,'ro',markersize=10,label='EM') # plot the point at the bottom
plot3.set_title('Histogram')
plot3.set_xlabel("BMA weight")
#plot3.set_xlabel("BMA $\\sigma$")
plot3.set_ylabel("Posterior density")
plot3.legend(loc='best')

# Packing all the plots and displaying them
plt.tight_layout()

plt.savefig(Folder1+'BMA weight.jpg',dpi=300)
#plt.savefig(Folder1+'BMA sigma.jpg',dpi=300)

plt.close()

print("DONE!")

