# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Purpose: Metropolis-Hastings MCMC Algorithm for BMA
#         Proposal distribution is the uniform distribution
#
# Created by Tao Huang, January, 2022
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import numpy as np  
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

### log likelihood function  
def logLikelihood(weight, sigma, D, f, K, T):
    log_P = 0.0
    P = np.zeros((T,K))
    
    for k in range(K):
        # assume Gaussian distribution
        P[:,k] = weight[k]*norm.pdf(D, f[:,k], sigma[k])

    for t in range(T):
        Pt = np.log(np.sum(P[t,:]))
        log_P += Pt
        
    return log_P

### function that generates K uniform random numbers which add up to 1
def K_weight(K):
    mid_points = np.random.random(K-1)
    head = np.array([0])
    tail = np.array([1])

    split_points = np.sort(np.hstack((head,mid_points,tail)))
    
    weight = np.zeros(K)
    for k in range(K):
        weight[k] = split_points[k+1]-split_points[k]

    return weight

### function that generates the variance folowing a uniform distribution
def K_sigma(D,f,K,T):
    sigma = np.array([0.0]*K)

    for k in range(K):
        y_f = np.sum((D-f[:,k])**2)
        sigma[k] = np.random.uniform(0,1.5*np.sqrt(y_f/T))

    return sigma
        
### function for initialization of parameters
def init_data(D, f):
    # weight for each model
    weight = K_weight(K)

    # standard deviation (sigma) for each model
    sigma = np.array([0.0]*K)

    for k in range(K):
        y_f = np.sum((D-f[:,k])**2)
        sigma[k] = np.sqrt(y_f/T)

    log_P = logLikelihood(weight, sigma, D, f, K, T)

    return weight, sigma, log_P

### function of Metropolis-Hastings (MH) MCMC sampling
def MH_MCMC(D,f,old_weight,old_sigma,old_log_P):

    new_weight = K_weight(K)
    new_sigma = K_sigma(D,f,K,T)
    
    new_log_P = logLikelihood(new_weight, new_sigma, D, f, K, T)

    accept_rate = min([1,np.exp(min([1,new_log_P-old_log_P]))])

    u = np.random.random(1)

    if u <= accept_rate:
        return new_weight, new_sigma, new_log_P, 1
    else:
        return old_weight, old_sigma, old_log_P, 0


# function to create a folder to store the results if it does not exist

def ResultsFolder(Folder):
    if os.path.exists(Folder) == False:
        os.mkdir(Folder)

Folder1 = './Results_MCMC/'
ResultsFolder(Folder1)


### Main Program
# record the start time
time_start = time.process_time()

# observation data
D = np.genfromtxt('ObservedData_Example.csv', delimiter=',')
#D = np.genfromtxt('ObservedData.csv', delimiter=',')

# ensemble of all model predictions
f = np.genfromtxt('PredictionData_Example.csv', delimiter=',',skip_header=True)
#f = np.genfromtxt('PredictionData.csv', delimiter=',')

# number of obervations
T = len(D)

# number of models
K = f.shape[1]

# number of MCMC chains
chain = 100

# max number of sampling
N = 2000

# length of the burn-in period
#b = 200

# list to store the process values
# list for final weights
W_final = []

# list for final sigma
S_final = []

# list for final log likelihood
LL_final = []

# acceptance rate of each chain
acc_rate = []

'''
print("MCMC Algorithm for Calculation of BMA Parameters")
print("------------------------------------------------")
print("Initial weight:", weight)
print("Initial sigma:", sigma)
print("Initial log likelihood:", log_P)
print("------------------------------------------------")
'''

for c in range(chain):
    W = []
    S = []
    LL = []
    ar = 0

    # set initial values
    weight, sigma, log_P = init_data(D, f)

    W.append(list(weight))
    S.append(list(sigma))
    LL.append(log_P)

    for i in range(N):
        old_weight, old_sigma, old_log_P = W[i],S[i],LL[i]
        new_weight, new_sigma, new_log_P, n = MH_MCMC(D,f,old_weight,old_sigma,old_log_P)
    
        W.append(list(new_weight))
        S.append(list(new_sigma))
        LL.append(new_log_P)

        ar=ar+n

    W_final.append(W[-1])
    S_final.append(S[-1])
    LL_final.append(LL[-1])
    acc_rate.append(ar/N)
    
    if (c+1)%10 == 0:
        print("MCMC chain " + str(c+1) + " is done!")

# record the end time
time_end = time.process_time()

# compute the running time in minutes
time_running = (time_end - time_start)/60

print("The elapsed time is " + str(time_running) + " minutes.")

W_final = np.array(W_final)
S_final = np.array(S_final)
LL_final = np.array(LL_final)
acc_rate = np.array(acc_rate)

print("------------------------------------------------")
print("MCMC Algorithm for Calculation of BMA Parameters")
print("------------------------------------------------")
print("Final weight:",[np.mean(W_final[:,i]) for i in range(K)])
print("Final sigma:", [np.mean(S_final[:,i]) for i in range(K)])
print("Final log likelihood:",np.mean(LL_final))
print("Mean acceptance rate of multiple chains:",np.mean(acc_rate))

### plot the process values
# plot the weight 
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
#fig.tight_layout()

ax0.plot(W_final)
#ax0.hist(W_final[1:chain+1,0])
ax0.set_title('BMA Parameters from MCMC Algorithm')
ax0.set_ylabel("Weight")
#ax0.legend(["Model "+ str(i) for i in range(1,K+1)], loc='best',edgecolor='k')

# plot the sigma
ax1.plot(S_final)
#ax1.hist(S_final)
ax1.set_ylabel("$\\sigma$")
#ax1.legend(["Model "+ str(i) for i in range(1,K+1)], loc='best',edgecolor='k')

# plot the log likelihood
ax2.plot(LL_final, 'k--')
ax2.set_xlabel("Number of iteration")
ax2.set_ylabel("Log likelihood")

plt.show()

# Output the process and final results
header_K = np.array(list(range(1,K+1)))
#header_T = np.array(list(range(1,T+1))).T

W_final = np.row_stack((header_K,W_final))
S_final = np.row_stack((header_K,S_final))

Chain_No = np.array(list(range(chain)))
LL_final = np.column_stack((Chain_No,LL_final))

# results from a single Markov chain
np.savetxt('./Results_MCMC/OneChain_MCMC_weights.csv', W, fmt='%f',delimiter = ',')
np.savetxt('./Results_MCMC/OneChain_MCMC_standard_deviation.csv', S, fmt='%f',delimiter = ',')
np.savetxt('./Results_MCMC/OneChain_MCMC_log_likelihood.csv', LL, fmt='%f',delimiter = ',')

# results from multiple Markov chains
np.savetxt('./Results_MCMC/BMA_MCMC_weights.csv', W_final, fmt='%f',delimiter = ',')
np.savetxt('./Results_MCMC/BMA_MCMC_standard_deviation.csv', S_final, fmt='%f',delimiter = ',')
np.savetxt('./Results_MCMC/BMA_MCMC_log_likelihood.csv', LL_final, fmt='%f',delimiter = ',')


print("DONE!")

