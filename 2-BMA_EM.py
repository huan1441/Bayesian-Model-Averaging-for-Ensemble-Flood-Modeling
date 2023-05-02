# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Purpose: EM Algorithm for estimating Bayesian Model Averaging parameters
#
# Created by Tao Huang, April, 2021
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

### expectation step  
def E_step(D, f, weight, sigma, K, T):
    Numer = np.zeros((T,K))
    Denom = np.zeros((T,K))

    for k in range(K):
        Numer[:,k] = weight[k]*norm.pdf(D, f[:,k], sigma[k])

    for t in range(T):
        Denom[t,:] = np.sum(Numer[t,:]) 

    z_expectation = Numer / Denom    
    return z_expectation

### maximization step 
def M_step(D, f, K, T, z_expectation):

    for k in range(K):
        weight[k] = np.sum(z_expectation[:,k])/T
        sigma[k] = np.sqrt(np.sum(z_expectation[:,k]*(D-f[:,k])**2)/np.sum(z_expectation[:,k]))

    return weight, sigma 

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
        
### function for initialization of parameters
def init_data(D, f):
    # weight for each model
    #weight = np.array([1/K]*K)
    weight = K_weight(K)

    # standard deviation (sigma) for each model
    sigma = np.array([0.0]*K)

    var_sum = 0.0

    for k in range(K):
        y_f = np.sum((D-f[:,k])**2)
        var_sum += y_f

    sigma[:] = np.sqrt(var_sum/T/K)

    log_P = logLikelihood(weight, sigma, D, f, K, T)

    return weight, sigma, log_P

# function to create a folder to store the results if it does not exist

def ResultsFolder(Folder):
    if os.path.exists(Folder) == False:
        os.mkdir(Folder)

Folder1 = './Results_EM/'
ResultsFolder(Folder1)


### Main Program
# record the start time
time_start = time.process_time()

# observation data
D = np.genfromtxt('ObservedData_Example.csv', delimiter=',')

# ensemble of all model predictions
f = np.genfromtxt('PredictionData_Example.csv', delimiter=',',skip_header=True)

# number of obervations
T = len(D)

# number of models
K = f.shape[1]

# max number of iteration
Iter = 1000

# pre-specified tolerance level
Epsilon = 1e-4

# list to store the process values
# list for weight
W = []
# list for sigma
S = []
# list for log likelihood
LL = []

weight, sigma, log_P = init_data(D, f)

W.append(list(weight))
S.append(list(sigma))
LL.append(log_P)

print("EM Algorithm for Calculation of BMA Parameters")
print("----------------------------------------------")
print("Initial weight:", weight)
print("Initial sigma:", sigma)
print("Initial log likelihood:", log_P)
print("----------------------------------------------")

for i in range(Iter):  
    old_log_P = log_P
    
    z_expectation = E_step(D, f, weight, sigma, K, T)
    weight, sigma = M_step(D, f, K, T, z_expectation)   
    log_P = logLikelihood(weight, sigma, D, f, K, T)
    
    W.append(list(weight))
    S.append(list(sigma))
    LL.append(log_P)
    #print("Log likelihood from Iteration",i+1,":",log_P)
    
    if abs(log_P - old_log_P) < Epsilon:  
        break

# record the end time
time_end = time.process_time()

# compute the running time in minutes
time_running = (time_end - time_start)/60

print("The elapsed time is " + str(time_running) + " minutes.")


print("----------------------------------------------")
print("Final results from Iteration",i+1,":")
print("Final weight:",weight)
print("Final sigma:", sigma)
print("Final log likelihood:",log_P)

### plot the process values
# plot the weight 
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
#fig.tight_layout()

ax0.plot(W)
ax0.set_title('BMA Parameters from EM Algorithm')
#ax0.set_xlabel("Number of iteration")
ax0.set_ylabel("Weight")
#ax0.legend(["Model "+ str(i) for i in range(1,K+1)], loc='best',edgecolor='k')

# plot the sigma
ax1.plot(S)
#ax1.set_title('BMA Sigma for Each Models from EM Algorithm')
#ax1.set_xlabel("Number of iteration")
ax1.set_ylabel("$\\sigma$")
#ax1.legend(["Model "+ str(i) for i in range(1,K+1)], loc='best',edgecolor='k')

# plot the log likelihood
ax2.plot(LL, 'k--')
#ax2.set_title('BMA Log Likelihood from EM Algorithm')
ax2.set_xlabel("Number of iteration")
ax2.set_ylabel("Log likelihood")

plt.show()

# Output the process and final results
header_K = np.array(list(range(1,K+1)))
header_T = np.array(list(range(1,T+1))).T

W = np.row_stack((header_K,np.array(W)))
S = np.row_stack((header_K,np.array(S)))

Iter_No = np.array(list(range(i+2)))
LL = np.column_stack((Iter_No,np.array(LL)))

# calculate the BMA mean
final_W = W[-1].reshape(-1,1)
BMA_mean = np.dot(f,final_W)

# calculate the BMA variance
BMA_var = np.full([T,1],np.nan)
final_var = (S[-1].reshape(-1,1))**2

for i in range(T):
    BMA_var[i] = np.dot(final_W.T,(f[i,:]-BMA_mean[i])**2) + np.dot(final_W.T,final_var)

# (1-alpha)*100% confidence interval
alpha = 0.10

Upper_BMA_mean = BMA_mean + norm.ppf(1-alpha/2,0,1)*np.sqrt(BMA_var*(1+1/K))
Lower_BMA_mean = BMA_mean + norm.ppf(alpha/2,0,1)*np.sqrt(BMA_var*(1+1/K))

BMA_result = np.column_stack((header_T,BMA_mean,Upper_BMA_mean,Lower_BMA_mean))

BMA_var = np.column_stack((header_T,BMA_var))

np.savetxt('./Results_EM/EM_weights.csv', W, fmt='%f',delimiter = ',')
np.savetxt('./Results_EM/EM_standard_deviation.csv', S, fmt='%f',delimiter = ',')
np.savetxt('./Results_EM/EM_log_likelihood.csv', LL, fmt='%f',delimiter = ',')

np.savetxt('./Results_EM/EM_variance.csv', BMA_var, fmt='%f',header="Time,BMA_variance",delimiter = ',',comments='')
np.savetxt('./Results_EM/EM_mean.csv', BMA_result, fmt='%f',header="Time,BMA_mean,Upper_bound,Lower_bound",delimiter = ',',comments='')

print("DONE!")

