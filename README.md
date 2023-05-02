# Bayesian Model Averaging for Ensemble Flood Modeling
These Python scripts are developed for estimating the Bayesian Model Averaging (BMA) parameters (weights and variances) of ensemble flood modeling by using the Expectation-Maximization (EM) algorithm and the multiple Markov Chains Monte Carlo (MCMC) algorithm. For more information, please refer to the paper, “[<b><i> Improving Bayesian Model Averaging for Ensemble Flood Modeling Using Multiple Markov Chains Monte Carlo Sampling </b></i>](https://www.authorea.com/doi/full/10.22541/essoar.168056821.18559558)” (Huang and Merwade, 2023).

A brief introduction to the features of each Python script is as follows.

(1) [1-Numerical_Experiment-Ensemble.py](https://github.com/huan1441/Bayesian-Model-Averaging-for-Ensemble-Flood-Modeling/blob/main/1-Numerical_Experiment-Ensemble.py) is developed to generate the ensemble of HEC-RAS simulations for water stage by adding multiple white-noise errors into the observed data.

(2) [2-BMA_EM.py](https://github.com/huan1441/Bayesian-Model-Averaging-for-Ensemble-Flood-Modeling/blob/main/2-BMA_EM.py) is developed to estimate BMA parameters by using the EM Algorithm.

(3.1) [3.1-BMA_MCMC_Uniform.py](https://github.com/huan1441/Bayesian-Model-Averaging-for-Ensemble-Flood-Modeling/blob/main/3.1-BMA_MCMC_Uniform.py) is developed to estimate BMA parameters by using the Metropolis-Hastings MCMC algorithm, in which the proposal distribution is the uniform distribution.

(3.2) [3.2-BMA_MCMC_Normal_Gamma.py](https://github.com/huan1441/Bayesian-Model-Averaging-for-Ensemble-Flood-Modeling/blob/main/3.2-BMA_MCMC_Normal_Gamma.py) is developed to estimate BMA parameters by using the Metropolis-Hastings MCMC algorithm, in which the proposal distribution is the normal distribution and the conditional PDF of observed data is Gamma instead of Normal.

(3.3) [3.3-BMA_MCMC_PostProcess.py](https://github.com/huan1441/Bayesian-Model-Averaging-for-Ensemble-Flood-Modeling/blob/main/3.3-BMA_MCMC_PostProcess.py) is developed to post-process (trace plots, ACFs, histograms) for BMA parameters from M-H MCMC Algorithm.

(4.1) [4.1-Sampling_BMA_Normal.py](https://github.com/huan1441/Bayesian-Model-Averaging-for-Ensemble-Flood-Modeling/blob/main/4.1-Sampling_BMA_Normal.py) is developed to sample water stage predictions based on BMA parameters (Normal PDF).

(4.2) [4.2-Sampling_BMA_Gamma.py](https://github.com/huan1441/Bayesian-Model-Averaging-for-Ensemble-Flood-Modeling/blob/main/4.2-Sampling_BMA_Gamma.py) is developed to sample water stage predictions based on BMA parameters (Gamma PDF).

(5) [5-Visualizing_BMA_Parameters.py](https://github.com/huan1441/Bayesian-Model-Averaging-for-Ensemble-Flood-Modeling/blob/main/5-Visualizing_BMA_Parameters.py) is developed to create 3D histograms for BMA parameters from M-H MCMC Algorithm.
