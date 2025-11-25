import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 2000
n_covariates = 50

# Generate confounders
X = np.random.randn(n_samples, n_covariates)
# Heterogeneous true treatment effect
tau = 2 + X[:,0] - 0.5*X[:,1]
# Propensity score (nonlinear, depends on some X)
ps = 1 / (1 + np.exp(-0.5*X[:,5] + 0.25*X[:,10] - 0.2*X[:,15])) 
T = np.random.binomial(1, ps)
# Outcome model
Y = (5 + tau*T + 0.5*X[:,0] + 1.5*X[:,1] - X[:,2] + np.random.normal(0, 1, n_samples))

data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(n_covariates)])
data["T"] = T
data["Y"] = Y
data.to_csv("synthetic_data.csv", index=False)
