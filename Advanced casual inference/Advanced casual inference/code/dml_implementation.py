import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

data = pd.read_csv("synthetic_data.csv")
X = data.iloc[:, :-2].values
T = data["T"].values
Y = data["Y"].values

n_splits = 2
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
Y_hat = np.zeros_like(Y)
T_hat = np.zeros_like(T, dtype=float)
for train_idx, test_idx in kf.split(X):
    # Fit ML for outcome
    mu_model = RandomForestRegressor(random_state=42).fit(X[train_idx], Y[train_idx])
    Y_hat[test_idx] = mu_model.predict(X[test_idx])
    # Fit ML for treatment
    m_model = RandomForestRegressor(random_state=42).fit(X[train_idx], T[train_idx])
    T_hat[test_idx] = m_model.predict(X[test_idx])
np.savez("dml_crossfit_residuals.npz", Y_hat=Y_hat, T_hat=T_hat)
