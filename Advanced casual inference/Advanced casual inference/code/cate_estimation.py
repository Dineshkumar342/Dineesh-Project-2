import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("synthetic_data.csv")
Y_hat, T_hat = np.load("dml_crossfit_residuals.npz")["Y_hat"], np.load("dml_crossfit_residuals.npz")["T_hat"]
Y_resid = data["Y"].values - Y_hat
T_resid = data["T"].values - T_hat
final = LinearRegression(fit_intercept=False).fit(T_resid.reshape(-1,1), Y_resid)
cate = final.coef_[0]
data["CATE_est"] = cate
data.to_csv("cate_estimates.csv", index=False)
print(f"Estimated CATE (DML): {cate:.3f}")
