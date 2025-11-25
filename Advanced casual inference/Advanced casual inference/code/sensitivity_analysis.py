import pandas as pd
data = pd.read_csv("cate_estimates.csv")
for col in ["X1", "X2", "X3"]:
    below = data[data[col] < data[col].median()]['CATE_est'].mean()
    above = data[data[col] >= data[col].median()]['CATE_est'].mean()
    print(f"CATE for {col}<median: {below:.3f}, {col}>=median: {above:.3f}")
