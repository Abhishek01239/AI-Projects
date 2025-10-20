import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns = data.feature_names)

print("Original Data Shape:" , X.shape)
print("Columns: ", X.columns.tolist())

standard = StandardScaler()
minmax = MinMaxScaler()
robust = RobustScaler()

X_standard= pd.DataFrame(standard.fit_transform(X), columns = X.columns)
X_minmax= pd.DataFrame(minmax.fit_transform(X), columns = X.columns)
X_robust= pd.DataFrame(robust.fit_transform(X), columns = X.columns)

feature = "MedInc"

plt.figure(figsize=(12,6))
plt.suptitle(f"Feature Scaling Comparison for {feature}", fontsize =14, fontweight = "bold")

plt.subplot(1,3,1)
plt.title("StandardScaler")
plt.hist(X_standard[feature], bins = 30, color= "skyblue", edgecolor = "black")

plt.subplot(1,3,2)
plt.title("MinMaxScaler")
plt.hist(X_minmax[feature], bins = 30, color = "orange", edgecolor = "black")

plt.subplot(1,3,3)
plt.title("RobustScaler")
plt.hist(X_robust[feature], bins = 30, color = "green", edgecolor = "black")

plt.tight_layout(rect = [0,0,1,0.95])
plt.show()

print(f"\n Mean & Std for '{feature}:"
      )
print(f"Original -> Mean: {X[feature].mean():.2f}, Std: {X[feature].std():.2f}")
print(f"Standard -> Mean: {X_standard[feature].mean():.2f}, Std: {X_standard[feature].std():.2f}")
print(f"MinMax -> Min: {X_minmax[feature].min():.2f}, Max: {X_minmax[feature].max():.2f}")
print(f"Robust -> Median: {X_robust[feature].median():.2f}")


