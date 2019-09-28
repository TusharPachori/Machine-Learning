import os
import pandas as pd
import matplotlib.pyplot as plt
import hashlib

HOUSING_PATH = "datasets/housing"


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
print(housing.head())
housing.info()

housing.hist(bins=50, figsize=(20,15))
plt.show()

