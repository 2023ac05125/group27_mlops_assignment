import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

def save_raw(path="data/raw/housing.csv"):
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print("Saved raw dataset to:", path)

if __name__ == '__main__':
    save_raw()
