from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

X = pd.read_csv("LUAD_features_nogene.csv", index_col=0)
y = X["label"].values
X = X.drop(columns="label").values


train_idx, temp_idx = train_test_split(
    X,
    test_size=0.30,
    stratify=y,
    random_state=42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=y[temp_idx],
    random_state=42
)