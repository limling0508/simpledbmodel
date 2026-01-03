# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load data
df = pd.read_csv("sales.csv")

X = df[["units_sold", "region", "product"]]
y = df["revenue"]

# Preprocessing
preprocessor = ColumnTransformer(
    [
        ("onehot", OneHotEncoder(), ["region", "product"])
    ],
    remainder="passthrough"
)

# Pipeline
model = Pipeline(
    [
        ("preprocess", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Train
model.fit(X, y)

# Save model 
joblib.dump(model, "revenue_model.pkl")
print("Model saved as revenue_model.pkl")
