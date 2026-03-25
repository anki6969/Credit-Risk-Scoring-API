import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# load dataset
df = pd.read_csv("data/loan.csv")

# target: Y = approved (0), N = default (1)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 0, "N": 1})

# handle missing values
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["Credit_History"].fillna(1.0, inplace=True)

df["ApplicantIncome"].fillna(df["ApplicantIncome"].median(), inplace=True)
df["CoapplicantIncome"].fillna(df["CoapplicantIncome"].median(), inplace=True)

# features
X = df[
    [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History"
    ]
]

y = df["Loan_Status"]

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# model
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_scaled, y)

# save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("Model trained successfully")
