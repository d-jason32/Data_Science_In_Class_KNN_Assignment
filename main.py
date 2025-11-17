### Data Science In Class KNN Assignment

## Jason Devaraj

# Install kagglehub
import kagglehub
import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

'''
Exercise 1

Fit a 10-nearest neighbors model to predict the SalePrice of an 1800 square foot, 3 bedroom, 2 bathroom home.

A DataFrame consisting of this "test data" is provided for you.

X_test = pd.DataFrame([

   {"Gr Liv Area": 1800,

    "Bedroom AbvGr": 3,

    "Full Bath": 2}

])

X_test'''
# File from Kaggle

path = kagglehub.dataset_download("shashanknecrothapa/ames-housing-dataset")

# Show the path to the file
# print("Path to dataset files:", path)

# print(os.listdir(path))

# The path to the CSV
csv_path = os.path.join(path, "AmesHousing.csv")
df = pd.read_csv(csv_path)

# Exercise 1

def exercise1():
    # the input features
    X = df[["Gr Liv Area",

            "Bedroom AbvGr",

            "Full Bath"]]

    # the output features
    y = df["SalePrice"]


    # 10 nearest neighbors
    knn = KNeighborsRegressor(n_neighbors=10)

    # Fit Model
    knn.fit(X, y)

    # Test Data
    X_test = (pd
    .DataFrame([
        {
            "Gr Liv Area":
                1800,

            "Bedroom AbvGr":

                3,
            "Full Bath":

                2
        }
    ]))

    # Prediction
    print(knn.predict(X_test))


'''
Exercise 2
Fit a 10 -nearest neighbors model to predict the Sale Price of a home in each neighborhood of Ames. That is, the categorical variable Neighborhood should be the only input feature to your model.

A DataFrame consisting of this "test data" is provided for you.

Can you explain intuitively what the 𝑘-nearest neighbors model is doing? For example, what is the prediction for the Bluestem ("Blueste") neighborhood?

X_test = pd.DataFrame({

   "Neighborhood": df["Neighborhood"].unique()

})

X_test
'''
def exercise2():


    # Input feature: Neighborhood (categorical)
    X2 = df[["Neighborhood"]]
    y2 = df["SalePrice"]

    # One-hot encode Neighborhood
    encoder = OneHotEncoder(sparse_output=False)
    X2_encoded = encoder.fit_transform(X2)

    # 10 nearest neighbors regressor
    knn2 = KNeighborsRegressor(n_neighbors=10)
    knn2.fit(X2_encoded, y2)

    # Test data: each unique neighborhood
    X2_test = pd.DataFrame({
        "Neighborhood": df["Neighborhood"].unique()
    })

    # Encode test data
    X2_test_encoded = encoder.transform(X2_test)

    # Predictions for each neighborhood
    predictions2 = knn2.predict(X2_test_encoded)

    print("\nExercise 2 — Predicted SalePrice per Neighborhood:")
    for neigh, pred in zip(df["Neighborhood"].unique(), predictions2):
        print(f"{neigh}: ${pred:,.2f}")

'''
Exercise 3

Fit a 10-nearest neighbors model to predict the Sale Price of a 1800 square foot, 3 bedroom, 2 bathroom two-story home (House Style) in the Veenker neighborhood (Neighborhood) of Ames.
'''

def exercise3():
    # Input features: numeric + categorical
    X3 = df[[
        "Gr Liv Area",
        "Bedroom AbvGr",
        "Full Bath",
        "House Style",
        "Neighborhood",
    ]]
    y3 = df["SalePrice"]

    # Split into numeric and categorical parts
    X3_numeric = X3[["Gr Liv Area", "Bedroom AbvGr", "Full Bath"]]
    X3_cats = X3[["House Style", "Neighborhood"]]

    # One-hot encode the categorical features
    encoder3 = OneHotEncoder(sparse_output=False)
    X3_cats_encoded = encoder3.fit_transform(X3_cats)

    import numpy as np

    # Combine numeric and encoded categorical features
    X3_combined = np.hstack([X3_numeric.values, X3_cats_encoded])

    # 10-nearest neighbors regressor
    knn3 = KNeighborsRegressor(n_neighbors=10)
    knn3.fit(X3_combined, y3)

    # Test data for the specific home
    X3_test = pd.DataFrame([
        {
            "Gr Liv Area": 1800,
            "Bedroom AbvGr": 3,
            "Full Bath": 2,
            "House Style": "2Story",
            "Neighborhood": "Veenker",
        }
    ])

    X3_test_numeric = X3_test[["Gr Liv Area", "Bedroom AbvGr", "Full Bath"]]
    X3_test_cats = X3_test[["House Style", "Neighborhood"]]

    X3_test_cats_encoded = encoder3.transform(X3_test_cats)
    X3_test_combined = np.hstack([X3_test_numeric.values, X3_test_cats_encoded])

    prediction3 = knn3.predict(X3_test_combined)[0]

    print("\nExercise 3 — Predicted SalePrice for 1800 sq ft, 3 bed, 2 bath, 2Story in Veenker:")
    print(f"${prediction3:,.2f}")





if __name__ == "__main__":
    exercise1()
    exercise2()
    exercise3()