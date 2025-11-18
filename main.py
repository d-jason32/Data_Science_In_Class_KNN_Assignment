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
import numpy as np


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


    # Input feature: neighborhood
    X = df[["Neighborhood"]]
    y = df["SalePrice"]

    # One-hot encode Neighborhood
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X)

    # 10 nearest neighbors
    knn2 = KNeighborsRegressor(n_neighbors=10)
    knn2.fit(X_encoded, y)

    # Test data: each unique neighborhood
    X_test = pd.DataFrame({
        "Neighborhood": df["Neighborhood"].unique()
    })

    # Encode test data
    X_test_encoded = encoder.transform(X_test)

    # Predictions for each neighborhood
    predictions = knn2.predict(X_test_encoded)

    for neigh, pred in zip(df["Neighborhood"].unique(), predictions):
        print(f"{neigh}: ${pred:,.2f}")

'''
The KNN model in this case is taking in the neighborhood as the input. The output is the saleprice.
The model will take one specific point and see which 10 other points are closest.
It sees the category of those other 10 points and if they are the majority,
it will be classified in that way.

For bluestem, the prediction is  $143,590.00
'''

'''
Exercise 3

Fit a 10-nearest neighbors model to predict the Sale Price of a 1800 square foot, 3 bedroom, 2 bathroom two-story home (House Style) in the Veenker neighborhood (Neighborhood) of Ames.
'''




if __name__ == "__main__":
    exercise1()
    exercise2()
