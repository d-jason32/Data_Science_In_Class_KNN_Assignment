### Data Science In Class KNN Assignment

## Jason Devaraj

# Install kagglehub
import kagglehub
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Download latest version
path = kagglehub.dataset_download("shashanknecrothapa/ames-housing-dataset")

print("Path to dataset files:", path)



'''
# Exersise 1
Fit a 10-nearest neighbors model to predict the SalePrice of an 1800 square foot, 3 bedroom, 2 bathroom home.

A DataFrame consisting of this "test data" is provided for you.

X_test = pd.DataFrame([

   {"Gr Liv Area": 1800,

    "Bedroom AbvGr": 3,

    "Full Bath": 2}

])

X_test

'''






'''

# Exersise 2

Exercise 2

Fit a 10 -nearest neighbors model to predict the Sale Price of a home in each neighborhood of Ames. That is, the categorical variable Neighborhood should be the only input feature to your model.

A DataFrame consisting of this "test data" is provided for you.

Can you explain intuitively what the 𝑘-nearest neighbors model is doing? For example, what is the prediction for the Bluestem ("Blueste") neighborhood?

X_test = pd.DataFrame({

   "Neighborhood": df["Neighborhood"].unique()

})

X_test

'''

'''
Exercise 3

Fit a 10-nearest neighbors model to predict the Sale Price of a 1800 square foot, 3 bedroom, 2 bathroom two-story home (House Style) in the Veenker neighborhood (Neighborhood) of Ames.

How does this compare with the predictions from the models you fit in Exercises 1 and 2?
'''