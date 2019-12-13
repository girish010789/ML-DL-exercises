##predict company profit based on the multiple features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#First play around with the data by plotting it first to get a fair idea of the dataset we are dealing with
companies = pd.read_csv("/Users/gr959202/1000_companies.csv")
X = companies.iloc[:,:-1 ].values
Y = companies.iloc[:,4].values
print(companies.head)

##plot the heatmap of the dataset
s = sns.heatmap(companies.corr())
plt.show(s)

##setting up linear regression model
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3], handle_unknown='ignore')
X = onehotencoder.fit_transform(X).toarray()
print("Encoded input data is", X)

##avoiding dummy variable trap
X = X[:,1:]

##2 steps of linear regression
##1. split data to train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

##2. create a linear regression model
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

##predicting the test set results
Y_pred = regressor.predict(X_test)

##find coefficient and intercept 
print("Coefficient m is :", regressor.coef_)
print("Intercept c is :" ,regressor.intercept_)

#evaluate the summation of squared error value
print("Cost function:", r2_score(Y_test,Y_pred))
