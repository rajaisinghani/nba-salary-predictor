import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression

# importing the dataset
salary_file_path = '2017-18_NBA_salary.csv'
salary_data = pd.read_csv(salary_file_path)

#visualising the data
salary_data.Salary.hist(bins=40, alpha=1.0)
plt.title("NBA Players' Salaries in 2017-18 Season Histogram")
plt.xlabel("Salary($)")
plt.ylabel("Frequency")

#feature selection

#remove the team and country columns (irrelevant)
salary_data = salary_data.dropna()
salary_data = salary_data.drop("Tm", axis=1)
salary_data = salary_data.drop("NBA_Country", axis=1)

#extract np array from df
X = salary_data.iloc[:, 2:]
y = salary_data.iloc[:, 1]

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=f_regression, k=12)
fit = bestfeatures.fit(X,y)
selected_features = fit.fit_transform(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#print(featureScores.nlargest(12,'Score'))  #print 10 best features
#print(fit.get_support(indices=True))

#perform regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size = 0.25)

from sklearn.linear_model import LinearRegression
classifier = LinearRegression()
classifier.fit(X_train, y_train)
#print(classifier.coef_)
y_pred = classifier.predict(X_test)

predicted = classifier.predict(selected_features)
salary_data['predicted'] = predicted

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(salary_data['Salary'].mean())


##### without feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

classifier = LinearRegression()
classifier.fit(X_train, y_train)
#print(classifier.coef_)
y_pred = classifier.predict(X_test)

print('Mean Absolute Error (without feature selection):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (without feature selection):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (without feature selection):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(salary_data['Salary'].mean())