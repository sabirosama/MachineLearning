import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

credit_data = pd.read_csv('credit_data.csv')

features = credit_data[['income','age', 'loan']]
target = credit_data.default

##machine learning operates in arrays so
## to avoid indicies we have to reshape the dataset from 3
X = np.array(features).reshape(-1,3)
y = np.array(target)

model = LogisticRegression()
##cross validation for model in features and target with 5 folds
predicted = cross_validate(model, X, y, cv=5)

print(np.mean(predicted['test_score']))
