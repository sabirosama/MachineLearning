import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix ##for analyzing the results
from sklearn.model_selection import train_test_split ##for training and testing the data
from sklearn.metrics import accuracy_score  ##to measure the accuracy of the model

credit_data = pd.read_csv('credit_data.csv')



#print(credit_data.head()) ##prints the first five values of the data
#print(credit_data.describe()) ##gives numerical stats about total count,mean,std,min max etc
#print((credit_data.corr()) ##relation between variables

features = credit_data[['income', 'age', 'loan']]
target = credit_data.default  ##default is the target data

## 30% of tha data_test is for testing,70% is for training
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(feature_train, target_train) ##put the training part in model fit

print(model.intercept_) ##for Bo
print(model.coef_) ##for B1,B2,B3

prediction = model.fit.predict(feature_test) ##predict the test features

print(confusion_matrix(target_test,prediction))
print(accuracy_score(target_test,prediction))


