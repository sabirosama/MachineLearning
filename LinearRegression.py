import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

#Read CVS

house_data = pd.read_csv('house_prices.csv')
print(house_data)
size = house_data['sqft_living']
price = house_data['price']

#convert data frame to array for ML
#reshaping for removing Id

x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)

#ML Model i.e Linear Regression, and .fit for training

model = LinearRegression()
model.fit(x,y)

#MSE and R values

regression_model_mse = mean_squared_error(x,y) #MSE value from x and y
print("MSE: ", math.sqrt(regression_model_mse)) #printing the MSE value
print("R squarred values:", model.score(x,y)) #Printing the R

#getting the B1 and B0 values after we .fit the model (training)
print(model.coef_[0]) #B1
print(model.intercept_[0]) #B0

#Visulaize the data-set with the fitted model
plt.scatter(x ,y, color ='green')
plt.plot(x, model.predict(x), color='black')
plt.title('Linear Regression')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

#Predict the price
print("Prediction by model: ", model.predict([[2000]]))