import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("car_data.csv")
x = data[["miles","age"]].values
y = data["Price"].values

#split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
#create linear regression model
x_train = x_train.reshape(-1, 2)

model = LinearRegression().fit(x_train, y_train)
#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 
coef = model.coef_
y_int = model.intercept_
r_squared = model.score(x, y)

print(coef)
print(y_int)
print(r_squared)
#Loop through the data and print out the predicted prices and the 

prediction = model.predict(x_test)
prediction = np.round(prediction, 2)
#actual prices
print("***************")
print("Testing Results")

for i in range(len(x_test)): print("Test: ", x_test[i], " Prediction: ", prediction[i], " Answer: ", y_test[i])


ans = 0
for i in range(len(prediction)):
	p = prediction[i]
	a = y_test[i]
	ans += (abs(p-a)/a)/(len(prediction))
ans = 1 - ans
print("accuracy: ", ans)

print(model.predict([[150000, 20]]))