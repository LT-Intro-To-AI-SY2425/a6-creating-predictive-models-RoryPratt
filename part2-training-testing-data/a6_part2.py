import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("chirping_data.csv")
x = data["Temp"].values
y = data["Chirps"].values

# Create your training and testing datasets:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# Use reshape to turn the x values into 2D arrays:
x_train = x_train.reshape(-1,1)

# Create the model
model = LinearRegression().fit(x_train,y_train)
# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = model.coef_[0]
y_int = model.intercept_
r_squared = model.score(x_train, y_train)

# Print out the linear equation and r squared value:
print("equation: bp*", coef, " + ", y_int)
print("r squared: ", r_squared)
'''
**********TEST THE MODEL**********
'''
# reshape the xtest data into a 2D array
x_test = x_test.reshape(-1,1)
# get the predicted y values for the xtest values - returns an array of the results
prediction = model.predict(x_test)
# round the value in the np array to 2 decimal places
prediction = np.round(prediction, 2)

# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")

for i in range(len(x_test)): print("Test: ", x_test[i][0], " Prediction: ", prediction[i], " Answer: ", y_test[i])

'''
**********CREATE A VISUAL OF THE RESULTS**********
'''

xpoints = np.array(x)
ypoints = np.array(y)

plt.plot(xpoints, ypoints, 'o')

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals =  coef * x_vals + y_int
plt.plot(x_vals, y_vals, '-')

plt.show()