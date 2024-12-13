import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1,1)

# Create the model
model = LinearRegression().fit(x,y)
# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = model.coef_[0]
y_int = model.intercept_
r_squared = model.score

# Print out the linear equation and r squared value
print("equation: bp*", coef, " + ", y_int)
print("r squared: ", r_squared)
# Predict the the blood pressure of someone who is 43 years old.
# Print out the prediction

print("Bp of someone who is 40: ", model.predict([[44]])[0])

# Create the model in matplotlib and include the line of best fit


xpoints = np.array(x)
ypoints = np.array(y)

plt.plot(xpoints, ypoints, 'o')

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals =  coef * x_vals + y_int
plt.plot(x_vals, y_vals, '-')

plt.show()