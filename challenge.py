import pandas as pd # For reading dataset
from sklearn import linear_model # Machine learning library
from sklearn import metrics
import matplotlib.pyplot as plt # For visualization

# Read data from file
dataframe = pd.read_csv('challenge_dataset.txt',names=['x','y'])
x_values = dataframe[['x']]
y_values = dataframe[['y']]

# Train model on data
model_linear = linear_model.LinearRegression()
model_linear.fit(x_values, y_values)

# Visualize data
print(dataframe)
print(x_values)
print(y_values)

# Find out the mean squared error
y_predicted = model_linear.predict(x_values)
error = metrics.mean_squared_error(y_values, y_predicted)
msErr = 'Mean Squared Error = '+ str(error)
print(msErr)

# Plot data
plt.scatter(x_values, y_values)
plt.plot(x_values, model_linear.predict(x_values), label=msErr)
plt.legend()
plt.show()
