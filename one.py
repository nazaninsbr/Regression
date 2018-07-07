import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_prediction = regr.predict(diabetes_X_test)

print('Coefficients: ', regr.coef_)
print('MSE: %.2f' % mean_squared_error(diabetes_y_test, diabetes_y_prediction))

# plt.scatter(diabetes_X_train, diabetes_y_train, color='black')
plt.scatter(diabetes_X_test, diabetes_y_test, color='blue')
plt.plot(diabetes_X_test, diabetes_y_prediction, color='pink')

plt.show()