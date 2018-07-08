from statistics import mean 
import numpy as np 
import matplotlib.pyplot as plt

xs = np.array([1, 2, 3, 4, 5, 6])
ys = np.array([5, 4, 6, 5, 8, 7])
plt.scatter(xs, ys)
plt.show()


def best_fit_slope(xs, ys):
	m = ((mean(xs)*mean(ys)) - mean(xs*ys))/((mean(xs)**2 - mean(xs**2)))
	return m 

def best_fit_intercept(xs, yx, m):
	b = mean(ys) - m*mean(xs)
	return b

m = best_fit_slope(xs, ys)
b = best_fit_intercept(xs, ys, m)
print(m)
print(b)

regression_line = [(m*x)+b for x in xs]
plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()