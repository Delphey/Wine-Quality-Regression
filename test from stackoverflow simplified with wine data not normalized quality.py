import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing

#import data
dataset = np.genfromtxt('winequality-white.csv', delimiter=',')

# remove outliers in factors
##dataset = preprocessing.robust_scale(dataset)

# define factors and searched value
factors = dataset[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
quality = dataset[:, 11]


# make a version that is 0 to 1 scale
xnormal = preprocessing.MinMaxScaler().fit_transform(factors)

# add intercept to both original and normalized types
xnormal = sm.add_constant(xnormal)
x = sm.add_constant(factors)

# do OLS
result = sm.OLS(endog=quality, exog=x).fit().summary()
resultnormal = sm.OLS(endog=quality, exog=xnormal).fit().summary()

# output
print(result)
print(resultnormal)
