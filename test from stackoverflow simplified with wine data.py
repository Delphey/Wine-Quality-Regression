import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing

dataset = np.genfromtxt('winequality-white.csv', delimiter=',')

x = dataset[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
quality = dataset[:, 11]

result = sm.OLS(endog=quality, exog=x).fit().summary()

print(result)

datasetnormal = preprocessing.MinMaxScaler().fit_transform(dataset)

xnormal = datasetnormal[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
qualitynormal = datasetnormal[:, 11]

resultnormal = sm.OLS(endog=qualitynormal, exog=xnormal).fit().summary()

print(resultnormal)
