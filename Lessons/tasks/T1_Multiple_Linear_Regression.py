#libraries
from statistics import mode
import pandas as pd
import numpy as np
import matplotlib as plt

#-----codes-----
#>>>data loads

datapath = "D:\\Dolap\\LearnPython\\LearnMachineLearning\\Lessons\\dataLib\\task1_tennis.csv"
data = pd.read_csv(datapath)


#>>>data processing

temperature = data.iloc[:,1:2]

humidity = data.iloc[:,2:3]


#>>> work with categorized datas

outlook = data.iloc[:,:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

outlook[:,0] = le.fit_transform(data.iloc[:,0])

ohe = preprocessing.OneHotEncoder()

outlook = ohe.fit_transform(outlook).toarray()

play = data.iloc[:,-1:].values

play[:,-1] = le.fit_transform(data.iloc[:,-1]).astype(float)

windy = data.iloc[:,3:4].values.astype(float)

windy[:,-1] = le.fit_transform(data.iloc[:,3:4])


#>>> combine datas and crate dataframes

windy = pd.DataFrame(data = windy, index = range(14), columns = ["windy"])

dataresult = pd.DataFrame(data = outlook, index = range(14), columns = ["overcast","rainy","sunny"])

dataresult2 = pd.DataFrame(data = play, index = range(14), columns = ["play"])

lastresult = pd.concat([dataresult, windy], axis = 1)

lastresult2 = pd.concat([lastresult, temperature], axis = 1)

lastresult3 = pd.concat([lastresult2, dataresult2], axis = 1)

lastresult4 = pd.concat([lastresult3, humidity], axis = 1)


#>>>split datas as train and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(lastresult4.iloc[:,:-1], lastresult4.iloc[:,-1:], test_size = 0.33, random_state = 0)


#>>> Train machine with datas

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)


#>>> Asking to predict after training

y_pred = regressor.predict(x_test)


#>>> Adding β0 (constant multiplier) to the data

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values = lastresult4.iloc[:,:-1], axis = 1)

X_list = lastresult4.iloc[:,[0,1,2,3,4,5]].values
X_list = np.array(X_list, dtype=float )


#>>> Measuring the impact of data on the prediction with the "Ordinary Least Squares" (OLS) method

model = sm.OLS(lastresult4.iloc[:,-1:], X_list).fit()

print(model.summary())


#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:               humidity   R-squared:                       0.294
# Model:                            OLS   Adj. R-squared:                 -0.148
# Method:                 Least Squares   F-statistic:                    0.6653
# Date:                Mon, 29 Aug 2022   Prob (F-statistic):              0.661
# Time:                        19:42:37   Log-Likelihood:                -49.542
# No. Observations:                  14   AIC:                             111.1
# Df Residuals:                       8   BIC:                             114.9
# Df Model:                           5
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# x1            52.3891     50.214      1.043      0.327     -63.404     168.183
# x2            55.6337     45.233      1.230      0.254     -48.673     159.940
# x3            49.4291     48.337      1.023      0.336     -62.035     160.893
# x4            -4.0286      7.229     -0.557     >0.593<    -20.698      12.641
# x5             0.4920      0.597      0.825      0.433      -0.884       1.868
# x6            -8.2778      8.029     -1.031      0.333     -26.793      10.237
# ==============================================================================
# Omnibus:                        0.935   Durbin-Watson:                   2.416
# Prob(Omnibus):                  0.627   Jarque-Bera (JB):                0.823
# Skew:                           0.389   Prob(JB):                        0.663
# Kurtosis:                       2.103   Cond. No.                     2.08e+03
# ==============================================================================


#>>> First step of Backward Elimination by p-value

X_list = lastresult4.iloc[:,[0,1,2,3,5]].values
X_list = np.array(X_list, dtype=float )


#>>> OLS again

model = sm.OLS(lastresult4.iloc[:,-1:], X_list).fit()

print(model.summary())


#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:               humidity   R-squared:                       0.234
# Model:                            OLS   Adj. R-squared:                 -0.107
# Method:                 Least Squares   F-statistic:                    0.6859
# Date:                Mon, 29 Aug 2022   Prob (F-statistic):              0.619
# Time:                        19:46:21   Log-Likelihood:                -50.114
# No. Observations:                  14   AIC:                             110.2
# Df Residuals:                       9   BIC:                             113.4
# Df Model:                           4
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# x1            92.8464     10.538      8.811      0.000      69.008     116.685
# x2            92.3911      7.587     12.177      0.000      75.227     109.555
# x3            88.9050      6.621     13.429      0.000      73.928     103.882
# x4            -6.8324      6.265     -1.091     >0.304<    -21.005       7.340
# x5           -10.4302      7.457     -1.399      0.195     -27.299       6.439
# ==============================================================================
# Omnibus:                        0.336   Durbin-Watson:                   2.056
# Prob(Omnibus):                  0.845   Jarque-Bera (JB):                0.466
# Skew:                           0.071   Prob(JB):                        0.792
# Kurtosis:                       2.118   Cond. No.                         5.27
# ==============================================================================


#>>> First step of Backward Elimination by p-value

X_list = lastresult4.iloc[:,[0,1,2,5]].values
X_list = np.array(X_list, dtype=float )


#>>> OLS again

model = sm.OLS(lastresult4.iloc[:,-1:], X_list).fit()

print(model.summary())


#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:               humidity   R-squared:                       0.132
# Model:                            OLS   Adj. R-squared:                 -0.128
# Method:                 Least Squares   F-statistic:                    0.5085
# Date:                Mon, 29 Aug 2022   Prob (F-statistic):              0.685
# Time:                        19:50:13   Log-Likelihood:                -50.982
# No. Observations:                  14   AIC:                             110.0
# Df Residuals:                      10   BIC:                             112.5
# Df Model:                           3
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# x1            86.5833      8.919      9.708      0.000      66.711     106.456
# x2            87.9500      6.462     13.610      0.000      73.551     102.349
# x3            85.0333      5.641     15.075      0.000      72.465      97.602
# x4            -7.5833      7.051     -1.076     >0.307<    -23.294       8.127
# ==============================================================================
# Omnibus:                        0.290   Durbin-Watson:                   1.909
# Prob(Omnibus):                  0.865   Jarque-Bera (JB):                0.418
# Skew:                          -0.253   Prob(JB):                        0.811
# Kurtosis:                       2.321   Cond. No.                         3.89
# ==============================================================================


#>>> First step of Backward Elimination by p-value

X_list = lastresult4.iloc[:,[0,1,2]].values
X_list = np.array(X_list, dtype=float )


#>>> Last OLS for check

model = sm.OLS(lastresult4.iloc[:,-1:], X_list).fit()

print(model.summary())


#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:               humidity   R-squared:                       0.032
# Model:                            OLS   Adj. R-squared:                 -0.144
# Method:                 Least Squares   F-statistic:                    0.1818
# Date:                Mon, 29 Aug 2022   Prob (F-statistic):              0.836
# Time:                        19:52:04   Log-Likelihood:                -51.749
# No. Observations:                  14   AIC:                             109.5
# Df Residuals:                      11   BIC:                             111.4
# Df Model:                           2
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# x1            79.0000      5.500     14.363      0.000      66.894      91.106
# x2            83.4000      4.920     16.952      0.000      72.572      94.228
# x3            82.0000      4.920     16.668      0.000      71.172      92.828
# ==============================================================================
# Omnibus:                        4.583   Durbin-Watson:                   2.107
# Prob(Omnibus):                  0.101   Jarque-Bera (JB):                1.347
# Skew:                          -0.146   Prob(JB):                        0.510
# Kurtosis:                       1.509   Cond. No.                         1.12
# ==============================================================================