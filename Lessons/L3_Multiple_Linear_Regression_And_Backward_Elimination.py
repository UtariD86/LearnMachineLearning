#libraries
import imp
import pandas as pd
import numpy as np
import matplotlib as plt

#-----codes----
#>>> Data loads

datapath = "D:\\Dolap\\LearnPython\\LearnMachineLearning\\Lessons\\dataLib\\datas.csv"
data = pd.read_csv(datapath)


#>>> Data processing

age = data.iloc[:,1:4].values


#>>> Work with categorized datas

country = data.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

country[:,0] = le.fit_transform(data.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()


#>>> Work with categorized datas

altgender = data.iloc[:,-1:].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

altgender[:,-1] = le.fit_transform(data.iloc[:,-1])

ohe = preprocessing.OneHotEncoder()
altgender = ohe.fit_transform(altgender).toarray()


#>>> Combine datas and crate dataframes

dataresult = pd.DataFrame(data = country, index = range(22), columns = ["france","turkey","united states"])

dataresult2 = pd.DataFrame(data = age, index = range(22), columns = ["Boy","Kilo","Yaş"])

gender = data.iloc[:,-1].values

dataresult3 = pd.DataFrame(data = altgender[:,:1], index = range(22), columns = ["Cinsiyet"])

lastresult = pd.concat([dataresult, dataresult2], axis = 1)

lastresult2 = pd.concat([lastresult, dataresult3], axis = 1)


#>>> Split datas as train and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(lastresult, dataresult3, test_size = 0.33, random_state = 0) 

#>>> Train machine with datas

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#>>> Asking to predict after training

y_pred = regressor.predict(x_test)


#>>> Same things for heights

height = lastresult2.iloc[:,3:4].values


#>>> Getting other datas for training

left = lastresult2.iloc[:,:3]
right = lastresult2.iloc[:,4:]


#>>> Combine datas

heightdatas = pd.concat([left, right], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(heightdatas, height, test_size = 0.33, random_state = 0)


#>>> Prediction

from sklearn.linear_model import LinearRegression
regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)

y_predheight = regressor2.predict(x_test)


#>>> Adding β0 (constant multiplier) to the data

import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values = heightdatas, axis = 1)

X_list = heightdatas.iloc[:,[0,1,2,3,4,5]].values
X_list = np.array(X_list, dtype=float )


#>>> Measuring the impact of data on the prediction with the "Ordinary Least Squares" (OLS) method

model = sm.OLS(height, X_list).fit()

print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.885
# Model:                            OLS   Adj. R-squared:                  0.849
# Method:                 Least Squares   F-statistic:                     24.69
# Date:                Tue, 09 Aug 2022   Prob (F-statistic):           5.41e-07
# Time:                        15:46:01   Log-Likelihood:                -73.950
# No. Observations:                  22   AIC:                             159.9
# Df Residuals:                      16   BIC:                             166.4
# Df Model:                           5
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# x1           114.0688      8.145     14.005      0.000      96.802     131.335
# x2           108.3030      5.736     18.880      0.000      96.143     120.463
# x3           104.4714      9.195     11.361      0.000      84.978     123.964
# x4             0.9211      0.119      7.737      0.000       0.669       1.174
# x5             0.0814      0.221      0.369    ->0.717<-      -0.386       0.549 
# x6           -10.5980      5.052     -2.098      0.052     -21.308       0.112
# ==============================================================================
# Omnibus:                        1.031   Durbin-Watson:                   2.759
# Prob(Omnibus):                  0.597   Jarque-Bera (JB):                0.624
# Skew:                           0.407   Prob(JB):                        0.732
# Kurtosis:                       2.863   Cond. No.                         524.
# ==============================================================================

#>>> First step of Backward Elimination by p-value

X_list = heightdatas.iloc[:,[0,1,2,3,5]].values
X_list = np.array(X_list, dtype=float )


#>>> OLS again

model = sm.OLS(height, X_list).fit()

print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.884
# Model:                            OLS   Adj. R-squared:                  0.857
# Method:                 Least Squares   F-statistic:                     32.47
# Date:                Tue, 09 Aug 2022   Prob (F-statistic):           9.32e-08
# Time:                        17:01:31   Log-Likelihood:                -74.043
# No. Observations:                  22   AIC:                             158.1
# Df Residuals:                      17   BIC:                             163.5
# Df Model:                           4
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# x1           115.6583      6.734     17.175      0.000     101.451     129.866
# x2           109.0786      5.200     20.978      0.000      98.108     120.049
# x3           106.5445      7.090     15.026      0.000      91.585     121.504
# x4             0.9405      0.104      9.029      0.000       0.721       1.160
# x5           -11.1093      4.733     -2.347    ->0.031<-     -21.096      -1.123
# ==============================================================================
# Omnibus:                        0.871   Durbin-Watson:                   2.719
# Prob(Omnibus):                  0.647   Jarque-Bera (JB):                0.459
# Skew:                           0.351   Prob(JB):                        0.795
# Kurtosis:                       2.910   Cond. No.                         397.
# ==============================================================================


#>>> Second step of Backward Elimination by p-value

X_list = heightdatas.iloc[:,[0,1,2,3]].values
X_list = np.array(X_list, dtype=float )


#>>> Last OLS for check

model = sm.OLS(height, X_list).fit()

print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.847
# Model:                            OLS   Adj. R-squared:                  0.821
# Method:                 Least Squares   F-statistic:                     33.16
# Date:                Tue, 09 Aug 2022   Prob (F-statistic):           1.52e-07
# Time:                        17:04:57   Log-Likelihood:                -77.131
# No. Observations:                  22   AIC:                             162.3
# Df Residuals:                      18   BIC:                             166.6
# Df Model:                           3
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# x1           119.8136      7.265     16.491      0.000     104.550     135.077
# x2           109.8084      5.804     18.919      0.000      97.615     122.002
# x3           114.4212      6.984     16.382      0.000      99.747     129.095
# x4             0.7904      0.092      8.595      0.000       0.597       0.984
# ==============================================================================
# Omnibus:                        2.925   Durbin-Watson:                   2.855
# Prob(Omnibus):                  0.232   Jarque-Bera (JB):                1.499
# Skew:                           0.605   Prob(JB):                        0.473
# Kurtosis:                       3.416   Cond. No.                         369.
# ==============================================================================