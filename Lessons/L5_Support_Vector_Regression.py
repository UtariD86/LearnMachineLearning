#libraries
from statistics import mode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-----codes----
#>>> Data loads

datapath = "D:\\Dolap\\LearnPython\\LearnMachineLearning\\Lessons\\dataLib\\salaries.csv"
data = pd.read_csv(datapath)


#>>> Data processing

x = data.iloc[:,1:-1]
y = data.iloc[:,-1:]


#>>> Converting to numpy array
X = x.values
Y = y.values


#>>> Split datas as train and test(I skip this step as i will be using all the data for the training.)

# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(, , test_size = , random_state = )


#>>> Train machine with datas (Linear Regression)

from sklearn.linear_model import LinearRegression

lReg = LinearRegression()

lReg.fit(X, Y)

#>visualization
plt.scatter(X, Y)
plt.plot(x,lReg.predict(X))
plt.show()


#>>> Train machine with datas (Polynomial Regression)

from sklearn.preprocessing import PolynomialFeatures

pReg = PolynomialFeatures(degree = 2)

x_poly = pReg.fit_transform(X)

lReg2 = LinearRegression()

lReg2.fit(x_poly, y)

#>visualization
plt.scatter(X, Y)
plt.plot(X, lReg2.predict(pReg.fit_transform(X)))
plt.show()


#>>> Changed degree

pReg2 = PolynomialFeatures(degree = 4)

x_poly = pReg2.fit_transform(X)

lReg3 = LinearRegression()

lReg3.fit(x_poly, y)

#>visualization
plt.scatter(X, Y)
plt.plot(X, lReg3.predict(pReg2.fit_transform(X)))
plt.show()


#>>> Asking to predict after training

print(lReg.predict([[11]]))
print(lReg.predict([[6.6]]))

print(lReg2.predict(pReg.fit_transform([[11]])))
print(lReg2.predict(pReg.fit_transform([[6.6]])))

print(lReg3.predict(pReg2.fit_transform([[11]])))
print(lReg3.predict(pReg2.fit_transform([[6.6]])))


#>>> Data Scaling

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
sc2 = StandardScaler()

x_scaled = sc1.fit_transform(X)
y_scaled = sc2.fit_transform(Y)


#>>> Using SVR

from sklearn.svm import SVR

sReg = SVR(kernel = "rbf")
sReg.fit(x_scaled, y_scaled)

plt.scatter(x_scaled, y_scaled, color = "red")
plt.plot(x_scaled, sReg.predict(x_scaled), color = "blue")
plt.show()

print(sReg.predict([[11]]))
print(sReg.predict([[6.6]]))