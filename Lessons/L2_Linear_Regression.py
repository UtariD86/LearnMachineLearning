#libraries
import imp
from pyexpat.model import XML_CQUANT_REP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-----codes-----
#>>>data loads

datapath = "C:\\Users\\l340\\Desktop\Dolap\\LearnPython\\LearnMachineLearning\\Lessons\\dataLib\\sales.csv"
data = pd.read_csv(datapath)


#>>>data processing

months = data[["Aylar"]]

sales = data[["Satislar"]]

sales2 = data.iloc[:,:1].values


#>>>split datas as train and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size = 0.33, random_state = 0) 


#>>>Data Scaling
'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()


X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''

#>>>linear regration model construction

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train, y_train)

prediction = lr.predict(x_test)


#>>>sorting

x_train = x_train.sort_index()
y_train = y_train.sort_index()


#visualization

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.show()

