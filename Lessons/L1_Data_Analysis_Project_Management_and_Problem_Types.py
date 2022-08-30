#libraries
import imp
import pandas as pd
import numpy as np
import matplotlib as plt

#-----codes-----
#>>>data loads

datapath = "C:\\Users\\l340\\Desktop\Dolap\\LearnPython\\LearnMachineLearning\\Lessons\\dataLib\\missingdatas.csv"
data = pd.read_csv(datapath)


#>>>data processing

height = data["height"]

class human:
    height = 180


peter = human()


#>>>work with missing datas

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") #take avarage for missing datas

age = data.iloc[:,1:4].values
imputer = imputer.fit(age[:,1:4]) #learn datas
age[:,1:4] = imputer.transform(age[:,1:4]) #change missing (nan) datas


#>>> work with categorized datas

country = data.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

country[:,0] = le.fit_transform(data.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()


#>>> combine datas and crate dataframes

dataresult = pd.DataFrame(data = country, index = range(22), columns = ["france","turkey","united states"])

dataresult2 = pd.DataFrame(data = age, index = range(22), columns = ["Boy","Kilo","Yaş"])

gender = data.iloc[:,-1].values

dataresult3 = pd.DataFrame(data = gender, index = range(22), columns = ["Cinsiyet"])


#concat -> üleştirmek

lastresult = pd.concat([dataresult, dataresult2], axis = 1)

lastresult2 = pd.concat([lastresult, dataresult3], axis = 1)


#>>>split datas as train and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(lastresult, dataresult3, test_size = 0.33, random_state = 0) 


#>>>Data Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

