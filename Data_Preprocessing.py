import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('diabetes_null.csv')
df
print(df.isnull().sum().sort_values(ascending=False))
df=df.dropna() #drop row
df
y= df['Outcome']
X=df.drop(['Outcome'],1)
X
X=pd.get_dummies(X)
X
from sklearn.impute import SimpleImputer 
# Call the Simple Imputer Class and pass the Necessary Arguments
imputer = SimpleImputer(missing_values= np.nan, strategy='mean') #np.nan means all the Empty cells in the array
# Fit The imputer on the Desired Columns then Transform it 
X = imputer.fit_transform(X.values)
X
X=X.astype(int)
X
from sklearn.preprocessing import LabelEncoder
print(y)
encoder = LabelEncoder()
label_vector = encoder.fit_transform(y)
label_vector
# import the modules 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(len(X_train))
print(len(X_test))
print(len(X))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# take a look how the values was before and after the scalling 
print('before scalling, max is %d and min is %d'%(np.max(X_train), np.min(X_train)))
# Learn the paramters from X_train and  Transfrom  X_Train
X_train = scaler.fit_transform(X_train)
# Then transform X_test 
X_test = scaler.transform(X_test)
print('after scalling, max is %d and min is %d'%(np.max(X_train), np.min(X_train)))
