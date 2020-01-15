# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:23:32 2020

@author: sanjeev
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

train=pd.read_excel('D:/data-analysis/Titanic project cluster/titanic.xls')
test=pd.read_excel('D:/data-analysis/Titanic project cluster/titanic.xls')
train.columns.values
train.isnull().sum()
test.isna().sum()
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)
train.isna().sum()

train[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().sort_values(by='survived', ascending=False)
train[['sex','survived']].groupby(['sex'], as_index=False).mean().sort_values(by='survived' , ascending=False)       
train[['sibsp','survived']].groupby(['sibsp'], as_index=False).mean().sort_values(by='survived', ascending=False)        

g=sns.FacetGrid(train,col='survived')
g.map(plt.hist, 'age', bins=20)        
                
grid=sns.FacetGrid(train, col='survived', row='pclass', size=2.2 , aspect=1.6)
grid.map(plt.hist, 'age', bins=20)
grid.add_legend()        
        
train.info()        
train.drop(['name','ticket','cabin','embarked','home.dest'], axis=1, inplace= True)
test.drop(['name','ticket','cabin','embarked','home.dest'], axis=1, inplace= True)
train.drop(['boat'], axis=1, inplace=True)
test.drop(['boat'], axis=1, inplace=True)

train.info()

labelencoder=LabelEncoder()  
labelencoder.fit(train['sex'])
labelencoder.fit(test['sex'])  
train['sex']=labelencoder.transform(train['sex'])
test['sex']=labelencoder.transform(test['sex'])

train.info()   

test.drop(['survived'], axis=1 , inplace=True) 
test.info()      

X=np.array(train.drop(['survived'], 1). astype(float))  
Y=np.array(train['survived'])      

train.info()  
        
kmeans=KMeans(n_clusters=2) # you want cluster passengers in two grop  survived or not survived
kmeans.fit(X)
len(X)
correct=0

for i in range (len(X)):
    predict_me= np.array(X[i].astype(float))
    predict_me=predict_me.reshape(-1 , len(predict_me))
    prediction=kmeans.predict(predict_me)
    if prediction[0]==Y[i]:
        correct +=1
        
print(correct/len(X))


#step 2
scaler=MinMaxScaler()
scaled_X=scaler.fit_tranform(X)
kmeans.fit(scaled_X)  
correct=0
for i in range (len(X)):
    predict_me= np.array(X[i].astype(float))
    predict_me=predict_me.reshape(-1 , len(predict_me))
    prediction=kmeans.predict(predict_me)
    if prediction[0]==Y[i]:
        correct +=1
        
print(correct/len(X))    


        