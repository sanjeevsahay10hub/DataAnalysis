# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:41:06 2019

@author: sanjeev
"""

import pandas as pd
import numpy as np

df=pd.read_csv("C:/Projects/p1/breast-cancer.csv")
df.isnull()
df.isnull().sum()
#df_unique =pd.unique(['age','Menopause','Tumor-Size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat'])
#df_unique
col_names= df.columns.values
col_names
#df[col_names].unique()
#pd.concat([df['age']]).unique()

uniques_dict = {}
for i in col_names:
    uniques_dict[str(i)] = pd.concat([df[str(i)]]).unique()
unique_dict = {}
#unique_values = {i:pd.concat([df[str(i)]]).unique() for i in col_names}
#unique_values
#print(uniques_dist)

#df_unique.Tumor-Size.sortby('14-Oct','2-May')
df.groupby('Tumor-Size').size()
df.groupby('inv-nodes').size()
df.groupby('node-caps').size()
df.groupby('breast-quad').size()
df = df.drop(df[df['Tumor-Size']=='14-Oct'].index)
df = df.drop(df[df['Tumor-Size']=='9-May'].index)
df = df.drop(df[df['inv-nodes']=='8-Jun'].index)
df = df.drop(df[df['inv-nodes']=='11-Sep'].index)
df = df.drop(df[df['inv-nodes']=='5-Mar'].index)
df = df.drop(df[df['inv-nodes']=='14-Dec'].index)
df = df.drop(df[df['node-caps']=='?'].index)
df = df.drop(df[df['breast-quad']=='?'].index)
df.groupby('Tumor-Size').size()
df.groupby('breast-quad').size()
df.groupby('inv-nodes').size()
df['Class']=df.Class.map({'no-recurrence-events':0, 'recurrence-events':1})
Y = df['Class']
df = pd.get_dummies(df)
X = df.drop(columns=['Class'])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=7)

from sklearn.naive_bayes import GaussianNB 

gnb=GaussianNB()
gnb.fit(X_train,Y_train)
gnb.predict(X_test)
Y_test
gnb.score(X_test,Y_test)

from sklearn.naive_bayes import MultinomialNB 

mnb=MultinomialNB()
mnb.fit(X_train,Y_train)
mnb.predict(X_test)
Y_test
mnb.score(X_test,Y_test)

#from sklearn.model_selection import cross_val_score
#cv_score = cross_val_score(mnb, X, Y, cv=5)
#mean_test_score = np.mean(cv_score)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,Y_train)
#model.predict(X_test)
#Y_test
model.score(X_test,Y_test)

from sklearn.model_selection import cross_val_score
cv_score_rf = cross_val_score(model, X, Y, cv=5)
cv_score_mnb = cross_val_score(mnb, X, Y, cv=5)
mean_test_score = np.mean(cv_score_rf)


Y_predicted=model.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_predicted)
cm

#matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figure=(100,5))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")





