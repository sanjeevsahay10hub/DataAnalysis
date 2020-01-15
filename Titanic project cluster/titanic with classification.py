# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:30:20 2020

@author: sanjeev
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

sample=pd.read_csv('D:/data-analysis/Titanic project cluster/train.csv')

sample.fillna(sample.mean(), inplace= True)

sample.isnull()
sample.isnull().sum()
sample.drop(['Name','Ticket','Cabin','Embarked'], axis=1, inplace= True)
sample.info()

le=LabelEncoder()
le.fit(sample['Sex'])
sample['Sex']=le.transform(sample['Sex'])
A=np.array(sample.drop(['Survived'],1)).astype(float)
B=np.array(sample['Survived'])
#new_df=pd.get_dummies(A)
from sklearn.model_selection import train_test_split
A_train,A_test,B_train,B_test=train_test_split(A,B, test_size=0.2, random_state=7)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
model.fit(A_train,B_train)
model.predict(A_test)
B_test
model.score(A_test,B_test)

from sklearn.model_selection import cross_val_score
cv_score_mnb=cross_val_score(model, A, B, cv=5)
cv_score_mean=np.mean(cv_score_mnb)
from sklearn.naive_bayes import MultinomialNB

mn=MultinomialNB()
mn.fit(A_train,B_train)
mn.predict(A_test)
B_test

mn.score(A_test,B_test)

from sklearn.tree import DecisionTreeClassifier

ts=DecisionTreeClassifier()
ts.fit(A_train,B_train)
ts.predict(A_test)
B_test
ts.score(A_test,B_test)


















