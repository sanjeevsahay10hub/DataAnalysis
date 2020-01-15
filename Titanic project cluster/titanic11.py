# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:20:17 2020

@author: sanjeev
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing

df=pd.read_excel('D:/data-analysis/Titanic project cluster/titanic.xls')
df.drop(['body','name'], axis=1, inplace= True)

df.fillna(0,inplace= True)
#This can be done by simply take a set of the column values. From here, 
#the index within that set can be the new "numerical" value or "id" of the text data.
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df2= handle_non_numerical_data(df)
df2.head()
df2.drop(['ticket'], axis=1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X=preprocessing.scale(X)
y = np.array(df['survived'])

clf=KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))










