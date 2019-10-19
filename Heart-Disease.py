# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:04:22 2019

@author: sanjeev
"""

import numpy as np
import pandas as pd

with open("D:/data-analysis/Heart-Disease/data/Heart disease/hungarian.data") as myfile:
    raw_data = myfile.readlines()

merge_data = [''.join(raw_data)]
split_by_name = [i.split("name") for i in merge_data][0]


# Loss of some values is occuring in below operations. Most probably because of newline.
split_by_name = [text.splitlines() for text in split_by_name]
split_by_name = [" ".join(text) for text in split_by_name]# changing data behaviour while merging line last
#previous line merging from new line 
split_by_name = [line.replace('-', '') for line in split_by_name]
#split_by_name = [line.replace('.', '1') for line in split_by_name]


#convert_to_float = []
#for i in range(len(split_by_name)):
  #  convert_to_float.append(list(map(int, split_by_name[i].split())))
    
output=[]    
for column in split_by_name:
    output.append(column.split(' '))
    
print(output)

df=pd.DataFrame(output)

df.columns=["col1","ID","CCF","Age","Sex","Painloc","Painexer","Relrest","Pncaden","CP","Trestbps","HTN","Chol",
      "Smoke","Cigs","Years","Fbs","DM","Famhist","Restecg","Ekgmo","Ekgday","Ekgyr","Dig","Prop","Nitr",
       "Pro","Diuretic","Proto","Thaldur","Thaltime","Met","Thalach","Thalrest","Tpeakbps","Tpeakbpd",
        "dummy","Trestbpd","Exang","Xhypo","Oldpeak","Slope","Rldv5","Rldv5e","CA","Restckm","Exerckm",
         "Restef","Restwm","Exeref","Exerwm","Thal","Thalsev","Thalpul","Earlobe","CMO","Cday","Cyr","Num",
          "Lmt","Ladprox","Laddist","Diag","Cxmain","Ramus","OM1","OM2","Rcaprox","Rcadist","Lvx1","Lvx2",
           "Lvx3","Lvx4","Lvf","Cathef","Junk","col76"]

df1=df.drop([294])
new_df=df1.drop(['col1', 'col76',"ID","CCF","Dig","Thalsev","Thalpul","Earlobe","Lvx1","Lvx2","Lvx3","Lvx4","Cathef","Junk","dummy","Lvf"], axis=1)


dict={'2':'1','3':'1','4':'1' }
new_df=new_df.replace({"Num":dict})
df.groupby('Num').size()
Y=new_df['Num']
X = new_df.drop(columns=['Num'])
new_df1 = pd.get_dummies(X)
X=new_df1

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4, random_state=9)

#USING NAIVE BAYES
#from sklearn.naive_bayes import MultinomialNB 

#mnb=MultinomialNB()
#mnb.fit(X_train,Y_train)
#mnb.predict(X_test)
##mnb.score(X_test,Y_test)


#from sklearn.model_selection import cross_val_score
#cv_score_mnb = cross_val_score(mnb, X, Y, cv=5)
#mean_test_score_mnb = np.mean(cv_score_mnb)

#Y_predicted=mnb.predict(X_test)

#from sklearn.metrics import confusion_matrix

#cm=confusion_matrix(Y_test,Y_predicted)
#cm

#USING RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
model.fit(X_train,Y_train)
model.predict(X_test)
Y_test
model.score(X_test,Y_test)

from sklearn.model_selection import cross_val_score
cv_score_rf = cross_val_score(model, X, Y, cv=5)
mean_test_score_rf = np.mean(cv_score_rf)

Y_predicted=model.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,Y_predicted)
cm

from sklearn.metrics  import roc_curve, auc
false_positive_rate,true_positive_rate
thresolds=roc_curve(Y_test,Y_predicted)
roc_auc=auc(false_positive_rate,true_positive_rate)
roc_auc
#USING DECISION TREE

#from sklearn.tree import DecisionTreeClassifier

#dtc=DecisionTreeClassifier()
#dtc.fit(X_train,Y_train)
#dtc.predict(X_test)
#Y_test
#dtc.score(X_test,Y_test)

#from sklearn.model_selection import cross_val_score

#cv_score_dtc=cross_val_score(dtc,X,Y,cv=5)
#mean_test_score_dtc=np.mean(cv_score_dtc)


#Y_predicted=dtc.predict(X_test)

#from sklearn.metrics import confusion_matrix

#cm=confusion_matrix(Y_test,Y_predicted)
#cm

 












