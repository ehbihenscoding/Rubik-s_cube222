# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:51:40 2019

@author: Matthieu
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, BaggingClassifier
import re
import operator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error

traininp = pd.read_csv("C:/Users/Matthieu/Desktop/MVA/Projet Malla/data/train_input_MaPtRPJ.csv",)
test = pd.read_csv("C:/Users/Matthieu/Desktop/MVA/Projet Malla/data/test_input_5c0imze.csv",)

target = pd.read_csv("C:/Users/Matthieu/Desktop/MVA/Projet Malla/data/train_output_xd4VV9Q.csv",)

train = pd.concat([traininp, target], axis=1)
train = train.drop('ID',1)[0:421448]
target = train["distance"].values

f=[]
b=[]
l=[]
r=[]
u=[]
d=[]

for i in range(len(train['pos0'])):
    #f+=[[train['pos18'][i],train['pos17'][i],train['pos19'][i],train['pos16'][i]]]
    #b+=[[train['pos21'][i],train['pos22'][i],train['pos20'][i],train['pos23'][i]]]
    #l+=[[train['pos1'][i],train['pos5'][i],train['pos0'][i],train['pos4'][i]]]
    #r+=[[train['pos2'][i],train['pos6'][i],train['pos3'][i],train['pos7'][i]]]
    #u+=[[train['pos14'][i],train['pos13'][i],train['pos10'][i],train['pos9'][i]]]
    #d+=[[train['pos11'][i],train['pos8'][i],train['pos15'][i],train['pos12'][i]]]
    f.append(len(np.unique([train['pos18'][i],train['pos17'][i],train['pos19'][i],train['pos16'][i]])))
    b.append(len(np.unique([train['pos21'][i],train['pos22'][i],train['pos20'][i],train['pos23'][i]])))
    l.append(len(np.unique([train['pos1'][i],train['pos5'][i],train['pos0'][i],train['pos4'][i]])))
    r.append(len(np.unique([train['pos2'][i],train['pos6'][i],train['pos3'][i],train['pos7'][i]])))
    u.append(len(np.unique([train['pos14'][i],train['pos13'][i],train['pos10'][i],train['pos9'][i]])))
    d.append(len(np.unique([train['pos11'][i],train['pos8'][i],train['pos15'][i],train['pos12'][i]])))
train['front face']=f
train['back face']=b
train['left face']=l
train['right face']=r
train['up face']=u
train['down face']=d
    

#%%
train = train.drop('distance',1)[0:421448]
features = ['front face','back face','left face','right face','up face','down face']

df_selected=train[features]
df_features = df_selected.to_dict(orient='records')

vec = DictVectorizer()

features2 = vec.fit_transform(df_features).toarray()

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(
    features2, target, 
    test_size=0.20, random_state=42)



clf = RandomForestClassifier()

clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
mae=mean_absolute_error(labels_test,predictions)
print ("Mean Absolute Error:", mae)
#%%
f=[]
b=[]
l=[]
r=[]
u=[]
d=[]
for i in range(len(test['pos0'])):
    f.append(len(np.unique([test['pos18'][i],test['pos17'][i],test['pos19'][i],test['pos16'][i]])))
    b.append(len(np.unique([test['pos21'][i],test['pos22'][i],test['pos20'][i],test['pos23'][i]])))
    l.append(len(np.unique([test['pos1'][i],test['pos5'][i],test['pos0'][i],test['pos4'][i]])))
    r.append(len(np.unique([test['pos2'][i],test['pos6'][i],test['pos3'][i],test['pos7'][i]])))
    u.append(len(np.unique([test['pos14'][i],test['pos13'][i],test['pos10'][i],test['pos9'][i]])))
    d.append(len(np.unique([test['pos11'][i],test['pos8'][i],test['pos15'][i],test['pos12'][i]])))
test['front face']=f
test['back face']=b
test['left face']=l
test['right face']=r
test['up face']=u
test['down face']=d

#%%
test = test.drop('ID',1)
df_test=test[features]
df_features_test = df_test.to_dict(orient='records')
features_sub= vec.fit_transform(df_features_test).toarray()
predic_sub=clf.predict(features_sub)