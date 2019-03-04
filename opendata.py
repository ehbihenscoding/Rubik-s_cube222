#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:38:47 2019

@author: ehbihenscoding
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import re
import operator
from sklearn.feature_selection import SelectKBest, f_classif

traininp = pd.read_csv("data/train_input_MaPtRPJ.csv",)
test = pd.read_csv("data/test_input_5c0imze.csv",)

target = pd.read_csv("data/train_output_xd4VV9Q.csv",)

train = pd.concat([traininp, target], axis=1)
train = train.drop('ID',1)[0:421448]
target = train["distance"].values

# FEATURES -----------------------------------------------------------------------------------
features = ['pos1','pos2','pos3','pos4','pos5','pos6','pos7','pos8','pos9','pos10','pos11','pos12','pos13','pos14','pos15','pos16','pos17','pos18','pos19','pos20','pos21','pos23']

selector = SelectKBest(f_classif, k=len(features))
selector.fit(train[features], target)
scores = -np.log10(selector.pvalues_)
indices = np.argsort(scores)[::-1]
print("Features importance :")
for f in range(len(scores)):
    print("%0.2f %s" % (scores[indices[f]],features[indices[f]]))

#rfc = RandomForestClassifier(n_estimators=125, min_samples_split=3)#, class_weight={1:6,1:14})
rfc = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)

# CROSS VALIDATION WITH RANDOM FOREST CLASSIFIER METHOD-----------------------------------------
kf = cross_validation.KFold(train.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(rfc, train[features], target, cv=kf)

print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean()*100, scores.std()*100, 'RFC Cross Validation'))
rfc.fit(train[features], target)
score = rfc.score(train[features], target)
print("Accuracy: %0.3f            [%s]" % (score*100, 'RFC full test'))
#importances = rfc.feature_importances_
#indices = np.argsort(importances)[::-1]
#for f in range(len(features)):
#    print("%d. feature %d (%f) %s" % (f + 1, indices[f]+1, importances[indices[f]]*100, features[indices[f]]))

print("début prédiction")

# PREDICTION  -----------------------------------------------------------------------------------
rfc.fit(train[features], target)

predictions = rfc.predict(test[features][0:918540/2])

# OUTPUT FILE -----------------------------------------------------------------------------------
PassengerId =np.array(test["ID"][0:918540/2]).astype(int)
my_prediction = pd.DataFrame(predictions, PassengerId, columns = ["distance"])

my_prediction.to_csv("predict/my_prediction1_4.csv", index_label = ["ID"])
print("fin étape 1")

predictions = rfc.predict(test[features][918540/2:918540])
# OUTPUT FILE -----------------------------------------------------------------------------------
PassengerId =np.array(test["ID"][918540/2:918540]).astype(int)
my_prediction = pd.DataFrame(predictions, PassengerId, columns = ["distance"])

my_prediction.to_csv("predict/my_prediction2_4.csv", index_label = ["ID"])
print("fin étape 2")

predictions = rfc.predict(test[features][918540:918540+918540/2])
# OUTPUT FILE -----------------------------------------------------------------------------------
PassengerId =np.array(test["ID"][918540:918540+918540/2]).astype(int)
my_prediction = pd.DataFrame(predictions, PassengerId, columns = ["distance"])

my_prediction.to_csv("predict/my_prediction3_4.csv", index_label = ["ID"])
print("fin étape 3")

predictions = rfc.predict(test[features][918540+918540/2:])
# OUTPUT FILE -----------------------------------------------------------------------------------
PassengerId =np.array(test["ID"][918540+918540/2:]).astype(int)
my_prediction = pd.DataFrame(predictions, PassengerId, columns = ["distance"])

my_prediction.to_csv("predict/my_prediction4_4.csv", index_label = ["ID"])
print("fin étape 4")

print("The end ...")
