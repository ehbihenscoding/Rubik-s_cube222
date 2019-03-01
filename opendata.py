#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:38:47 2019

@author: ehbihenscoding
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn import cross_validation
import re
import operator
from sklearn.feature_selection import SelectKBest, f_classif

traininp = pd.read_csv("data/train_input_MaPtRPJ.csv",)
test = pd.read_csv("data/test_input_5c0imze.csv",)

target = pd.read_csv("data/train_output_xd4VV9Q.csv",)

train = pd.concat([traininp, target], axis=1)
train = train.drop('ID',1)
target = train["distance"].values

# FEATURES -----------------------------------------------------------------------------------
features = ['pos1','pos2','pos3','pos4','pos5','pos6','pos7','pos8','pos9','pos10','pos11','pos12','pos13','pos14','pos15','pos16','pos17','pos18','pos19','pos20','pos21','pos23']

rfc = RandomForestClassifier(n_estimators=3000, min_samples_split=4, class_weight={1:6,1:14})


# CROSS VALIDATION WITH RANDOM FOREST CLASSIFIER METHOD-----------------------------------------
kf = cross_validation.KFold(train.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(rfc, train[features], target, cv=kf)
print("etape 2")
print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean()*100, scores.std()*100, 'RFC Cross Validation'))
rfc.fit(train[features], target)
score = rfc.score(train[features], target)
print("Accuracy: %0.3f            [%s]" % (score*100, 'RFC full test'))
importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(features)):
    print("%d. feature %d (%f) %s" % (f + 1, indices[f]+1, importances[indices[f]]*100, features[indices[f]]))


# PREDICTION  -----------------------------------------------------------------------------------
rfc.fit(train[features], target)
predictions = rfc.predict(test[features])

# OUTPUT FILE -----------------------------------------------------------------------------------
PassengerId =np.array(test["ID"]).astype(int)
my_prediction = pd.DataFrame(predictions, PassengerId, columns = ["distance"])

my_prediction.to_csv("data/my_prediction.csv", index_label = ["ID"])

print("The end ...")
