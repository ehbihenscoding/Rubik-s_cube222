#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:38:47 2019

@author: ehbihenscoding
"""

import pandas as pd
#import numpy as np

traininp = pd.read_csv("data/train_input_MaPtRPJ.csv",)
test = pd.read_csv("data/test_input_5c0imze.csv",)

target = pd.read_csv("data/train_output_xd4VV9Q.csv",)

train = pd.concat([traininp, target], axis=1)
train = train.drop('ID',1)
print(train.values)
