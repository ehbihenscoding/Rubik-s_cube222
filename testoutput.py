#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:38:47 2019

@author: ehbihenscoding
"""

import pandas as pd

test1 = pd.read_csv("submit/test_output_s01.csv",)
test2 = pd.read_csv("submit/test_output_ps02.csv",)

print(test1["distance"]-test2["distance"])

