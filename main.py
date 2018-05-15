# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:02:59 2018

@author: user
"""
import numpy as np
from keras.models import load_model
import util

utils = util.Util()
ctable = util.CharacterTable(utils.chars)
def vectorization(questions, expected, len1, len2):
    x = np.zeros((len(questions), len1, len(utils.chars)), dtype=np.bool)
    y = np.zeros((len(expected), len2, len(utils.chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, len1)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, len2)
    return x, y

test_x = 'dataset/val_x.txt'
test_y = 'dataset/val_y.txt'
with open(test_x, 'r', encoding='utf-8') as f:
    val_x = np.array(f.read().splitlines())
with open(test_y, 'r', encoding='utf-8') as f:
    val_y = np.array(f.read().splitlines())
    
x_val, y_val = vectorization(val_x, val_y, utils.MAXLEN, utils.DIGITS + 1)
model = load_model("my_model.h5")
print(model.evaluate(x_val, y_val, verbose=0))