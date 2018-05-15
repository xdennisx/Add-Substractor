# -*- coding: utf-8 -*-
"""
Created on Mon May 14 23:14:49 2018

@author: user
"""

import numpy as np
from keras.models import Sequential
from keras import layers
import util

utils = util.Util()
colors = util.colors
ctable = util.CharacterTable(utils.chars)

def vectorization(questions, expected, len1, len2):
    x = np.zeros((len(questions), len1, len(utils.chars)), dtype=np.bool)
    y = np.zeros((len(expected), len2, len(utils.chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, len1)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, len2)
    return x, y
        
with open('dataset/train_x.txt', 'r', encoding='utf-8') as f:
    train_x = np.array(f.read().splitlines())
with open('dataset/train_y.txt', 'r', encoding='utf-8') as f:
    train_y = np.array(f.read().splitlines())
with open('dataset/val_x.txt', 'r', encoding='utf-8') as f:
    val_x = np.array(f.read().splitlines())
with open('dataset/val_y.txt', 'r', encoding='utf-8') as f:
    val_y = np.array(f.read().splitlines())
    
x_train, y_train = vectorization(train_x, train_y, utils.MAXLEN, utils.DIGITS + 1)
x_val, y_val = vectorization(val_x, val_y, utils.MAXLEN, utils.DIGITS + 1)

print('Build model...')
model = Sequential()
model.add(utils.RNN(utils.HIDDEN_SIZE, input_shape=(utils.MAXLEN, len(utils.chars))))
model.add(layers.RepeatVector(utils.DIGITS + 1))
for _ in range(utils.LAYERS):
    model.add(utils.RNN(utils.HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(len(utils.chars))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())


print()
print('-' * 50)
model.fit(x_train, y_train,
          batch_size=utils.BATCH_SIZE,
          epochs=100,
          validation_data=(x_val, y_val))
for i in range(10):
    ind = np.random.randint(0, len(x_val))
    rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
    preds = model.predict_classes(rowx, verbose=0)
    q = ctable.decode(rowx[0])
    correct = ctable.decode(rowy[0])
    guess = ctable.decode(preds[0], calc_argmax=False)
    print('Q', q[::-1] if utils.REVERSE else q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print(colors.ok + '☑' + colors.close, end=' ')
    else:
        print(colors.fail + '☒' + colors.close, end=' ')
    print(guess)