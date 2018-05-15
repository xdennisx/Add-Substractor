# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:37:41 2018

@author: user
"""
import numpy as np
import util

util = util.Util()

questions = []
expected = []
seen = set()
isadd = True
print('Generating data...')
while len(questions) < util.TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, util.DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    if isadd:
        q = '{}+{}'.format(a, b)
        ans = str(a + b)
        isadd = False
    else:
        smaller, larger = key
        q = '{}-{}'.format(larger, smaller)
        ans = str(larger - smaller)
        isadd = True
    query = q + ' ' * (util.MAXLEN - len(q))
    ans += ' ' * (util.DIGITS + 1 - len(ans))
    if util.REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))
print(questions[:5], expected[:5])

x = np.array(questions)
y = np.array(expected)

#indices = np.arange(len(y))
#np.random.shuffle(indices)
#x = x[indices]
#y = y[indices]

# train_test_split
train_x = x
train_y = y

split_at = len(train_x) - len(train_x) // 10
(x_train, x_val) = train_x[:split_at], train_x[split_at:]
(y_train, y_val) = train_y[:split_at], train_y[split_at:]

with open('dataset/train_x.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join("".join(x) for x in x_train))
with open('dataset/train_y.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join("".join(x) for x in y_train))
with open('dataset/val_x.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join("".join(x) for x in x_val))
with open('dataset/val_y.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join("".join(x) for x in y_val))