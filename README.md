# Add-Substractor
# data_generator.py
I generate 80000 dataset with half of add and half of substract. And 10% of 
the dataset used to be the vaildation set. 
# train.py
Use [this tutorial](https://github.com/IKMLab/Adder-practice) to build model and learn
# analysis
- the accuracy of dataset size
With 100 epoch and 128 batch-size

| dataset size(10% for validation) | 80000 | 70000 | 60000 |
|:--------------------------------:|:-----:|:-----:|:-----:|
| validation accuracy              | 0.9932 | 0.9909 | 1.9846 |

So, I think the size 70000 is enough

- time of batch size
With 70000 dataset size

| batch size | 64 | 128 | 256 | 512 |
|:----------:|:--:|:---:|:---:|:---:|
| Achieve 0.99 acc at epoch | 100 | 100 | can't achieve | can't achieve(run 200 epoch) |
| Time of achieve 0.99 acc(if higher) | 2594 | 1713 | - | - |

With 80000 dataset size

| batch size | 128 | 256 | 512 |
|:----------:|:---:|:---:|:---:|
| Achieve 0.99 acc at epoch | 78 | 98 | can't achieve(run 150 epoch) |
| Time of achieve 0.99 acc(if higher) | 1405 | 1360 | - |

So, I choose data size with 80000, batch size with 256 and epoch with 100

- multi-digits accuracy

| Digits | 4 |
|:------:|:-:|
| accuracy | 0.9250 |

# Use saved model
```python
python3 main.py
```
