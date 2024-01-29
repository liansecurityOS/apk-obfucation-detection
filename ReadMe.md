## use the model to predict
```python predicat_apk.py <apkPath>```
## use the model to retrain
```python lstm.py```

It will read data from the new_train.csv.

About the model and training process you can reference from https://www.liansecurity.com/#/main/news/FzJZjIkBUQjGUXE2xvvb/detail

If you want to retrain with your own data, pay attention to the training set you use. As the doc above showing, choosing a training set can be tricky and can affect accuracy.
The training set in the doc above was randomly selected from our sample, and the accuracy was quite surprising to us.I hope it was helpful
