# 使用GridSearchCV对比各算法在手写数字识别的精确度

本篇文章主要介绍如何使用GridSearchCV在交叉验证的前提下进行算法识别的精确度比较， 如果对GridSearchCV不了解的同学请移步[这里](https://github.com/DunHe/ocr_tutorial/tree/master/%E6%A8%A1%E5%9E%8B%E9%80%89%E6%8B%A9:%E9%80%89%E6%8B%A9%E4%BC%B0%E8%AE%A1%E5%87%BD%E6%95%B0%E5%8F%8A%E5%85%B6%E5%8F%82%E6%95%B0)了解.

### 科普小知识

- Estimator
这里摘抄一段对Estimator的解释，通俗地将Estimator就是值识别器:
> An Estimator abstracts the concept of a learning algorithm or any algorithm that fits or trains on data. Technically, an Estimator implements a method fit(), which accepts a DataFrame and produces a Model, which is a Transformer. For example, a learning algorithm such as LogisticRegression is an Estimator, and calling fit() trains a LogisticRegressionModel, which is a Model and hence a Transformer.

- Generator
其实就是函数的意思...在Cross Validation中代表不同的split方案，就是把一个完整的数据库，拆分为训练集和测试集. 常见的算法有K-Fold, Loo-CV等.


### 小试牛刀

**准备训练及验证数据**

比较重要的函数是**train_test_split**, 主要是使用交叉验证的思想将数据按比例分割为训练集和验证集:

```python
digits = datasets.load_digits()
n_samples = len(digits.images)
x_digits = digits.images.reshape((n_samples, -1))
y_digits = digits.target
x_train, x_test, y_train, y_test = train_test_split(x_digits, y_digits, test_size = 0.3, random_state = 0)
```

**准备识别器参数**

本文使用svm, adaboost, randomforest三种算法进行对比， 参数设置如下:

```python
tuned_parameters = [
        {'svm': [
            {'kernel': ['rbf'],
             'C': np.logspace(-2, 10, 13),
             'gamma': np.logspace(-9, 3, 13)},

            {'kernel': ['linear', 'sigmoid'],
             'C': np.logspace(-2, 10, 13)}
        ]}, 

        {'adaboost': [
            {'n_estimators': range(10, 110, 10),
             'learning_rate': np.linspace(1, 4, 8),
             'algorithm': ['SAMME', 'SAMME.R']}
        ]}, 

        {'randomforest': [
            {'n_estimators': range(10, 110, 10),
             'criterion': ['gini', 'entropy'],
             'max_leaf_nodes': range(2, 11),
             'bootstrap': [True, False],
             'max_features': ['auto', 'log2']}
        ]}
]
```

**构造识别器**

根据上述参数列表中的算法进行识别器构造:

```python
for i in tuned_parameters:

    if i.keys()[0] is 'svm':
        clf = GridSearchCV(svm.SVC(), i.values()[0], n_jobs = -1)

    elif i.keys()[0] is 'adaboost':
        clf = GridSearchCV(AdaBoostClassifier(), i.values()[0], n_jobs=-1)

    elif i.keys()[0] is 'randomforest':
        clf = GridSearchCV(RandomForestClassifier(), i.values()[0], n_jobs = -1)
```

**训练识别器**

```python
clf.fit(x_train, y_train)
```

**对比识别器**

```python
print clf.best_params_
print clf.best_score_

print classification_report(y_test, clf.predict(x_test))
```

**svm最佳参数以及各项精确度**

```python
training svm classifier......
svm classifier best parameter and score:
{'kernel': 'rbf', 'C': 10.0, 'gamma': 0.001}
0.989657915672
svm classifier classification report......
             precision    recall  f1-score   support

          0       1.00      1.00      1.00        45
          1       0.98      1.00      0.99        52
          2       1.00      0.98      0.99        53
          3       1.00      1.00      1.00        54
          4       1.00      1.00      1.00        48
          5       0.98      0.96      0.97        57
          6       0.98      1.00      0.99        60
          7       0.98      1.00      0.99        53
          8       1.00      0.98      0.99        61
          9       0.98      0.98      0.98        57

avg / total       0.99      0.99      0.99       540
```

**AdaBoost最佳参数以及各项精确度**

```python
training AdaBoost classifier......
AdaBoost classifier best parameter and score:
{'n_estimators': 100, 'learning_rate': 1.0, 'algorithm': 'SAMME'}
0.82338902148
AdaBoost classifier classification report......
             precision    recall  f1-score   support

          0       0.90      1.00      0.95        45
          1       0.77      0.63      0.69        52
          2       1.00      0.62      0.77        53
          3       0.76      0.70      0.73        54
          4       0.89      0.81      0.85        48
          5       0.94      0.81      0.87        57
          6       1.00      0.87      0.93        60
          7       0.98      0.87      0.92        53
          8       0.63      0.95      0.76        61
          9       0.66      0.93      0.77        57

avg / total       0.85      0.82      0.82       540
```

**RandomForest最佳参数以及各项精确度**

```python
training RandomForest classifier......
RandomForest classifier best parameter and score:
{'max_features': 'auto', 'max_leaf_nodes': 10, 'bootstrap': True, 'criterion': 'entropy', 'n_estimators': 80}
0.907716785998
classifier classification report......
             precision    recall  f1-score   support

          0       0.96      1.00      0.98        45
          1       0.86      0.71      0.78        52
          2       0.88      0.87      0.88        53
          3       0.79      0.93      0.85        54
          4       0.88      0.94      0.91        48
          5       0.91      0.91      0.91        57
          6       0.92      1.00      0.96        60
          7       0.91      0.92      0.92        53
          8       0.96      0.84      0.89        61
          9       0.87      0.84      0.86        57

avg / total       0.90      0.89      0.89       540
```

通过观察:

score: svm > RandomForest > AdaBoost
综合测评: svm > RandomForest > AdaBoost

### 结论

在手写数字识别中, svm算法的效果较好.
