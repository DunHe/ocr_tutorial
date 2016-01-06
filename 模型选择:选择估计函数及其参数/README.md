# 模型选择:选择估计函数及其参数

### 准确度, 和交叉验证的准确度

如我们所见, 每一个估计函数都有一个**score**方法用于判断训练(或者预测)未知数据的质量. 该数值越大越好.

```python
>>> from sklearn import datasets, svm
>>> digits = datasets.load_digits()
>>> X_digits = digits.data
>>> y_digits = digits.target
>>> svc = svm.SVC(C=1, kernel='linear')
>>> svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])
0.97999999999999998
```

为了得到更好的预测精度的方法, 我们可以将用于训练的数据进行分割并存储在folds变量中:

```python
>>> import numpy as np
>>> X_folds = np.array_split(X_digits, 3)
>>> y_folds = np.array_split(y_digits, 3)
>>> scores = list()
>>> for k in range(3):
...     # We use 'list' to copy, in order to 'pop' later on
...     X_train = list(X_folds)
...     X_test  = X_train.pop(k)
...     X_train = np.concatenate(X_train)
...     y_train = list(y_folds)
...     y_test  = y_train.pop(k)
...     y_train = np.concatenate(y_train)
...     scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
>>> print(scores)
[0.93489148580968284, 0.95659432387312182, 0.93989983305509184]
```

其实上述的处理方式称为**KFold**交叉验证.

### 交叉验证生成器

上述分割训练数据的方式过于繁杂, Scikit-learn提供了用于生成训练数据和测试数据的交叉验证生成器:

```python
>>> from sklearn import cross_validation
>>> k_fold = cross_validation.KFold(n=6, n_folds=3)
>>> for train_indices, test_indices in k_fold:
...      print('Train: %s | test: %s' % (train_indices, test_indices))
Train: [2 3 4 5] | test: [0 1]
Train: [0 1 4 5] | test: [2 3]
Train: [0 1 2 3] | test: [4 5]
```

使用scikit-learn进行交叉验证:

```python
>>> kfold = cross_validation.KFold(len(X_digits), n_folds=3)
>>> [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
...          for train, test in kfold]
[0.93489148580968284, 0.95659432387312182, 0.93989983305509184]
```

为了计算估计函数的**score**, sklearn提供了一个方便有用的函数:

```python
>>> cross_validation.cross_val_score(svc, X_digits, y_digits, cv=kfold, n_jobs=-1)
array([ 0.93489149,  0.95659432,  0.93989983])
```

n_jobs=-1即使用电脑上所有的CPU进行计算.

**常用的交叉验证生成器**

- KFold(n, k): 将数据分割为K组, 训练其中的K-1组并用剩下的一组用于测试.
- StratifiedKFold(y, k): 在每组中保存类的比例以及标签的权重.
- LeaveOneOut(n): 类似KFold，是一个迭代算法, 每次留一个用于测试, 其余用于训练.

**练习**

在sklearn的digits数据库中, 我们可以为svm.SVC设置线性核, 然后尝试不同的参数C在SVC估计函数下的交叉验证的精确度:

```python
import numpy as np
from sklearn import cross_validation, datasets, svm

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)
```

一个完整的例子: [Cross-validation on Digits Dataset Exercise](http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_digits.html#example-exercises-plot-cv-digits-py)

### 网格搜索和交叉验证生成器

**网格搜索**

sklearn提供了对象通过给定的数据, 在参数grid上拟合估计函数时计算score, 并选择多个模型中能使交叉验证score最大的参数. 这个对象在构造时需要提供一个估计函数, 并提供了估计函数的API:

```python
>>> from sklearn.grid_search import GridSearchCV
>>> Cs = np.logspace(-6, -1, 10)
>>> clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),
...                    n_jobs=-1)
>>> clf.fit(X_digits[:1000], y_digits[:1000])        
GridSearchCV(cv=None,...
>>> clf.best_score_                                  
0.925...
>>> clf.best_estimator_.C                            
0.0077...

>>> # Prediction performance on test set is not as good as on train set
>>> clf.score(X_digits[1000:], y_digits[1000:])      
0.943...
```

默认情况下GridSearchCV使用3-fold的交叉验证. 然而, 当检测到传入的是一个分类器, 而不是回归器, 它就会使用stratified 3-fold.

**嵌套交叉验证**

```python
>>> cross_validation.cross_val_score(clf, X_digits, y_digits)
...                                                  
array([ 0.938...,  0.963...,  0.944...])
```

两个交叉验证的循环将会并行运行: 一个是GridSearchCV估计函数用于设置gamma的值, 另一个是cross_val_score用于测量估计函数的预测结果. 结果在未知数据的score预测中是无偏估计的.

**提示:** 不能在嵌套对象中使用并行运算(n_jobs不等于1).

**具备交叉验证功能的估计函数**

交叉验证在算法与算法之间能更加有效地设置参数. 这就是为什么sklearn提供了这些能够通过交叉验证自动设置参数的[估计函数](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation).

```python
>>> from sklearn import linear_model, datasets
>>> lasso = linear_model.LassoCV()
>>> diabetes = datasets.load_diabetes()
>>> X_diabetes = diabetes.data
>>> y_diabetes = diabetes.target
>>> lasso.fit(X_diabetes, y_diabetes)
LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
    max_iter=1000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
    precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
    verbose=False)
>>> # The estimator chose automatically its lambda:
>>> lasso.alpha_ 
0.01229...
```

这些函数与他们本身相似, 只是在原先的名字后面加上'CV'.

**练习**:
一个完整使用交叉验证训练diabetes数据集的[例子](http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#example-exercises-plot-cv-diabetes-py).
