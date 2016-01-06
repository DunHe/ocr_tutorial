# 使用scikit-learn进行机器学习

**章节内容**

在这一节中, 我们会介绍在scikit-learn中广泛使用的[机器学习](https://en.wikipedia.org/wiki/Machine_learning)的专业名词, 然后给出一个简单的机器学习例子.

### 机器学习: 问题设定

通常来说, 一个机器学习的问题被视为具有n个样例的数据集合和尽可能去预测未知数据的可能性. 如果每个样例中有多于一个数据项或者说数据项是多维的, 这就说这个数据集具有多个属性或者**特征**.

我们能把机器学习的问题分为如下几种大类:

- [监督学习](https://en.wikipedia.org/wiki/Supervised_learning), 目的是预测数据中的附加属性. 这个问题可以分为:
    - [分类](https://en.wikipedia.org/wiki/Statistical_classification): 样本数据属于两个以上的类别, 我们要做的是从已经标记的数据中学习如何预测未标记数据所属的类别. 一个分类问题的例子是手写数字识别, 目的是分配输入向量到一个有限数量的离散类别中. 另一种方法去理解分类是监督学习输出结果的离散形式(与连续相反), 即每一个样本都对应输出结果中的一个固定分类.
    - [回归](https://en.wikipedia.org/wiki/Regression_analysis): 如果输出结果由一个或者多个连续变量组成, 则称该问题为回归. 一个回归问题的例子是三文鱼的长度作为其年龄和体重的函数的预测.

- [非监督学习](https://en.wikipedia.org/wiki/Unsupervised_learning), 其训练数据是由输入向量集合x组成, 没有任何相应的目标值. 非监督学习的目标是发现由输入数据中相似的数据所组成的群组, 称为聚类. 或者是确认输入数据在其输入空间中的贡献, 称为概率分布. 或者是将数据的项从一个较高的维度降为二维或者三维便于可视化.

**训练集和测试集**

机器学习是关于学习一个数据集中的某些属性并将其应用于新数据中的方法. 这就是为什么机器学习在评估算法中需要将手中的数据分为两个集合, 一个是训练集, 我们用于从中学习数据的属性. 另一个是测试集, 我们用于测试这些属性.

### scikit-learn开发环境的搭建
详情可参考[不同平台的scikit-learn开发环境搭建](http://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge), 这里简单介绍scikit-learn在Ubuntu14.04上的搭建:

基本环境搭建:
```bash
sudo apt-get install build-essential python-dev python-pip python-setuptools
```

安装numpy, scipy, scikit-learn:
```bash
pip install -U numpy scipy scikit-learn
```

大部分机器在安装过程中都会出现如下问题:
>numpy.distutils.system_info.NotFoundError: no lapack/blas resources found

这里的意思是缺少libatlas和liblapack部分库, 执行如下命令即可:
```bash
sudo apt-get install libatlas-dev liblapack-dev libatlas-base-dev libatlas3gf-base gfortran
```

### 读取一个样本数据集

scikit-learn拥有一些官方数据集, 例如[iris](https://en.wikipedia.org/wiki/Iris_flower_data_set)和[digits](http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)用于分类和[boston house prices dataset](http://archive.ics.uci.edu/ml/datasets/Housing).

接下来, 我们会在shell中使用python以及读取**iris**和**digits**数据集. 我们的标记约定是，$表示shell提示符，而>>>表示Python解释器的提示:

```python
$ python
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> digits = datasets.load_digits()
```

一个数据集是一个类似字典的对象, 包含了所有的数据和一些元数据. 这些数据存储在**.data**成员中, 是一个**n_samples**, **n_features**的数组. 在监督问题的情况下, 一个以上的响应值会存储在**.target**成员中. 更多不同数据集的细节请访问[dedicated section](http://scikit-learn.org/stable/datasets/index.html#datasets).

例如, 在digits数据集的例子中, **digits.data** 给出了数据的特征值, 用于数字样本的分类.
```python
>>> print(digits.data)
[[  0.   0.   5. ...,   0.   0.   0.]
 [  0.   0.   0. ...,  10.   0.   0.]
 [  0.   0.   0. ...,  16.   9.   0.]
 ...,
 [  0.   0.   1. ...,   6.   0.   0.]
 [  0.   0.   2. ...,  12.   0.   0.]
 [  0.   0.  10. ...,  12.   1.   0.]]
```

**digits.target** 给出了digits数据集的ground truth(监督学习中分类的准确性), 即相应到每一个我们想学习的digit图像的数量:

```python
>>> digits.target
array([0, 1, 2, ..., 8, 9, 8])
```

**数据数组的大小**

数据总是一个2D(二维)数组, 大小为(n\_samples, n\_features), 即使原始数据可能会有不同的大小. 在数据集digits的例子中, 每一个原始例子都是一个大小为(8, 8)的图片, 可通过下面的方式访问:

```python
>>> digits.images[0]
array([[  0.,   0.,   5.,  13.,   9.,   1.,   0.,   0.],
       [  0.,   0.,  13.,  15.,  10.,  15.,   5.,   0.],
       [  0.,   3.,  15.,   2.,   0.,  11.,   8.,   0.],
       [  0.,   4.,  12.,   0.,   0.,   8.,   8.,   0.],
       [  0.,   5.,   8.,   0.,   0.,   9.,   8.,   0.],
       [  0.,   4.,  11.,   0.,   1.,  12.,   7.,   0.],
       [  0.,   2.,  14.,   5.,  10.,  12.,   0.,   0.],
       [  0.,   0.,   6.,  13.,  10.,   0.,   0.,   0.]])
```

[一个简单的例子](http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#example-classification-plot-digits-classification-py)阐述了如何利用scikit-learn从原始问题入手修改数据的大小用于预测.

### 学习和预测

在数据集digits的例子中, 我们的任务是在给定的图片中预测图片对应的数字. 我们得到的样本中每个样本包含了10种可能的类别(从数字0到9), 这里我们拟合[概率分布](https://en.wikipedia.org/wiki/Estimator)来预测未见过的样本所属的类别.

在scikit-learn中, 分类中的概率分布是一个Python对象, 对应的实现方法为**fit(X, y)**和**predict(T)**.

一个概率分布的例子是通过[支持向量机](https://en.wikipedia.org/wiki/Support_vector_machine)实现的类**sklearn.svm.SVC**. 概率分布的构造函数的参数使用了模型的参数, 但同时我们可以认为概率分布就是一个黑盒子:

```python
>>> from sklearn import svm
>>> clf = svm.SVC(gamma=0.001, C=100.)
```

**选择模型的参数**

在这个例子中我们手动设置了gamma的值. 其实通过使用例如[grid search](http://scikit-learn.org/stable/modules/grid_search.html#grid-search)和[cross validation](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)来自动找到好的参数值也是可行的.

由于概率分布的实例是一个分类器, 所以我们称它为**clf**. 现在它必须从模型中拟合, 也就是说, 它必须从模型中学习. 这需要传递我们的训练集到**fit**方法中. 在训练集中, 让我们使用数据集中除了最后一个图片的所有图片. 我们通过使用Python语法 [:-1] 选择这个训练集, 这将产生包括除了**digits.data**中最后一项的新的数组:

```python
>>> clf.fit(digits.data[:-1], digits.target[:-1])
SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
```

现在你可以预测新的数值, 特别地, 我们可以通过询问分类器得到**digits**数据集中最后一副我们还没有在分类器的训练中使用的图的数字:

```python
>>> clf.predict(digits.data[:-1])
array([8])
```

相应图案如下所示:
<center>![](http://scikit-learn.org/stable/_images/plot_digits_last_image_0011.png)</center>

如你所见, 这是一个有挑战性的任务: 图片的分辨率非常低, 你是否同意分类器的结果?

这个关于分类问题的完整例子可作为一个能运行和学习的例子: [手写数字识别](http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#example-classification-plot-digits-classification-py)

### 模型的持久性

在scikit中使用Python内置的持久模型[pickle](https://docs.python.org/2/library/pickle.html)去存储一个模型是可行的:

```python
>>> from sklearn import svm
>>> from sklearn import datasets
>>> clf = svm.SVC()
>>> iris = datasets.load_iris()
>>> X, y = iris.data, iris.target
>>> clf.fit(X, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

>>> import pickle
>>> s = pickle.dumps(clf)
>>> clf2 = pickle.loads(s)
>>> clf2.predict(X[0:1])
array([0])
>>> y[0]
0
```

在scikit某些特殊的例子中, 使用pickle的替代品joblib可能会更有趣(**joblib.dump** & **joblib.load**), 这将在大数据中更高效, 但只能pickle到磁盘中而非字符串:

```python
>>> from sklearn.externals import joblib
>>> joblib.dump(clf, 'filename.pkl')
```

接着你可以读取到pickled的模型(可能在另一个Python进程中):

```python
>>> clf = joblib.load('filename.pkl')
```

**提示**: joblib.dump返回一个文件名字的列表. 每一个包含在clf对象中的独立的numpy数组都被序列化在文件系统的独立文件中. 当使用joblib.load从模型中重新读取时所有的文件需要一个相同的文件夹.

需要注意的是pickle存在一定的安全性和可维护性问题. 请到[模型的持久性](http://scikit-learn.org/stable/modules/model_persistence.html#model-persistence)章节理解关于更多scikit-learn中模型持久性的细节.

### 约束

scikit-learn的概率分布遵循一定的规则, 使得它们的行为更具可预测性.

#### 类型转换

除非其他特殊情况, 输入的类型必须转化为**float64**:

```python
>>> import numpy as np
>>> from sklearn import random_projection

>>> rng = np.random.RandomState(0)
>>> X = rng.rand(10, 2000)
>>> X = np.array(X, dtype='float32')
>>> X.dtype
dtype('float32')

>>> transformer = random_projection.GaussianRandomProjection()
>>> X_new = transformer.fit_transform(X)
>>> X_new.dtype
dtype('float64')
```

在这个例子中, **X**的类型为**float32**, 通过**fit_transform(X)**转化为**float64**.

回归结果被转化为**float64**, 分类结果保持不变:

```python
>>> from sklearn import datasets
>>> from sklearn.svm import SVC
>>> iris = datasets.load_iris()
>>> clf = SVC()
>>> clf.fit(iris.data, iris.target)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

>>> list(clf.predict(iris.data[:3]))
[0, 0, 0]

>>> clf.fit(iris.data, iris.target_names[iris.target])  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

>>> list(clf.predict(iris.data[:3]))  
['setosa', 'setosa', 'setosa']
```

这里, 第一个**predict()**返回一个整型数组, 是因为**iris.target**(整型数组)在**fit**方法中被使用. 第二个**predict**返回一个字符串数组, 是因为**iris.target_names**用于拟合.

#### 重拟合和更新参数

一个概率分布中的超参数被[sklearn.pipeline.Pipeline.set_params](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.set_params)这个方法构造后可以更新. 调用**fit()**一次以上将会重写所有之前在**fit()**中学习到的内容:

```python
>>> import numpy as np
>>> from sklearn.svm import SVC

>>> rng = np.random.RandomState(0)
>>> X = rng.rand(100, 10)
>>> y = rng.binomial(1, 0.5, 100)
>>> X_test = rng.rand(5, 10)

>>> clf = SVC()
>>> clf.set_params(kernel='linear').fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
>>> clf.predict(X_test)
array([1, 0, 1, 1, 0])

>>> clf.set_params(kernel='rbf').fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
>>> clf.predict(X_test)
array([0, 0, 0, 1, 0])
```

这里, 默认核**rbf**在概率分布被**SVC()**构造后会首次转为线性的, 然后转回rbf重新拟合概率分布, 并进行第二次预测.
