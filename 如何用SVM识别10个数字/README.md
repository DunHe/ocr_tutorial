# 如何用SVM识别10个数字

### 开门见山

数字识别在模式识别, 尤其是[OCR](http://baike.baidu.com/view/230331.htm?fromtitle=OCR&fromid=25995&type=syn)(光学字符识别)中被广泛应用. 鉴于其的重要性, 本文将介绍一个强大的分类算法: [SVM](http://baike.baidu.com/view/960509.htm)(支持向量机), 分析它的算法过程以及对比其与其他算法如[KNN](http://baike.baidu.com/view/1485833.htm?fromtitle=Knn&fromid=3479559&type=syn)的实现效果.

### 初识SVM


### 华山论剑

#### 科普小知识

> 用知识武装自己的头脑

- true positives(纳真tp): 检测出来的正确的样本数.

- false positives(纳伪fp): 检测出来的错误的样本数.

- false negative(去真fn): 未检测出来的正确的样本数.

- Confusion matrix(模糊矩阵): 当分类结果的数量为N时, 模糊矩阵的大小为N*N. 矩阵的横坐标为预测值, 纵坐标为目标值. 其中对角线上表示预测出正确的信息, 在对角线之外为检测出的不正确的信息或者未检测出的正确的信息.

- Precision(准确率): 被检测出来的信息当中正确的或者相关的(也就是你想要的)信息中所占的比例.  
    计算公式为: tp/(tp + fp)

- Recall(召回率): 所有正确的信息或者相关的信息被检测出来的比例.  
    计算公式为: tp/(tp + fn)

- F1-Measure(综合评价指标): 是Precision和Recall加权调和平均. 当该值越高说明实验方法越好.
    计算公式为: 2PR/(P + R)

> 实践是检验真理的唯一标准.

有了理论基础, 我们还需要借助锋利的武器. sciki-learn, OpenCV, theano等开发库都集成了机器学习的相关内容, 本篇文章使用sciki-learn.

#### 算法流程

- 数据处理
- 分类器训练
- 分类器预测
- 对比不同算法

由于scikit-learn中的KNN分类器只接受维度\<=2的numpy数组, 所以从digits数据集获取到的训练数据需通过reshape转换为1维的数组:

```python
from sklearn import datasets

digits = datasets.load_digits()

n_samples = len(digits.images)

data = digits.images.reshape(n_samples, -1)
train_data = data[:n_samples/2]
test_data = data[n_samples/2:]

target_data = digits.target[:n_samples/2]
expected_data = digits.target[n_samples/2:]
```

**KNN分类器训练与预测**

由于待分类的数字类别共有10类, 所以设置n_neighbors为10:

```python
from sklearn.neighbors import KNeighborsClassifier

neigh_classifier = KNeighborsClassifier(n_neighbors=10)
neigh_classifier.fit(train_data, target_data)

neigh_predicted = neigh_classifier.predict(test_data)
```

**SVM分类器训练与预测**

```python
from sklearn import svm

svm_classifier = svm.SVC(gamma=0.001)
svm_classifier.fit(train_data, target_data)

svm_predicted = svm_classifier.predict(test_data)
```

为了对比KNN和SVM两种算法在数字识别上精确度的差别, 本文采用归一化的模糊矩阵以及精确度, 召回度和综合评价指标进行对比. 为了使模糊矩阵的输出更为直观, 需对numpy的输出格式进行设置:

```python
np.set_printoptions(precision=2)
```

[metrics.confusion_matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)函数根据目标值和预测值生成模糊矩阵, [metrics.classification_report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)函数则根据目标值和预测值生成精确度等参数. 以下分别为KNN和SVM关于这两个函数的输出结果:

```python
neigh_mat = metrics.confusion_matrix(expected_data, neigh_predicted)
print metrics.classification_report(expected_data, neigh_predicted)
print neigh_mat.astype('float') / neigh_mat.sum(axis=1)[:, np.newaxis]
```

```python
                KNN算法识别10个数字精确度估计:
             precision    recall  f1-score   support

          0       0.98      1.00      0.99        88
          1       0.95      0.98      0.96        91
          2       0.98      0.93      0.95        86
          3       0.89      0.91      0.90        91
          4       1.00      0.93      0.97        92
          5       0.95      0.98      0.96        91
          6       0.99      1.00      0.99        91
          7       0.94      1.00      0.97        89
          8       0.96      0.88      0.92        88
          9       0.92      0.93      0.93        92

avg / total       0.96      0.95      0.95       899

[[ 1.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.98  0.    0.    0.    0.01  0.    0.    0.    0.01]
 [ 0.01  0.    0.93  0.06  0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.91  0.    0.02  0.    0.04  0.01  0.01]
 [ 0.01  0.    0.    0.    0.93  0.    0.    0.01  0.01  0.03]
 [ 0.    0.    0.    0.    0.    0.98  0.01  0.    0.    0.01]
 [ 0.    0.    0.    0.    0.    0.    1.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    1.    0.    0.  ]
 [ 0.    0.06  0.02  0.02  0.    0.    0.    0.01  0.88  0.01]
 [ 0.    0.    0.    0.03  0.    0.02  0.    0.    0.01  0.93]]
```

**SVM**

```python
svm_mat = metrics.confusion_matrix(expected_data, svm_predicted)
print metrics.classification_report(expected_data, svm_predicted)
print svm_mat.astype('float') / svm_mat.sum(axis=1)[:, np.newaxis]
```

```python
                SVM算法识别10个数字精确度估计:
             precision    recall  f1-score   support

          0       1.00      0.99      0.99        88
          1       0.99      0.97      0.98        91
          2       0.99      0.99      0.99        86
          3       0.98      0.87      0.92        91
          4       0.99      0.96      0.97        92
          5       0.95      0.97      0.96        91
          6       0.99      0.99      0.99        91
          7       0.96      0.99      0.97        89
          8       0.94      1.00      0.97        88
          9       0.93      0.98      0.95        92

avg / total       0.97      0.97      0.97       899

Confusion matrix:
[[ 0.99  0.    0.    0.    0.01  0.    0.    0.    0.    0.  ]
 [ 0.    0.97  0.01  0.    0.    0.    0.    0.    0.01  0.01]
 [ 0.    0.    0.99  0.01  0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.87  0.    0.03  0.    0.04  0.05  0.  ]
 [ 0.    0.    0.    0.    0.96  0.    0.    0.    0.    0.04]
 [ 0.    0.    0.    0.    0.    0.97  0.01  0.    0.    0.02]
 [ 0.    0.01  0.    0.    0.    0.    0.99  0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.01  0.    0.99  0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    1.    0.  ]
 [ 0.    0.    0.    0.01  0.    0.01  0.    0.    0.    0.98]]
 ```

观察输出结果, SVM在precision, recall, f1-score以及模糊矩阵对角线上的数值均比KNN算法的数值大, 可见SVM在数字识别方面是比KNN更优化的算法.

### 参考
[Confusion matrix](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)

[Precision, Recall, F1-Meature的含义及计算](http://blog.csdn.net/t710smgtwoshima/article/details/8215037)
