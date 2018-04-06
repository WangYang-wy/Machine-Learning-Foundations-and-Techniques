# Lecture 13 - Hazard of Overfitting

上节课我们主要介绍了非线性分类模型，通过非线性变换，将非线性模型映射到另一个空间，转换为线性模型，再来进行分类，分析了非线性变换可能会使计算复杂度增加。本节课介绍这种模型复杂度增加带来机器学习中一个很重要的问题：过拟合（overfitting）。

## What is Overfitting

首先，我们通过一个例子来介绍什么 bad generalization。假设平面上有 ${5}$ 个点，目标函数 ${f(x)}$ 是 ${2}$ 阶多项式，如果 hypothesis 是二阶多项式加上一些小的 noise 的话，那么这 ${5}$ 个点很靠近这个 hypothesis，${E_{in}}$ 很小。如果 hypothesis 是 ${4}$ 阶多项式，那么这 ${5}$ 点会完全落在 hypothesis 上，${E_{in} =0}$ 。虽然 ${4}$ 阶 hypothesis 的 ${E_{in}}$ 比 ${2}$ 阶 hypothesis 的要好很多，但是它的 ${E_{out}}$ 很大。因为根据 VC Bound 理论，阶数越大，即 VC Dimension 越大，就会让模型复杂度更高，${E_{out}}$ 更大。我们把这种 ${E_{in}}$ 很小，${E_{out}}$ 很大的情况称之为 bad generation，即泛化能力差。

![Bad Generalization](http://ofqm89vhw.bkt.clouddn.com/4531811ad2eb75f291324513c942c335.png)

我们回过头来看一下VC曲线：

![Bad Generalization and Overfitting](http://ofqm89vhw.bkt.clouddn.com/ded7dd33a45601bb5189d586208bf0a8.png)

Hypothesis 的阶数越高，表示 VC Dimension 越大。随着 VC Dimension 增大，${E_{in}}$ 是一直减小的，而 ${E_{out}}$ 先减小后增大。在${d^{\ast}}$位置，${E_{out}}$ 取得最小值。在 ${d_{VC}^{\ast}}$右侧，随着 VC Dimension 越来越大，${E_{in}}$ 越来越小，接近于 ${0}$，${E_{out}}$ 越来越大。即当 VC Dimension 很大的时候，这种对训练样本拟合过分好的情况称之为过拟合（overfitting）。另一方面，在 ${d_{VC}^{\ast}}$ 左侧，随着 VC Dimension 越来越小，${E_{in}}$ 和 ${E_{out}}$ 都越来越大，这种情况称之为欠拟合（underfitting），即模型对训练样本的拟合度太差，VC Dimension 太小了。

Bad generation 和 overfitting 的关系可以理解为：overfitting 是 VC Dimension 过大的一个过程，bad generation 是 overfitting 的结果。

![Cause of Overfitting: A Driving Analogy](http://ofqm89vhw.bkt.clouddn.com/d81f51511392073a01738fc8d9ca1c9c.png)

一个好的 fit，${E_{in}}$ 和 ${E_{out}}$ 都比较小，尽管 ${E_{in}}$ 没有足够接近零；而对 overfitting 来说，${E_{in} \approx 0}$，但是 ${E_{out}}$ 很大。那么，overfitting 的原因有哪些呢？

我们举个开车的例子，把发生车祸比作成 overfitting，那么造成车祸的原因包括：

- 车速太快（VC Dimension太大）；
- 道路崎岖（noise）；
- 对路况的了解程度（训练样本数量 ${N}$ 不够）；

也就是说，VC Dimension、noise、${N}$ 这三个因素是影响过拟合现象的关键。

## The Role of Noise and Data Size

为了尽可能详细地解释 overfitting，我们进行这样一个实验，试验中的数据集不是很大。首先，在二维平面上，一个模型的分布由目标函数 ${f(x)}$ （${x}$ 的 ${10}$ 阶多项式）加上一些 noise 构成，下图中，离散的圆圈是数据集，目标函数是蓝色的曲线。数据没有完全落在曲线上，是因为加入了 noise。

![Case Study (1/2)](http://ofqm89vhw.bkt.clouddn.com/6814d140f6a2d2c93380fa6964136681.png)

然后，同样在二维平面上，另一个模型的分布由目标函数 ${f(x)}$ （${x}$ 的 ${50}$ 阶多项式）构成，没有加入 noise。下图中，离散的圆圈是数据集，目标函数是蓝色的曲线。可以看出由于没有 noise，数据集完全落在曲线上。

![Case Study (2/2)](http://ofqm89vhw.bkt.clouddn.com/38cb42468bf56b54a42e0de37ebe4f65.png)

现在，有两个学习模型，一个是 ${2}$ 阶多项式，另一个是 ${10}$ 阶多项式，分别对上面两个问题进行建模。首先，对于第一个目标函数是 ${10}$ 阶多项式包含 noise 的问题，这两个学习模型的效果如下图所示：

![Irony of Two Learners](http://ofqm89vhw.bkt.clouddn.com/d296217a3be898f28426c922484fc126.png)

由上图可知，${2}$ 阶多项式的学习模型 ${E_{in} =0.050}$，${E_{out} =0.127}$；${10}$ 阶多项式的学习模型 ${E_{in} =0.034}$，${E_{out} =9.00}$。虽然 ${10}$ 阶模型的 ${E_{in}}$ 比 ${2}$ 阶的小，但是其 ${E_{out}}$ 要比 ${2}$ 阶的大得多，而 ${2}$ 阶的 ${E_{in}}$ 和 ${E_{out}}$ 相差不大，很明显用 ${10}$ 阶的模型发生了过拟合。

然后，对于第二个目标函数是 ${50}$ 阶多项式没有 noise 的问题，这两个学习模型的效果如图所示。

${2}$ 阶多项式的学习模型 ${E_{in} =0.029}$，${E_{out} =0.120}$；${10}$ 阶多项式的学习模型 ${E_{in} =0.00001}$，${E_{out} =7680}$。虽然 ${10}$ 阶模型的 ${E_{in}}$ 比 ${2}$ 阶的小，但是其 ${E_{out}}$ 要比 ${2}$ 阶的大得多的多，而 ${2}$ 阶的 ${E_{in}}$ 和 ${E_{out}}$ 相差不大，很明显用 ${10}$ 阶的模型仍然发生了明显的过拟合。

上面两个问题中，${10}$ 阶模型都发生了过拟合，反而 ${2}$ 阶的模型却表现得相对不错。这好像违背了我们的第一感觉，比如对于目标函数是 ${10}$ 阶多项式，加上noise的模型，按道理来说应该是 ${10}$ 阶的模型更能接近于目标函数，因为它们阶数相同。但是，事实却是 ${2}$ 阶模型泛化能力更强。这种现象产生的原因，从哲学上来说，就是“以退为进”。有时候，简单的学习模型反而能表现的更好。

下面从 learning curve 来分析一下具体的原因， learning curve 描述的是 ${E_{in}}$ 和 ${E_{out}}$ 随着数据量 ${N}$ 的变化趋势。下图中左边是 ${2}$ 阶学习模型的 learning curve，右边是 ${10}$ 阶学习模型的 learning curve。

![Learning Curves Revisited](http://ofqm89vhw.bkt.clouddn.com/c2ba28b4cc70052a479def15fc31e839.png)

在 learning curve中，横轴是样本数量 ${N}$，纵轴是 Error。 ${E_{in}}$ 和 ${E_{out}}$ 可表示为：

$${E_{in} =noiselevel \ast (1 - \frac{d+1}{N})}$$

$${E_{out} = noiselevel \ast (1 + \frac{d+1}{N})}$$

其中 ${d}$ 为模型阶次，左图中 ${d=2}$，右图中${d=10}$。

本节的实验问题中，数据量 ${N}$ 不大，即对应于上图中的灰色区域。左图的灰色区域中，因为 ${d=2}$，${E_{in}}$ 和 ${E_{out}}$ 相对来说比较接近；右图中的灰色区域中，${d=10}$，根据 ${E_{in}}$ 和 ${E_{out}}$ 的表达式，${E_{in}}$ 很小，而 ${E_{out}}$ 很大。这就解释了之前 ${2}$ 阶多项式模型的 ${E_{in}}$ 更接近 ${E_{out}}$，泛化能力更好。

值得一提的是，如果数据量 ${N}$ 很大的时候，上面两图中 ${E_{in}}$ 和 ${E_{out}}$ 都比较接近，但是对于高阶模型，${z}$ 域中的特征很多的时候，需要的样本数量 ${N}$ 很大，且容易发生维度灾难。

另一个例子中，目标函数是 ${50}$ 阶多项式，且没有加入 noise。这种情况下，我们发现仍然是 ${2}$ 阶的模型拟合的效果更好一些，明明没有 noise，为什么是这样的结果呢？

实际上，我们忽略了一个问题：这种情况真的没有 noise 吗？其实，当模型很复杂的时候，即 ${50}$ 阶多项式的目标函数，无论是 ${2}$ 阶模型还是 ${10}$ 阶模型，都不能学习的很好，这种复杂度本身就会引入一种 ‘noise’。所以，这种高阶无 noise 的问题，也可以类似于 ${10}$ 阶多项式的目标函数加上 noise 的情况，只是二者的 noise 有些许不同，下面一部分将会详细解释。

## Deterministic Noise

下面我们介绍一个更细节的实验来说明 什么时候小心 overfit 会发生。假设我们产生的数据分布由两部分组成：第一部分是目标函数${f(x)}$，${Q_f}$阶多项式；第二部分是噪声 ${\epsilon}$，服从 Gaussian 分布。接下来我们分析的是 noise 强度不同对 overfitting 有什么样的影响。总共的数据量是 ${N}$。

![A Detailed Experiment](http://ofqm89vhw.bkt.clouddn.com/50c333e33e7c41f6870622da28a25d91.png)

那么下面我们分析不同的 ${(N,\sigma^2)}$ 和 ${(N,Q_f)}$ 对 overfit 的影响。overfit 可以量化为 ${E_{out} - E_{in}}$ 。结果如下：

![The Results](http://ofqm89vhw.bkt.clouddn.com/f053250593ee67347c27b30ecf8afac4.png)

上图中，红色越深，代表 overfit 程度越高，蓝色越深，代表 overfit 程度越低。先看左边的图，左图中阶数 ${Q_f}$ 固定为 ${20}$，横坐标代表样本数量 ${N}$，纵坐标代表噪声水平 ${\sigma^2}$。红色区域集中在 ${N}$ 很小或者 ${\sigma^2}$ 很大的时候，也就是说 ${N}$ 越大，${\sigma^2}$越小，越不容易发生 overfit。右边图中 ${\sigma^2=0.1}$，横坐标代表样本数量 ${N}$，纵坐标代表目标函数阶数 ${Q_f}$ 。红色区域集中在 ${N}$ 很小或者 ${Q_f}$ 很大的时候，也就是说 ${N}$ 越大，${Q_f}$ 越小，越不容易发生 overfit。上面两图基本相似。

从上面的分析，我们发现 ${\sigma^2}$ 对 overfit 是有很大的影响的，我们把这种 noise 称之为 stochastic noise。同样地，${Q_f}$ 即模型复杂度也对 overfit 有很大影响，而且二者影响是相似的，所以我们把这种称之为 deterministic noise。之所以把它称为 noise，是因为模型高复杂度带来的影响。

总结一下，有四个因素会导致发生 overfitting ：

- data size N ${\downarrow}$
- stochastic noise ${\sigma^2}$ ${\uparrow}$
- deterministic noise ${Q_f}$ ${\uparrow}$
- excessive power ${\uparrow}$

我们刚才解释了如果目标函数 ${f(x)}$ 的复杂度很高的时候，那么跟有 noise 也没有什么两样。因为目标函数很复杂，那么再好的 hypothesis 都会跟它有一些差距，我们把这种差距称之为 deterministic noise。deterministic noise 与 stochastic noise 不同，但是效果一样。其实 deterministic noise 类似于一个伪随机数发生器，它不会产生真正的随机数，而只产生伪随机数。它的值与 hypothesis 有关，且固定点 ${x}$ 的 deterministic noise 值是固定的。

## Dealing with Overfitting

现在我们知道了什么是 overfitting，和 overfitting 产生的原因，那么如何避免 overfitting 呢？避免 overfitting 的方法主要包括：

- start from simple model
- data cleaning/pruning
- data hinting
- regularization
- validataion

这几种方法类比于之前举的开车的例子，对应如下：

![Driving Analogy Revisited](http://ofqm89vhw.bkt.clouddn.com/7a97f92740b30b9da11abab5563f6512.png)

regularization 和 validation 我们之后的课程再介绍，本节课主要介绍简单的 data cleaning/pruning 和 data hinting 两种方法。

Data cleaning/pruning 就是对训练数据集里 label 明显错误的样本进行修正（data cleaning），或者对错误的样本看成是 noise，进行剔除（data pruning）。Data cleaning/pruning 关键在于如何准确寻找 label 错误的点或者是 noise 的点，而且如果这些点相比训练样本 ${N}$ 很小的话，这种处理效果不太明显。

Data hinting是针对 ${N}$ 不够大的情况，如果没有办法获得更多的训练集，那么 data hinting 就可以对已知的样本进行简单的处理、变换，从而获得更多的样本。举个例子，数字分类问题，可以对已知的数字图片进行轻微的平移或者旋转，从而让N丰富起来，达到扩大训练集的目的。这种额外获得的例子称之为 virtual examples。但是要注意一点的就是，新获取的 virtual examples 可能不再是 ${iid}$ 某个 distribution。所以新构建的 virtual examples 要尽量合理，且是独立同分布的。

## 总结

本节课主要介绍了 overfitting 的概念，即当 ${E_{in}}$ 很小，${E_{out}}$ 很大的时候，会出现 overfitting。详细介绍了 overfitting 发生的四个常见原因data size ${N}$、stochastic noise、deterministic noise 和 excessive power。解决overfitting的方法有很多，本节课主要介绍了 data cleaning/pruning 和 data hinting 两种简单的方法，之后的课程将会详细介绍 regularization 和 validataion 两种更重要的方法。

## 参考

1. [台湾大学林轩田机器学习基石课程学习笔记13 -- Hazard of Overfitting](http://blog.csdn.net/red_stone1/article/details/72673204)
1. [机器学习中的维度灾难](http://blog.csdn.net/red_stone1/article/details/71692444)