# Lecture 05 - Training versus Testing

上节课，我们主要介绍了机器学习的可行性。首先，由 NFL 定理可知，机器学习貌似是不可行的。但是，随后引入了统计学知识，如果样本数据足够大，且 ${hypothesis}$ 个数有限，那么机器学习一般就是可行的。本节课将讨论机器学习的核心问题，严格证明为什么机器可以学习。从上节课最后的问题出发，即当 ${hypothesis}$ 的个数是无限多的时候，机器学习的可行性是否仍然成立？

## Recap and Preview

我们先来看一下基于统计学的机器学习流程图：

![the 'Statistical' Learning Flow](http://ofqm89vhw.bkt.clouddn.com/f48333f54e9c141ec8c7328e2e1db9c4.png)

该流程图中，训练样本 ${D}$ 和最终测试 ${h}$ 的样本都是来自同一个数据分布，这是机器能够学习的前提。另外，训练样本 ${D}$ 应该足够大，且 ${hypothesis\ set}$ 的个数是有限的，这样根据霍夫丁不等式，才不会出现 ${Bad\ Data}$，保证 ${E_{in} \approx E_{out}}$，即有很好的泛化能力。同时，通过训练，得到使 ${E_{in}}$ 最小的 ${h}$，作为模型最终的矩 ${g}$，${g}$ 接近于目标函数。

这里，我们总结一下前四节课的主要内容：第一节课，我们介绍了机器学习的定义，目标是找出最好的矩 ${g}$，使 ${g \approx f}$ ，保证 ${E_{out}(g) \approx 0}$；第二节课，我们介绍了如何让 ${E_{in} \approx 0}$，可以使用 PLA、pocket 等演算法来实现；第三节课，我们介绍了机器学习的分类，我们的训练样本是批量数据（batch），处理监督式（supervised）二元分类（binary classification）问题；第四节课，我们介绍了机器学习的可行性，通过统计学知识，把 ${E_{in}(g)}$ 与 ${E_{out}(g)}$ 联系起来，证明了在一些条件假设下，${E_{in}(g) \approx E_{out}(g)}$ 成立。

![Two Central Questions](http://ofqm89vhw.bkt.clouddn.com/4b3c2fe305ec71c29eaab4df93739229.png)

这四节课总结下来，我们把机器学习的主要目标分成两个核心的问题：

- ${E_{in}(g) \approx E_{out}(g)}$。
- ${E_{in}(g)}$ 足够小。

上节课介绍的机器学习可行的一个条件是 ${hypothesis\ set}$的个数 ${M}$ 是有限的，那 ${M}$ 跟上面这两个核心问题有什么联系呢？

我们先来看一下，当 ${M}$ 很小的时候，由上节课介绍的霍夫丁不等式，得到 ${E_{in}(g) \approx E_{out}(g)}$，即能保证第一个核心问题成立。但 ${M}$ 很小时，演算法 ${A}$ 可以选择的 ${hypothesis}$ 有限，不一定能找到使 ${E_{in}(g)}$ 足够小的 ${hypothesis}$，即不能保证第二个核心问题成立。当 ${M}$ 很大的时候，同样由霍夫丁不等式，${E_{in}(g)}$ 与 ${E_{out}(g)}$ 的差距可能比较大，第一个核心问题可能不成立。而 ${M}$ 很大，使的演算法 ${A}$ 的可以选择的 ${hypothesis}$ 就很多，很有可能找到一个 ${hypothesis}$，使 ${E_{in}(g)}$ 足够小，第二个核心问题可能成立。

![Trade-off on M](http://ofqm89vhw.bkt.clouddn.com/a782ae19da6e38f9ada73c62c7b4a8cd.png)

从上面的分析来看， ${M}$ 的选择直接影响机器学习两个核心问题是否满足， ${M}$ 不能太大也不能太小。那么如果 ${M}$ 无限大的时候，是否机器就不可以学习了呢？例如 ${PLA}$ 算法中直线是无数条的，但是 ${PLA}$ 能够很好地进行机器学习，这又是为什么呢？如果我们能将无限大的 ${M}$ 限定在一个有限的 ${m_H}$内，问题似乎就解决了。

## Effective Number of Line

我们先看一下上节课推导的霍夫丁不等式：

$${P[| E_{in}(g) - E_{out}(g) | > \epsilon]\leq 2 \cdot M \cdot exp(- 2 \epsilon^2 N)}$$

其中， ${M}$ 表示 ${hypothesis}$ 的个数。每个 ${hypothesis}$ 下的 BAD events ${B_m}$ 级联的形式满足下列不等式：

$${P[B_1 or B_2 or \cdots B_M] \leq P[B_1]+P[B_2]+\cdots +P[B_M]}$$

当 ${M = \infty}$时，上面不等式右边值将会很大，似乎说明 BAD events 很大，${E_{in}(g)}$ 与 ${E_{out}(g)}$ 也并不接近。但是 BAD events ${B_m}$ 级联的形式实际上是扩大了上界，union bound 过大。这种做法假设各个 ${hypothesis}$ 之间没有交集，这是最坏的情况，可是实际上往往不是如此，很多情况下，都是有交集的，也就是说 ${M}$ 实际上没那么大，如下图所示：

![Where Did Uniform Bound Fail?](http://ofqm89vhw.bkt.clouddn.com/8f2e2b7a6979e15fb59f96593c427e19.png)

也就是说 union bound 被估计过高了（over-estimating）。所以，我们的目的是找出不同 BAD events 之间的重叠部分，也就是将无数个 ${hypothesis}$ 分成有限个类别。

如何将无数个 ${hypothesis}$ 分成有限类呢？我们先来看这样一个例子，假如平面上用直线将点分开，也就跟 PLA 一样。如果平面上只有一个点 ${x_1}$，那么直线的种类有两种：一种将 ${x_1}$ 划为 ${+1}$，一种将 ${x_1}$ 划为 ${-1}$ ：

![How Many Lines Are There?(1/2)](http://ofqm89vhw.bkt.clouddn.com/9e850c2b8c58fd593acd4e38fe3ddc60.png)

如果平面上有两个点 ${x_1}$ 、 ${x_2}$，那么直线的种类共4种： ${x_1}$ 、 ${x_2}$ 都为 ${+1}$，${x_1}$ 、 ${x_2}$ 都为 ${-1}$，${x_1}$ 为 ${+1}$ 且 ${x_2}$ 为 ${-1}$，${x_1}$ 为 ${-1}$ 且 ${x_2}$ 为 ${+1}$ ：

![How Many Lines Are There?(2/2)](http://ofqm89vhw.bkt.clouddn.com/030bb7ebb42899f7508b5d1a0181c59b.png)

如果平面上有三个点 ${x_1}$ 、 ${x_2}$ 、 ${x_3}$，那么直线的种类共8种：

![How Many Kinds of Lines for Three Inputs? (1/2)](http://ofqm89vhw.bkt.clouddn.com/0e31cb678c049691df8606c6e4e605c4.png)

但是，在三个点的情况下，也会出现不能用一条直线划分的情况：

![How Many Kinds of Lines for Three Inputs? (2/2)](http://ofqm89vhw.bkt.clouddn.com/ac5b45513182e332c35d0389e78e8d73.png)

也就是说，对于平面上三个点，不能保证所有的 ${8}$ 个类别都能被一条直线划分。那如果是四个点 ${x_1}$ 、 ${x_2}$ 、 ${x_3}$ 、 ${x_4}$，我们发现，平面上找不到一条直线能将四个点组成的 ${16}$ 个类别完全分开，最多只能分开其中的 ${14}$ 类，即直线最多只有 ${14}$ 种：

![How Many Kinds of Lines for Four Inputs?](http://ofqm89vhw.bkt.clouddn.com/686b39e099077fdfbe3b258853a4d3be.png)

经过分析，我们得到平面上线的种类是有限的，${1}$ 个点最多有 ${2}$ 种线，${2}$ 个点最多有 ${4}$ 种线，${3}$ 个点最多有 ${8}$ 种线，${4}$ 个点最多有 ${14 (<24)}$ 种线等等。我们发现，有效直线的数量总是满足 ${\leq 2^N}$，其中，${N}$ 是点的个数。所以，如果我们可以用 ${effective(N)}$ 代替 ${M}$，霍夫丁不等式可以写成：

$${P[|E_{in}(g) - E_{out}(g)|>\epsilon] \leq 2 \cdot effective(N) \cdot exp(- 2\epsilon^2 N)}$$

已知 ${effective(N)< 2^N}$，如果能够保证 ${effective(N) \ll 2^N}$，即不等式右边接近于零，那么即使 ${M}$ 无限大，直线的种类也很有限，机器学习也是可能的。

## Effective Number of Hypotheses

接下来先介绍一个新名词：二分类（dichotomy）。dichotomy 就是将空间中的点（例如二维平面）用一条直线分成正类（蓝色 ${O}$）和负类（红色 ${X}$）。令 ${H}$是将平面上的点用直线分开的所有 ${hypothesis}$ h的集合，dichotomy ${H}$ 与hypotheses ${H}$ 的关系是：hypotheses ${H}$ 是平面上所有直线的集合，个数可能是无限个，而dichotomy ${H}$ 是平面上能将点完全用直线分开的直线种类，它的上界是 ${2^N}$ 。接下来，我们要做的就是尝试用 dichotomy 代替 ${M}$。

![Dichotomies: Mini-hypotheses](http://ofqm89vhw.bkt.clouddn.com/6ed45dfed835a97902f229d90c2170bc.png)

再介绍一个新的名词：成长函数（growth function），记为 ${m_{H}(H)}$ 。成长函数的定义是：对于由 ${N}$ 个点组成的不同集合中，某集合对应的 dichotomy 最大，那么这个 dichotomy 值就是 ${m_{H}(H)}$，它的上界是 ${2^N}$，成长函数其实就是我们之前讲的 effective lines 的数量最大值。根据成长函数的定义，二维平面上，${m_{H}(H)}$ 随 ${N}$ 的变化关系是：

![Growth Function](http://ofqm89vhw.bkt.clouddn.com/fcc0c188a62e6eacf455c9109c242a64.png)

接下来，我们讨论如何计算成长函数。先看一个简单情况，一维的 Positive Rays：

![Growth Function for Positive Rays](http://ofqm89vhw.bkt.clouddn.com/a09e25428c69d9bea3ace2a9cbe0dad2.png)

若有 ${N}$ 个点，则整个区域可分为 ${N+1}$ 段，很容易得到其成长函数 ${m_{H}(N) = N + 1}$ 。注意当N很大时，${(N+1) \ll 2^N}$，这是我们希望看到的。

另一种情况是一维的 Positive Intervals：

![Growth Function for Positive Intervals](http://ofqm89vhw.bkt.clouddn.com/2daa365690eb12199cf537228ec5f681.png)

这种情况下，${m_{H}(N)=\frac{1}{2}N^2 +\frac{1}{2} N +1 \ll 2^N}$，在 ${N}$ 很大的时候，仍然是满足的。

再来看这个例子，假设在二维空间里，如果 ${hypothesis}$ 是凸多边形或类圆构成的封闭曲线，如下图所示，左边是 convex 的，右边不是 convex 的。那么，它的成长函数是多少呢？

![Growth Function for Convex Sets](http://ofqm89vhw.bkt.clouddn.com/4cfd1a80f8a0827886ab6bbe97897abb.png)

当数据集 ${D}$ 按照如下的凸分布时，我们很容易计算得到它的成长函数 ${m_H=2^N}$ 。这种情况下，${N}$ 个点所有可能的分类情况都能够被 hypotheses set 覆盖，我们把这种情形称为 shattered。也就是说，如果能够找到一个数据分布集，hypotheses set 对 ${N}$ 个输入所有的分类情况都做得到，那么它的成长函数就是 ${2^N}$。

![Growth Function for Convex Sets](http://ofqm89vhw.bkt.clouddn.com/d4287bb240e13f5045546b588e9d2b40.png)

## Break Point

上一小节，我们介绍了四种不同的成长函数，分别是：

![The Four Break Points](http://ofqm89vhw.bkt.clouddn.com/6c438c01aba154ff8bc67fd03f066550.png)

其中，positive rays 和 positive intervals 的成长函数都是 polynomial 的，如果用 ${m_H}$ 代替 ${M}$ 的话，这两种情况是比较好的。而 convex sets 的成长函数是 exponential 的，即等于 ${M}$，并不能保证机器学习的可行性。那么，对于 2D perceptrons，它的成长函数究竟是 polynomial 的还是 exponential 的呢？

对于 2D perceptrons，我们之前分析了 ${3}$ 个点，可以做出 ${8}$ 种所有的dichotomy，而 ${4}$ 个点，就无法做出所有 ${16}$ 个点的dichotomy了。所以，我们就把 ${4}$ 称为 2D perceptrons 的 break point（5、6、7等都是break point）。**令有 ${k}$ 个点，如果 ${k}$ 大于等于break point时，它的成长函数一定小于 ${2^k}$。**

根据 break point 的定义，我们知道满足 ${m_{H}(k) \neq 2^k}$ 的 ${k}$ 的最小值就是 break point。

通过观察，我们猜测成长函数可能与 break point 存在某种关系：对于 convex sets，没有 break point，它的成长函数是 ${2^N}$；对于 positive rays，break point ${k=2}$，它的成长函数是 ${O(N)}$ ；对于positive intervals，break point ${k=3}$，它的成长函数是 ${O(N^2)}$。则根据这种推论，我们猜测 2D perceptrons，它的成长函数 ${m_{H}(N) =O(N^{k-1})}$。如果成立，那么就可以用 ${m_H}$ 代替${M}$，就满足了机器能够学习的条件。关于上述猜测的证明，我们下节课再详细介绍。

## 总结

本节课，我们更深入地探讨了机器学习的可行性。我们把机器学习拆分为两个核心问题： ${E_{in}(g) \approx E_{out}(g)}$ 和 ${E_{in}(g) \approx 0}$。对于第一个问题，我们探讨了 ${M}$ 个 ${hypothesis}$ 到底可以划分为多少种，也就是成长函数 ${m_H}$。并引入了break point的概念，给出了 break point 的计算方法。下节课，我们将详细论证对于 2D perceptrons，它的成长函数与 break point 是否存在多项式的关系，如果是这样，那么机器学习就是可行的。

## 参考

1. [台湾大学林轩田机器学习基石课程学习笔记5 -- Training versus Testing](http://blog.csdn.net/red_stone1/article/details/71104654)