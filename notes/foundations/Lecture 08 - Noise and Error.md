# Lecture 08 - Noise and Error

上一节课，我们主要介绍了 VC Dimension 的概念。如果 Hypotheses set 的 VC Dimension 是有限的，且有足够多 ${N}$ 的资料，同时能够找到一个 hypothesis 使它的 ${E_{in} \approx 0}$，那么就能说明机器学习是可行的。本节课主要讲了数据集有 Noise 的情况下，是否能够进行机器学习，并且介绍了假设空间 ${H}$ 下演算法 ${A}$ 的 Error 估计。

## Noise and Probablistic target

上节课推导 VC Dimension 的数据集是在没有 Noise 的情况下，本节课讨论如果数据集本身存在 Noise，那 VC Dimension 的推导是否还成立呢？

首先，Data Sets 的 Noise 一般有三种情况：

- 由于人为因素，正类被误分为负类，或者负类被误分为正类；
- 同样特征的样本被模型分为不同的类；
- 样本的特征被错误记录和使用。

![Recap: The Learning Flow](http://ofqm89vhw.bkt.clouddn.com/39640a1cfd51c41d649c50361d343225.png)

之前的数据集是确定的，即没有 Noise 的，我们称之为 Deterministic。现在有 Noise 了，也就是说在某点处不再是确定分布，而是概率分布了，即对每个 ${(x,\ y)}$ 出现的概率是 ${P(y|x)}$ 。

因为 Noise 的存在，比如在 ${x}$ 点，有 ${0.7}$ 的概率 ${y=1}$，有 ${0.3}$ 的概率 ${y=0}$，即 ${y}$ 是按照 ${P(y|x)}$ 分布的。数学上可以证明如果数据集按照 ${P(y|x)}$ 概率分布且是 ${iid}$ 的，那么以前证明机器可以学习的方法依然奏效，VC Dimension 有限即可推断 ${E_{in}}$ 和 ${E_{out}}$ 是近似的。

![Probabilistic Marbles](http://ofqm89vhw.bkt.clouddn.com/d18c834127c142fff94de66c76c60984.png)

${P(y|x)}$ 称之为目标分布（Target Distribution）。它实际上告诉我们最好的选择是什么，同时伴随着多少 noise。其实，没有 noise 的数据仍然可以看成“特殊”的 ${P(y|x)}$ 概率分布，即概率仅是 ${1}$ 和 ${0}$ 。对于以前确定的数据集：

- ${P(y|x) =1,\ for \ y = f(x)}$
- ${P(y|x) =0,\ for \ y \neq f(x)}$

![Target Distribution ${P(y|x)}$](http://ofqm89vhw.bkt.clouddn.com/00f5c27558dc49c923ce1f6f2924e987.png)

在引入 noise 的情况下，新的学习流程图如下所示：

![The New Learning Flow](http://ofqm89vhw.bkt.clouddn.com/fce073e862502bf9100a53b962baf81b.png)

## ERROR Measure

机器学习需要考虑的问题是找出的 ${g}$ 与目标函数 ${f}$ 有多相近，我们一直使用 ${E_{out}}$ 进行误差的估计，那一般的错误测量有哪些形式呢？

我们介绍的矩g对错误的衡量有三个特性：

- out-of-sample：样本外的未知数据。
- pointwise：对每个数据点 ${x}$ 进行测试。
- classification：看 prediction 与 target 是否一致，classification error 通常称为 0/1 error。

![Error Measure](http://ofqm89vhw.bkt.clouddn.com/bacaf80c234cd7efef9767de8fc34208.png)

Pointwise error 实际上就是对数据集的每个点计算错误并计算平均，${E_{in}}$ 和 ${E_{out}}$ 的pointwise error的表达式为：

![Pointwise Error Measure](http://ofqm89vhw.bkt.clouddn.com/72d887c9d63aaa474224183b2a36f46c.png)

Pointwise error 是机器学习中最常用也是最简单的一种错误衡量方式，未来课程中，我们主要考虑这种方式。Pointwise error 一般可以分成两类：0/1 error 和 squared error。0/1 error 通常用在分类（classification）问题上，而 squared error 通常用在回归（regression）问题上。

![Two Important Pointwise Error Measures](http://ofqm89vhw.bkt.clouddn.com/374bfefd007efddfac8f374ce249e92f.png)

Ideal Mini-Target 由 ${P(y|x)}$ 和 err 共同决定，0/1 error 和 squared error 的 Ideal Mini-Target 计算方法不一样。例如下面这个例子，分别用 0/1 error 和 squared error 来估计最理想的mini-target 是多少。0/1 error 中的 mini-target 是取 ${P(y|x)}$ 最大的那个类，而 squared error 中的 mini-target 是取所有类的加权平方和。

![Ideal Mini-Target](http://ofqm89vhw.bkt.clouddn.com/f70e2eb4c6b2dd9681789f090b665246.png)

有了错误衡量，就会知道当前的矩 ${g}$ 是好还是不好，并会让演算法不断修正，得到更好的矩 ${g}$，从而使得 ${g}$ 与目标函数更接近。所以，引入 error measure 后，学习流程图如下所示：

![Learning Flow with Error Measure](http://ofqm89vhw.bkt.clouddn.com/210548bbe7ce3009c7f75f4b48fca3ea.png)

## Algorithmic Error Measure

Error 有两种：false accept 和 false reject。false accept 意思是误把负类当成正类，false reject 是误把正类当成负类。根据不同的机器学习问题，false accept 和 false reject 应该有不同的权重，这根实际情况是符合的，比如是超市优惠，那么 false reject 应该设的大一些；如果是安保系统，那么 false accept 应该设的大一些。

![two types of error](http://ofqm89vhw.bkt.clouddn.com/6217812e195650d3bc5380db691b39be.png)

机器学习演算法 ${A}$ 的 cost function error 估计有多种方法，真实的 err 一般难以计算，常用的方法可以采用 plausible 或者 friendly，根据具体情况而定。

![two types of error: false accept and false reject](http://ofqm89vhw.bkt.clouddn.com/a4c35b0779ae1531a865bf1e91f2e44c.png)

引入 algorithm error measure 之后，学习流程图如下：

![Learning Flow with Algorithmic Error Measure](http://ofqm89vhw.bkt.clouddn.com/cc5a9f17bbb072f1de2157b429ee294d.png)

## Weighted Classification

实际上，机器学习的 Cost Function 即来自于这些 error，也就是算法里面的迭代的目标函数，通过优化使得 ${Error(E_{in})}$ 不断变小。
cost function 中，false accept 和 false reject 赋予不同的权重，在演算法中体现。对不同权重的错误惩罚，可以选用 `virtual copying` 的方法。

![copying examples](http://ofqm89vhw.bkt.clouddn.com/93f35e2ad6996213740484cac4a7e6e0.png)

## 总结

本节课主要讲了在有 Noise 的情况下，即数据集按照 ${P(y|x)}$ 概率分布，那么 VC Dimension 仍然成立，机器学习算法推导仍然有效。机器学习 cost function 常用的 Error 有 0/1 error 和 squared error 两类。实际问题中，对 false accept 和 false reject 应该选择不同的权重。

## 参考

1. [台湾大学林轩田机器学习基石课程学习笔记8 -- Noise and Error](http://blog.csdn.net/red_stone1/article/details/71512186)