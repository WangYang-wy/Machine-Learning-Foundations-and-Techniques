# Lecture 06 - Theory of Generalization

上一节课，我们主要探讨了当 ${M}$ 的数值大小对机器学习的影响。如果 ${M}$ 很大，那么就不能保证机器学习有很好的泛化能力，所以问题转换为验证 ${M}$ 有限，即最好是按照多项式成长。然后通过引入了成长函数 ${m_{H}(N)}$ 和dichotomy以及break point的概念，提出2D perceptrons的成长函数 ${m_{H}(N)}$ 是多项式级别的猜想。这就是本节课将要深入探讨和证明的内容。

## Restriction of Break Point

我们先回顾一下上节课的内容，四种成长函数与break point的关系：

![The Four Break Points](http://ofqm89vhw.bkt.clouddn.com/48a1dfde0a043904b2c864442bb7acaf.png)

下面引入一个例子，如果 ${k=2}$ ，那么当 ${N}$ 取不同值的时候，计算其成长函数 ${m_{H}(N)}$ 是多少。很明显，当 ${N=1}$ 时，${m_{H}(N) = 2}$ ；当 ${N=2}$ 时，由break point为2可知，任意两点都不能被shattered（shatter的意思是对 ${N}$ 个点，能够分解为 ${2^{N}}$ 种dichotomies）； ${m_{H}(N)}$ 最大值只能是3；当 ${N=3}$ 时，简单绘图分析可得其 ${m_{H}(N)}$ =4，即最多只有4种dichotomies。

这里写图片描述

所以，我们发现当 ${N>k}$ 时，break point限制了 ${m_{H}(N)}$ 值的大小，也就是说影响成长函数 ${m_{H}(N)}$ 的因素主要有两个：

- 抽样数据集N。
- break point ${k}$（这个变量确定了假设的类型）。

那么，如果给定 ${N}$ 和 ${k}$，能够证明其 ${m_{H}(N)}$ 的最大值的上界是多项式的，则根据霍夫丁不等式，就能用 ${m_{H}(N)}$ 代替 ${M}$，得到机器学习是可行的。所以，证明 ${m_{H}(N)}$ 的上界是 ${poly(N)}$，是我们的目标。

这里写图片描述

## Bounding Function: Basic Cases

现在，我们引入一个新的函数：bounding function，${B(N,K)}$ 。Bound Function指的是当break point为 ${k}$ 的时候，成长函数 ${m_{H}(N)}$ 可能的最大值。也就是说 ${B(N,K)}$ 是 ${m_{H}(N)}$ 的上界，对应 ${m_{H}(N)}$ 最多有多少种dichotomy。那么，我们新的目标就是证明：

$${B(N,K) \leq poly(N)}$$

这里值得一提的是，${B(N,K)}$ 的引入不考虑是1D postive intrervals问题还是2D perceptrons问题，而只关心成长函数的上界是多少，从而简化了问题的复杂度。

![Bounding Function](http://ofqm89vhw.bkt.clouddn.com/dd5447424e25195c06f982ec95406af5.png)

求解 ${B(N,K)}$ 的过程十分巧妙：

- 当 ${k=1}$ 时，${B(N,1)}$ 恒为 ${1}$。
- 当 ${N < k}$ 时，根据break point的定义，很容易得到 ${B(N,K) =2^N}$。
- 当 ${N = k}$ 时，此时 ${N}$ 是第一次出现不能被shatter的值，所以最多只能有 ${2^N−1}$ 个dichotomies，则 ${B(N,K) =2^N−1}$。

![Table of Bounding Function](http://ofqm89vhw.bkt.clouddn.com/f7a4664377cf91fec7d0a98bc5461cfb.png)

到此，bounding function的表格已经填了一半了，对于最常见的 ${N>k}$ 的情况比较复杂，推导过程下一小节再详细介绍。

## Bounding Function: Inductive Cases

${N > k}$ 的情况较为复杂，下面给出推导过程：

以 ${B(4,3)}$ 为例，首先想着能否构建 ${B(4,3)}$ 与${B(3,x)}$ 之间的关系。

首先，把 ${B(4,3)}$ 所有情况写下来，共有11组。也就是说再加一种dichotomy，任意三点都能被shattered，11是极限。

!['Achieving' Dichotomies of B(4, 3)](http://ofqm89vhw.bkt.clouddn.com/ded60a2d565554d7ae89b07c46bcc88c.png)

对这11种dichotomy分组，目前分成两组，分别是orange和purple，orange的特点是，${x_1}$ ,${x_2}$和 ${x_3}$ 是一致的，${x_4}$不同并成对，例如1和5，2和8等，purple则是单一的，${x_1}$, ${x_2}$, ${x_3}$ 都不同，如6,7,9三组。

将Orange去掉 ${x_4}$ 后去重得到 ${4}$ 个不同的vector并成为 ${\alpha}$，相应的purple为 ${\beta}$。那么 ${B(4,3) = 2 \alpha + \beta}$，这个是直接转化。紧接着，由定义，${B(4,3)}$ 是不能允许任意三点shatter的，所以由\alpha和β构成的所有三点组合也不能shatter（alpha经过去重），即 ${\alpha+β \leq B(3,3)}$ 。

![Estimating Part of B(4, 3) (1/2)](http://ofqm89vhw.bkt.clouddn.com/153b4e55f9bbb2ca6b01f575f0024150.png)

另一方面，由于 ${\alpha}$ 中 ${x_4}$ 是成对存在的，且 ${\alpha}$ 是不能被任意三点shatter的，则能推导出 ${\alpha}$ 是不能被任意两点 shatter的。这是因为，如果 ${\alpha}$ 是不能被任意两点shatter，而 ${x_4}$ 又是成对存在的，那么 ${x_1, x_2, x_3, x_4}$组成的 ${\alpha}$ 必然能被三个点shatter。这就违背了条件的设定。这个地方的推导非常巧妙，也解释了为什么会这样分组。此处得到的结论是：

$${\alpha \leq B(3,2)}$$

由此得出B(4,3)与B(3,x)的关系为：

![Putting It All Together](http://ofqm89vhw.bkt.clouddn.com/56cef07f24ceadec8a4bb47eff30bf98.png)

最后，推导出一般公式为：

![Putting It All Together](http://ofqm89vhw.bkt.clouddn.com/be89a031d30f8c58222e4a0e1b583e33.png)

根据递推公式，推导出 ${B(N,K)}$ 满足下列不等式：

![Bounding Function: The Theorem](http://ofqm89vhw.bkt.clouddn.com/2e4c6c38f10ae4bcd8472bbbc275317f.png)

上述不等式的右边是最高阶为 ${k-1}$ 的 ${N}$ 多项式，也就是说成长函数 ${m_{H}(N)}$ 的上界 ${B(N,K)}$ 的上界满足多项式分布 ${poly(N)}$，这就是我们想要得到的结果。

得到了 ${m_{H}(N)}$ 的上界 ${B(N,K)}$ 的上界满足多项式分布 ${poly(N)}$ 后，我们回过头来看看之前介绍的几种类型它们的 ${m_{H}(N)}$ 与break point的关系：

![The Three Break Points](http://ofqm89vhw.bkt.clouddn.com/6af894d021c610aba9398458428dc2f6.png)

我们得到的结论是，对于2D perceptrons，break point为 ${k=4}$，${m_{H}(N)}$ 的上界是 ${Nk−1}$ 。推广一下，也就是说，如果能找到一个模型的break point，且是有限大的，那么就能推断出其成长函数 ${m_{H}(N)}$ 有界。

## A Pictorial Proof

我们已经知道了成长函数的上界是 ${poly(N)}$ 的，下一步，如果能将 ${m_{H}(N)}$ 代替 ${M}$，代入到Hoffding不等式中，就能得到 ${E_{out} \approx E_{in}}$ 的结论，实际上并不是简单的替换就可以了，正确的表达式为：

![BAD Bound for General H](http://ofqm89vhw.bkt.clouddn.com/f0d5cdb6f03a3b9205b837acf5893b26.png)

该推导的证明比较复杂，我们可以简单概括为三个步骤来证明：

![Step 1: Replace ${E_{out}}$ by ${E_{in}'}$](http://ofqm89vhw.bkt.clouddn.com/3220cb2fd3f8a59aae36e8317644b9fd.png)

![Step 2: Decompose ${H}$ by Kind](http://ofqm89vhw.bkt.clouddn.com/e3738557e54ff499a15d5cfb6c1d4c3c.png)

![Step 3: Use Hoeffding without Replacement](http://ofqm89vhw.bkt.clouddn.com/8e9e19eb748792aeddb1c85221fa1340.png)

最终，我们通过引入成长函数 ${m_H}$，得到了一个新的不等式，称为Vapnik-Chervonenkis(VC) bound：

![That's All!](http://ofqm89vhw.bkt.clouddn.com/c1af28d92d977be33488fff9a601fe28.png)

对于2D perceptrons，它的break point是4，那么成长函数 ${m_{H}(N) =O(N^3)}$ 。所以，我们可以说2D perceptrons是可以进行机器学习的，只要找到hypothesis能让 ${E_{in} \approx 0}$，就能保证 ${E_{in} \approx E_{out}}$ 。

## 总结

本节课我们主要介绍了只要存在break point，那么其成长函数 ${m_{H}(N)}$ 就满足 ${poly(N)}$ 。推导过程是先引入 ${m_{H}(N)}$ 的上界 ${B(N,K)}$，${B(N,K)}$ 的上界是 ${N}$ 的 ${k-1}$ 阶多项式，从而得到 ${m_{H}(N)}$ 的上界就是 ${N}$ 的 ${k-1}$ 阶多项式。然后，我们通过简单的三步证明，将 ${m_{H}(N)}$ 代入了Hoffding不等式中，推导出了Vapnik-Chervonenkis(VC) bound，最终证明了只要break point存在，那么机器学习就是可行的。

## 参考

1. [台湾大学林轩田机器学习基石课程学习笔记6 -- Theory of Generalization](http://blog.csdn.net/red_stone1/article/details/71122928)