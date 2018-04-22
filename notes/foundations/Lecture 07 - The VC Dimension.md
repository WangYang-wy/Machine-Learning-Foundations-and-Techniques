# Lecture 07 - The VC Dimension

前几节课着重介绍了机器能够学习的条件并做了详细的推导和解释。机器能够学习必须满足两个条件：

- 假设空间 ${H}$ 的 Size ${M}$ 是有限的，即当 ${N}$ 足够大的时候，那么对于假设空间中任意一个假设 ${g}$，${E_{out} \approx  E_{in}}$。
- 利用算法 ${A}$ 从假设空间 ${H}$ 中，挑选一个 ${g}$ ，使 ${E_{in} (g) \approx 0}$，则 ${E_{out} \approx 0}$。

这两个条件，正好对应着 test 和 trian 两个过程。train 的目的是使损失期望 ${E_{in} (g) \approx 0}$；test 的目的是使将算法用到新的样本时的损失期望也尽可能小，即 ${E_{out} \approx 0}$。

正因为如此，上次课引入了 break point，并推导出只要 break point 存在，则 ${M}$ 有上界，一定存在 ${E_{out} \approx E_{in}}$ 。

本次课程主要介绍 VC Dimension 的概念。同时也是总结 VC Dimension与 ${E_{in} (g) \approx 0}$，${E_{out}  \approx 0}$，Model Complexity Penalty（下面会讲到）的关系。

## Definition of VC Dimension

首先，我们知道如果一个假设空间 ${H}$ 有 break point ${k}$，那么它的成长函数是有界的，它的上界称为 Bound function。根据数学归纳法，Bound function 也是有界的，且上界为 ${N^{k-1}}$。从下面的表格可以看出，${N^{k-1}}$比 ${B(N,k)}$ 松弛很多。

![Recap: More on Growth Function](http://ofqm89vhw.bkt.clouddn.com/adb7b648b2603ecdacb7b072e994d021.png)

则根据上一节课的推导，VC bound就可以转换为：

![Recap: More on Vapnik-Chervonenkis (VC) Bound](http://ofqm89vhw.bkt.clouddn.com/0a818434e495ef33f3e1f3b18da73d16.png)

这样，不等式只与 ${k}$ 和 ${N}$ 相关了，一般情况下样本 ${N}$ 足够大，所以我们只考虑 ${k}$ 值。有如下结论：

- 若假设空间 ${H}$ 有 break point ${k}$，且${N}$ 足够大，则根据 VC bound 理论，算法有良好的泛化能力。
- 在假设空间中选择一个 ${g}$，使 ${E_{in} \approx 0}$，则其在全集数据中的错误率会较低。

![Conclusions](http://ofqm89vhw.bkt.clouddn.com/c0ee48b44ced41f5b5a2d44a5e39a575.png)

下面介绍一个新的名词：`VC Dimension`。VC Dimension 就是某假设集 ${H}$ 能够 shatter 的最多 inputs 的个数，即最大完全正确的分类能力。（注意，只要存在一种分布的 inputs 能够正确分类也满足）。

> shatter 的英文意思是“粉碎”，也就是说：对于inputs的所有情况都能列举出来。例如对 ${N}$ 个输入，如果能够将 ${2^N}$ 种情况都列出来，则称该 ${N}$ 个输入能够被假设集 ${H}$ shatter。

根据之前 break point 的定义：假设集不能被 shatter 任何分布类型的 inputs 的最少个数。则 VC Dimension 等于 break point 的个数减一。

![VC Dimension](http://ofqm89vhw.bkt.clouddn.com/e08529dc629989a6df769eb9e81a582e.png)

现在，我们回顾一下之前介绍的四种例子，它们对应的 VC Dimension 是多少：

![The Four VC Dimensions](http://ofqm89vhw.bkt.clouddn.com/87978a480a703080c6cbcaa697cdd646.png)

用 ${d_{vc}}$ 代替 ${k}$ ，那么 VC bound 的问题也就转换为与 ${d_{vc}}$ 和 ${N}$ 相关了。同时，如果一个假设集 ${H}$ 的 ${d_{vc}}$ 确定了，则就能满足机器能够学习的第一个条件 ${E_{out}  \approx  E_{in}}$，与算法、样本数据分布和目标函数都没有关系。

![VC Dimension and Learning](http://ofqm89vhw.bkt.clouddn.com/58154e105e9f0a8874c12589d8008343.png)

## VC Dimension of Perceptrons

回顾一下我们之前介绍的 ${2D}$ 下的 ${PLA}$ 算法，已知 Perceptrons 的 ${k=4}$，即 ${d_{vc} = 3}$。根据 VC Bound 理论，当 ${N}$ 足够大的时候， ${E_{out}(g) \approx  E_{in}(g)}$ 。如果找到一个${g}$，使 ${E_{in}(g) \approx 0}$，那么就能证明 ${PLA}$ 是可以学习的。

![2D PLA Revisited](http://ofqm89vhw.bkt.clouddn.com/0b872acafa482d72c862913a988b276e.png)

这是在 ${2D}$ 情况下，那如果是多维的 Perceptron，它对应的 ${d_{vc}}$ 又等于多少呢？

已知在 ${1D}$ Perceptron， ${d_{vc} =2}$，在${2D}$ Perceptrons，${d_{vc} =3}$，那么我们有如下假设：${d_{vc} = d+1}$，其中 ${d}$ 为维数。

要证明的话，只需分两步证明：

- ${d_{vc} \leq d+1}$。
- ${d_{vc} \geq d+1}$。

![VC Dimension of Perceptrons](http://ofqm89vhw.bkt.clouddn.com/286dd82773f2dd5e928d385b78ad7667.png)

首先证明第一个不等式：${d_{vc} \geq d+1}$。

在 ${d}$ 维里，我们只要找到某一类的 ${d+1}$ 个 inputs 可以被 shatter 的话，那么必然得到 ${d_{vc}  \geq d+1}$。所以，我们有意构造一个 ${d}$ 维的矩阵 ${X}$ 能够被 shatter 就行。${X}$ 是 ${d}$ 维的，有 ${d+1}$ 个inputs，每个 inputs 加上第零个维度的常数项 ${1}$ ，得到 ${X}$ 的矩阵：

![${d_{vc} \geq d+1}$](http://ofqm89vhw.bkt.clouddn.com/728f528fffd081ad66d653f2ea3f916f.png)

矩阵中，每一行代表一个 inputs，每个 inputs 是 ${d+1}$ 维的，共有 ${d+1}$ 个 inputs。这里构造的 ${X}$ 很明显是可逆的。shatter 的本质是假设空间 ${H}$ 对 ${X}$ 的所有情况的判断都是对的，即总能找到权重 ${W}$ ，满足 ${X \cdot W=y}$ ， ${W=X^{-1} \cdot y}$ 。由于这里我们构造的矩阵 ${X}$ 的逆矩阵存在，那么 ${d}$ 维的所有 inputs 都能被 shatter，也就证明了第一个不等式。

![Can We Shatter X?](http://ofqm89vhw.bkt.clouddn.com/2a05bb61341fd1592243a1c2f2d9e93a.png)

然后证明第二个不等式： ${d_{vc} \leq d+1}$。

在 ${d}$ 维里，如果对于任何的 ${d+2}$ 个inputs，一定不能被shatter，则不等式成立。我们构造一个任意的矩阵${X}$，其包含${d+2}$个inputs，该矩阵有${d+1}$列，${d+2}$行。这${d+2}$ 个向量的某一列一定可以被另外 ${d+1}$个向量线性表示，例如对于向量 ${X_{d+2}}$，可表示为：

$${X_{d+2} = a_1 \cdot X_1 + a_2 \cdot X_2+ \cdots +a_{d+1} \cdot X_{d+1}}$$

其中，假设 ${a1>0, a_2,\cdots ,a_{d+1} < 0}$。

那么如果 ${X_1}$ 是正类，${X2, \cdots ,X_{d+1}}$均为负类，则存在 ${W}$ ，得到如下表达式：

![d-D General Case](http://ofqm89vhw.bkt.clouddn.com/b8ee2702bc3c7c42285dcbc55fe80ab1.png)

因为其中蓝色项大于 ${0}$，代表正类；红色项小于 ${0}$，代表负类。所有对于这种情况，${X_{d+2}}$一定是正类，无法得到负类的情况。也就是说，${d+2}$个 inputs 无法被 shatter。证明完毕！

综上证明可得 ${d_{vc} =d+1}$。

## Physical Intuition VC Dimension

![Degrees of Freedom](http://ofqm89vhw.bkt.clouddn.com/85e47670536adc42cd838e41db63dee3.png)

上节公式中 ${W}$ 又名 features，即自由度。自由度是可以任意调节的，如同上图中的旋钮一样，可以调节。VC Dimension 代表了假设空间的分类能力，即反映了 ${H}$ 的自由度，产生 dichotomy 的数量，也就等于 features 的个数，但也不是绝对的。

![practical rule of thumb](http://ofqm89vhw.bkt.clouddn.com/1e7609aa4cbff27f1547222c7ea65156.png)

例如，对 ${2D}$ Perceptrons，线性分类，${d_{vc} =3}$，则 ${W={w_0,w_1,w_2}}$，也就是说只要 ${3}$ 个 features 就可以进行学习，自由度为 ${3}$。

介绍到这，我们发现 ${M}$ 与 ${d_{vc}}$ 是成正比的，从而得到如下结论：

![${M}$ and ${d_{VC}}$](http://ofqm89vhw.bkt.clouddn.com/976207bf25cbd9ffdefa6eb66d09b1db.png)

## Interpreting VC Dimension

下面，我们将更深入地探讨 VC Dimension 的意义。首先，把 VC Bound 重新写到这里：

![VC Bound Rephrase: Penalty for Model Complexity](http://ofqm89vhw.bkt.clouddn.com/1dbbb71f2f35745ebdc6e3827aa9dc38.png)

根据之前的泛化不等式，如果 ${|E_{in} - E_{out}| > \epsilon}$，即出现 bad 坏的情况的概率最大不超过${\delta}$。那么反过来，对于 good 好的情况发生的概率最小为 ${1 - \delta}$ ，则对上述不等式进行重新推导：

![VC Bound Rephrase: Penalty for Model Complexity](http://ofqm89vhw.bkt.clouddn.com/5ba6a0862aa90e5db63ee25c1a5f55a3.png)

${\epsilon}$ 表现了假设空间 ${H}$ 的泛化能力，${\epsilon}$ 越小，泛化能力越大。

至此，已经推导出泛化误差 ${E_{out}}$ 的边界，因为我们更关心其上界（${E_{out}}$ 可能的最大值）。

上述不等式的右边第二项称为模型复杂度，其模型复杂度与样本数量 ${N}$、假设空间 ${H(d_{vc})}$ 、${\epsilon}$ 有关。${E_{out}}$ 由 ${E_{in}}$ 共同决定。下面绘出 ${E_{out}}$、model complexity、 ${E_{in}}$ 随 ${d_{vc}}$ 变化的关系：

![THE VC Message](http://ofqm89vhw.bkt.clouddn.com/b60ff55523e587a73e5cff1ec7b8c242.png)

通过该图可以得出如下结论：

- ${d_{vc}}$ 越大，${E_{in}}$ 越小，${\Omega}$ 越大（复杂）。
- ${d_{vc}}$ 越小，${E_{in}}$ 越大，${\Omega}$ 越小（简单）。

随着 ${d_{vc}}$ 增大，${E_{out}}$ 会先减小再增大。

所以，为了得到最小的 ${E_{out}}$，不能一味地增大 ${d_{vc}}$ 以减小 ${E_{in}}$ ，因为 ${E_{in}}$ 太小的时候，模型复杂度会增加，造成 ${E_{out}}$ 变大。也就是说，选择合适的 ${d_{vc}}$ ，选择的 features 个数要合适。

下面介绍一个概念：样本复杂度（Sample Complexity）。如果选定 ${d_{vc}}$，样本数据 ${D}$ 选择多少合适呢？通过下面一个例子可以帮助我们理解：

![A Example](http://ofqm89vhw.bkt.clouddn.com/3dbd900033323858b052abe943b0391a.png)

通过计算得到 ${N=29300}$ ，刚好满足 ${\delta=0.1}$ 的条件。${N}$ 大约是 ${d_{vc}}$ 的 ${10000}$ 倍。这个数值太大了，实际中往往不需要这么多的样本数量，大概只需要 ${d_{vc}}$ 的 ${10}$ 倍就够了。${N}$ 的理论值之所以这么大是因为 VC Bound 过于宽松了，我们得到的是一个比实际大得多的上界。

![Looseness of VC Bound](http://ofqm89vhw.bkt.clouddn.com/378b927610c3bdc226f1f83caddb16b7.png)

值得一提的是，VC Bound 是比较宽松的，而如何收紧它却不是那么容易，这也是机器学习的一大难题。但是，令人欣慰的一点是，VC Bound 基本上对所有模型的宽松程度是基本一致的，所以，不同模型之间还是可以横向比较。从而，VC Bound 宽松对机器学习的可行性还是没有太大影响。

## 总结

本节课主要介绍了 VC Dimension 的概念就是最大的 non-break point。然后，我们得到了 Perceptrons 在 ${d}$ 维度下的 VC Dimension是 ${d+1}$ 。接着，我们在物理意义上，将 ${d_{vc}}$ 与自由度联系起来。最终得出结论 ${d_{vc}}$ 不能过大也不能过小。选取合适的值，才能让 ${E_{out}}$ 足够小，使假设空间 ${H}$ 具有良好的泛化能力。

## 参考

1. [台湾大学林轩田机器学习基石课程学习笔记7 -- The VC Dimension](http://blog.csdn.net/red_stone1/article/details/71191232)