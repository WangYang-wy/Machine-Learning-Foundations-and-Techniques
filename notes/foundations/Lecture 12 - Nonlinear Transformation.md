# Lecture 12 - Nonlinear Transformation

上一节课，我们介绍了分类问题的三种线性模型，可以用来解决 binary classification 和 multiclass classification 问题。本节课主要介绍非线性的模型来解决分类问题。

## Quadratic Hypothesis

之前介绍的线性模型，在 ${2D}$ 平面上是一条直线，在 ${3D}$ 空间中是一个平面。数学上，我们用线性得分函数 ${s}$ 来表示：${s = w^Tx}$。其中，${x}$ 为特征值向量，${w}$ 为权重，${s}$ 是线性的。

线性模型的优点就是，它的 VC Dimension 比较小，保证了 ${E_{in} \approx E_{out}}$ 。但是缺点也很明显，对某些非线性问题，可能会造成 ${E_{in}}$ 很大，虽然 ${E_{in} \approx E_{out}}$ ，但是也造成 ${E_{out}}$ 很大，分类效果不佳。

为了解决线性模型的缺点，我们可以使用非线性模型来进行分类。例如数据集 ${D}$ 不是线性可分的，而是圆形可分的，圆形内部是正类，外面是负类。假设它的 hypotheses 可以写成：

$${h_{SEP}(x) = sign(- x_{1}^{2} - x_{2}^{2}+ 0.6)}$$

基于这种非线性思想，我们之前讨论的 PLA、Regression 问题都可以有非线性的形式进行求解。

下面介绍如何设计这些非线性模型的演算法。还是上面介绍的平面圆形分类例子，它的 ${h(x)}$ 的权重 ${w_0=0.6,\ w_1=-1,\ w_2=-1}$ ，但是 ${h(x)}$ 的特征不是线性模型的 ${(1,\ x_1,\ x_2)}$ ，而是 ${(1,\ x^2_1,\ x^2_2)}$ 。我们令 ${z_0=1,\ z_1=x^2_1,\ z_2=x^2_2}$ ，那么， ${h(x)}$ 变成：

$${h(x)=sign(\hat{w_0} \cdot z_0+ \hat{w_1} \cdot z_1 + \hat{w_2} \cdot z_2) = sign(0.6 \cdot z_0 - 1 \cdot z_1 - 1 \cdot z_2)=sign({\hat{w}}^T z)}$$

这种 ${x_n \rightarrow z_n}$ 的转换可以看成是 ${x}$ 空间的点映射到 ${z}$ 空间中去，而在 ${z}$ 域中，可以用一条直线进行分类，也就是从 ${x}$ 空间的圆形可分映射到 ${z}$ 空间的线性可分。${z}$ 域中的直线对应于 ${x}$ 域中的圆形。因此，我们把 ${x_n \rightarrow z_n}$ 这个过程称之为特征转换（Feature Transform）。通过这种特征转换，可以将非线性模型转换为另一个域中的线性模型。

已知 ${x}$ 域中圆形可分在 ${z}$ 域中是线性可分的，那么反过来，如果在 ${z}$ 域中线性可分，是否在 ${x}$ 域中一定是圆形可分的呢？答案是否定的。由于权重向量w取值不同， ${x}$ 域中的hypothesis可能是圆形、椭圆、双曲线等等多种情况。

目前讨论的 ${x}$ 域中的圆形都是圆心过原点的，对于圆心不过原点的一般情况， ${x_n \rightarrow z_n}$ 映射公式包含的所有项为：

$${\Phi_2(x)=(1,\ x_1,\ x_2,\ x^2_1,\ x_1x_2,\ x^2_2)}$$

也就是说，对于二次 ${hypothesis}$ ，它包含二次项、一次项和常数项 ${1}$ ， ${z}$ 域中每一条线对应 ${x}$ 域中的某二次曲线的分类方式，也许是圆，也许是椭圆，也许是双曲线等等。那么 ${z}$ 域中的hypothesis可以写成：

![z域中的hypothesis可以写成](http://ofqm89vhw.bkt.clouddn.com/a0300ec43be7c2ce952574250377de30.png)

## Nonlinear Transform

上一部分我们定义了什么了二次hypothesis，那么这部分将介绍如何设计一个好的二次hypothesis来达到良好的分类效果。那么目标就是在 ${z}$ 域中设计一个最佳的分类线。

![Good Quadratic Hypothesis](http://ofqm89vhw.bkt.clouddn.com/40c06bff60466cef2ba531e60ab1788c.png)

其实，做法很简单，利用映射变换的思想，通过映射关系，把 ${x}$ 域中的最高阶二次的多项式转换为 ${z}$ 域中的一次向量，也就是从 quardratic hypothesis 转换成了 perceptrons 问题。用 ${z}$ 值代替 ${x}$ 多项式，其中向量 ${z}$ 的个数与 ${x}$ 域中 ${x}$ 多项式的个数一致（包含常数项）。这样就可以在 ${z}$ 域中利用线性分类模型进行分类训练。训练好的线性模型之后，再将 ${z}$ 替换为 ${x}$ 的多项式就可以了。具体过程如下：

![The Nonlinear Transform Steps](http://ofqm89vhw.bkt.clouddn.com/026b2f07ca25d125bdd83b985367d0f0.png)

整个过程就是通过映射关系，换个空间去做线性分类，重点包括两个：

- 特征转换。
- 训练线性模型。

其实，我们以前处理机器学习问题的时候，已经做过类似的特征变换了。比如数字识别问题，我们从原始的像素值特征转换为一些实际的 concrete 特征，比如密度、对称性等等，这也用到了 feature transform 的思想。

## Price of Nonlinear Transform

若 ${x}$ 特征维度是 ${d}$ 维的，也就是包含 ${d}$ 个特征，那么二次多项式个数，即 ${z}$ 域特征维度是：

$${\hat{d} = 1 + C_d^1 + C_d^2 + d = d(d+3)2+1}$$

如果 ${x}$ 特征维度是 ${2}$ 维的，即 ${(x_1,\ x_2)}$ ，那么它的二次多项式为 ${(1,\ x_1,\ x_2,\ x^2_1,\ x_1x_2,x^2_2)}$ ，有 ${6}$ 个。

现在，如果阶数更高，假设阶数为 ${Q}$ ，那么对于 ${x}$ 特征维度是 ${d}$ 维的，它的 ${z}$ 域特征维度为：

$${\hat{d}=C_Q^Q + d = C_Q^d + d = O(Q^d)}$$

由上式可以看出，计算 ${z}$ 域特征维度个数的时间复杂度是 ${Q}$ 的 ${d}$ 次方，随着 ${Q}$ 和 ${d}$ 的增大，计算量会变得很大。同时，空间复杂度也大。也就是说，这种特征变换的一个代价是计算的时间、空间复杂度都比较大。

另一方面，${z}$域中特征个数随着 ${Q}$ 和 ${d}$ 增加变得很大，同时权重 ${w}$ 也会增大，即自由度增加，VC Dimension 增大。令 ${z}$ 域中的特征维度是 ${1+\hat{d}}$，则在在域中，任何 ${\hat{d}+2}$ 的输入都不能被 shattered；同样，在 ${x}$ 域中，任何 ${\hat{d}+2}$ 的输入也不能被 shattered。${\hat{d}+1}$ 是 VC Dimension 的上界，如果 ${\hat{d}+1}$ 很大的时候，相应的 VC Dimension 就会很大。根据之前章节课程的讨论，VC Dimension 过大，模型的泛化能力会比较差。

![Model Complexity Price](http://ofqm89vhw.bkt.clouddn.com/777c7fa2d356f7756aa3703f927ec6e1.png)

下面通过一个例子来解释为什么 VC Dimension 过大，会造成不好的分类效果：

![Generalization Issue](http://ofqm89vhw.bkt.clouddn.com/1ff3fe1549525003c58e7231c9d27abb.png)

上图中，左边是用直线进行线性分类，有部分点分类错误；右边是用四次曲线进行非线性分类，所有点都分类正确，那么哪一个分类效果好呢？单从平面上这些训练数据来看，四次曲线的分类效果更好，但是四次曲线模型很容易带来过拟合的问题，虽然它的 ${E_{in}}$ 比较小，从泛化能力上来说，还是左边的分类器更好一些。也就是说VC Dimension过大会带来过拟合问题，${\hat{d}+1}$不能太大了。

那么如何选择合适的 ${Q}$，来保证不会出现过拟合问题，使模型的泛化能力强呢？一般情况下，为了尽量减少特征自由度，我们会根据训练样本的分布情况，人为地减少、省略一些项。但是，这种人为地删减特征会带来一些“自我分析”代价，虽然对训练样本分类效果好，但是对训练样本外的样本，不一定效果好。所以，一般情况下，还是要保存所有的多项式特征，避免对训练样本的人为选择。

## Structured Hypothesis Sets

下面，我们讨论一下从 ${x}$ 域到 ${z}$ 域的多项式变换。首先，如果特征维度只有 ${1}$ 维的话，那么变换多项式只有常数项：

$${\Phi_0(x)=(1)}$$

如果特征维度是两维的，变换多项式包含了一维的${\Phi_0(x)}$：

$${\Phi_1(x)=(\Phi_0(x),x_1,x_2,\ldots,x_d)}$$

如果特征维度是三维的，变换多项式包含了二维的${\Phi_1(x)}$：

$${\Phi_x(x) = (\Phi_1(x), x^2_1,x_1 x_2,\ldots,x^2_d)}$$

以此类推，如果特征维度是 ${Q}$ 次，那么它的变换多项式为：

$${\Phi_Q(x)=(\Phi_{Q-1}(x),x^Q_1,x^{Q-1}_1 x_2,\ldots,x^Q_d)}$$

那么对于不同阶次构成的 hypothesis 有如下关系：

$${H\Phi_0 \subset H \Phi_1 \subset  H\Phi_2\subset \cdots  \subset H\Phi_Q}$$

我们把这种结构叫做 Structured Hypothesis Sets ：

![Polynomial Transform Revisited](http://ofqm89vhw.bkt.clouddn.com/dece707eff218becb7992edf00b475a6.png)

那么对于这种 Structured Hypothesis Sets，它们的 VC Dimension 满足下列关系：

$${d_{VC}(H_0) \leq d_{VC}(H_1) \leq d_{VC}(H_2) \leq \ldots \leq d_{VC}(H_Q)}$$

它的 ${E_{in}}$ 满足下列关系：

$${E_{in}(g_0) \geq E_{in}(g_1) \geq E_{in}(g_2) \geq E_{in}(g_Q)}$$

![Structured Hypothesis Sets](http://ofqm89vhw.bkt.clouddn.com/6db4ca9e6b99c32a4e3665708514d971.png)

从上图中也可以看到，随着变换多项式的阶数增大，虽然 ${E_{in}}$ 逐渐减小，但是 model complexity 会逐渐增大，造成 ${E_{out}}$ 很大，所以阶数不能太高。

![Linear Model First](http://ofqm89vhw.bkt.clouddn.com/49eaa007d42464f85a57a801569a6ea4.png)

那么，如果选择的阶数很大，确实能使 ${E_{in}}$ 接近于${0}$，但是泛化能力通常很差，我们把这种情况叫做  tempting sin。所以，一般最合适的做法是先从低阶开始，如先选择一阶 hypothesis，看看 ${E_{in}}$ 是否很小，如果 ${E_{in}}$ 足够小的话就选择一阶，如果 ${E_{in}}$ 大的话，再逐渐增加阶数，直到满足要求为止。也就是说，尽量选择低阶的 hypothes，这样才能得到较强的泛化能力。

## 总结

这节课主要介绍了非线性分类模型，通过非线性变换，将非线性模型映射到另一个空间，转换为线性模型，再来进行线性分类。本节课完整介绍了非线性变换的整体流程，以及非线性变换可能会带来的一些问题：时间复杂度和空间复杂度的增加。最后介绍了在要付出代价的情况下，使用非线性变换的最安全的做法，尽可能使用简单的模型，而不是模型越复杂越好。

## 参考

1. [台湾大学林轩田机器学习基石课程学习笔记12 -- Nonlinear Transformation](http://blog.csdn.net/red_stone1/article/details/72630003)