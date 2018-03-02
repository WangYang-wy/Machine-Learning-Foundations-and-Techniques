# Lecture 02 - Dual Support Vector Machine

上节课我们主要介绍了线性支持向量机（Linear Support Vector Machine）。Linear SVM的目标是找出最“胖”的分割线进行正负类的分离，方法是使用二次规划来求出分类线。本节课将从另一个方面入手，研究对偶支持向量机（Dual Support Vector Machine），尝试从新的角度计算得出分类线，推广SVM的应用范围。

## Motivation of Dual SVM

首先，我们回顾一下，对于非线性SVM，我们通常可以使用非线性变换将变量从 ${x}$ 域转换到 ${z}$ 域中。然后，在 ${z}$ 域中，根据上一节课的内容，使用线性SVM解决问题即可。上一节课我们说过，使用SVM得到large-margin，减少了有效的VC Dimension，限制了模型复杂度；另一方面，使用特征转换，目的是让模型更复杂，减小 ${E_{in}}$。所以说，非线性SVM是把这两者目的结合起来，平衡这两者的关系。那么，特征转换下，求解QP问题在 ${z}$ 域中的维度设为 ${\hat{d}+1}$ ，如果模型越复杂，则 ${\hat{d}+1}$ 越大，相应求解这个QP问题也变得很困难。当 ${\hat{d}}$ 无限大的时候，问题将会变得难以求解，那么有没有什么办法可以解决这个问题呢？一种方法就是使SVM的求解过程不依赖 ${\hat{d}}$ ，这就是我们本节课所要讨论的主要内容。

![Non-Linear Support Vector Machine Revisited](http://ofqm89vhw.bkt.clouddn.com/ae11fc9ddc1ccdb31e530f805464569c.png)

比较一下，我们上一节课所讲的Original SVM二次规划问题的变量个数是 ${\hat{d}+1}$ ，有 ${N}$ 个限制条件；而本节课，我们把问题转化为对偶问题（'Equivalent' SVM），同样是二次规划，只不过变量个数变成 ${N}$ 个，有 ${N+1}$ 个限制条件。这种对偶SVM的好处就是问题只跟 ${N}$ 有关，与 ${\hat{d}}$ 无关，这样就不存在上文提到的当 ${\hat{d}}$ 无限大时难以求解的情况。

![Todo: SVM 'without' ${\hat{d}}$](http://ofqm89vhw.bkt.clouddn.com/ab70373485b5a3cb021ed2178175ab5e.png)

如何把问题转化为对偶问题（'Equivalent' SVM），其中的数学推导非常复杂，本文不做详细数学论证，但是会从概念和原理上进行简单的推导。

还记得我们在《机器学习基石》课程中介绍的Regularization中，在最小化 ${E_{in}}$ 的过程中，也添加了限制条件：${w^{T}w \leq C}$。我们的求解方法是引入拉格朗日因子${\lambda}$，将有条件的最小化问题转换为无条件的最小化问题：${min E_{aug}(w)=E_{in}(w)+\lambda N w^{T}w}$，最终得到的 ${w}$ 的最优化解为：

$${\nabla E_{in}(w)+ \frac{2\lambda}{N} w=0}$$

所以，在regularization问题中，${\lambda}$ 是已知常量，求解过程变得容易。那么，对于dual SVM问题，同样可以引入 ${\lambda}$ ，将条件问题转换为非条件问题，只不过 ${\lambda}$ 是未知参数，且个数是 ${N}$ ，需要对其进行求解。

![Key Tool: Lagrange Multipliers](http://ofqm89vhw.bkt.clouddn.com/34edab312915ba25e30a8609e108b520.png)

如何将条件问题转换为非条件问题？上一节课我们介绍的SVM中，目标是：${min \frac{1}{2}w^{T}w}$，条件是：${y_n(w^{T}z_n+b) \geq 1}$ , ${for \ n=1,2,\cdots,N}$。首先，我们令拉格朗日因子为 ${\alpha_n}$ （区别于regularization），构造一个函数：

$${L(b,w,\alpha)=\frac{1}{2}w^{T}w+\sum_{n=1}^{N}\alpha_n(1-y_n(w^{T}z_n+b))}$$

这个函数右边第一项是SVM的目标，第二项是SVM的条件和拉格朗日因子 ${\alpha_n}$ 的乘积。我们把这个函数称为拉格朗日函数，其中包含三个参数：${b}$ ，${w}$，${\alpha_n}$。

![Starting Point: Constrained to 'Unconstrained'](http://ofqm89vhw.bkt.clouddn.com/a23822410186f3491851ba287414dc97.png)

下面，我们利用拉格朗日函数，把SVM构造成一个非条件问题：

![Claim](http://ofqm89vhw.bkt.clouddn.com/ec7712512d966f2c8c1461fd23fc2840.png)

该最小化问题中包含了最大化问题，怎么解释呢？首先我们规定拉格朗日因子 ${\alpha_n \geq 0}$ ，根据SVM的限定条件可得：${(1-y_n(w^{T}z_n+b)) \leq 0}$，如果没有达到最优解，即有不满足 ${(1-y_n(w^{T}z_n+b)) \leq 0}$ 的情况，因为 ${\alpha_n \geq 0}$ ，那么必然有 ${\sum_n\alpha_n(1-y_n(w^{T}z_n+b)) \geq 0}$ 。对于这种大于零的情况，其最大值是无解的。如果对于所有的点，均满足 ${(1-y_n(w^{T}z_n+b)) \leq 0}$ ，那么必然有 ${\sum_n\alpha_n(1-y_n(w^{T}z_n+b)) \leq 0}$，则当 ${\sum_n\alpha_n(1-y_n(w^{T}z_n+b))=0}$ 时，其有最大值，最大值就是我们SVM的目标：${\frac{1}{2}w^{T}w}$。因此，这种转化为非条件的SVM构造函数的形式是可行的。

## Lagrange Dual SVM

现在，我们已经将SVM问题转化为与拉格朗日因子 ${\alpha_n}$ 有关的最大最小值形式。已知${\alpha_n \geq 0}$ ，那么对于任何固定的 ${\alpha'}$ ，且 ${\alpha'_n \geq 0}$ ，一定有如下不等式成立：

![Lagrange Dual Problem](http://ofqm89vhw.bkt.clouddn.com/2ac411ce13565ea7cb2bf4b9e7560ac4.png)

对上述不等式右边取最大值，不等式同样成立：

![Lagrange Dual Problem](http://ofqm89vhw.bkt.clouddn.com/88afecfb0e02a557567d8985f7c3d185.png)

上述不等式表明，我们对SVM的 ${min}$ 和 ${max}$ 做了对调，满足这样的关系，这叫做Lagrange dual problem。不等式右边是SVM问题的下界，我们接下来的目的就是求出这个下界。

已知 ${\geq}$ 是一种弱对偶关系，在二次规划QP问题中，如果满足以下三个条件：

- 函数是凸的（convex primal）。
- 函数有解（feasible primal）。
- 条件是线性的（linear constraints）。

那么，上述不等式关系就变成强对偶关系，${\geq}$ 变成 ${=}$，即一定存在满足条件的解${(b,w,\alpha)}$，使等式左边和右边都成立，SVM的解就转化为右边的形式。

经过推导，SVM对偶问题的解已经转化为无条件形式：

![Solving Lagrange Dual: Simplifications](http://ofqm89vhw.bkt.clouddn.com/9fd0bb255900901246344d1482b8ca9f.png)

其中，上式括号里面的是对拉格朗日函数${L(b,w,\alpha)}$ 计算最小值。那么根据梯度下降算法思想：最小值位置满足梯度为零。首先，令${L(b,w,\alpha)}$对参数 ${b}$ 的梯度为零：

$${\frac{\partial L(b,w,\alpha)}{\partial b}=0=-\sum_{n=1}^{N} \alpha_n y_n}$$

也就是说，最优解一定满足 ${\sum_{n=1}^{N} \alpha_ny_n=0}$。那么，我们把这个条件代入计算 ${max}$ 条件中（与${\alpha_n \geq 0}$ 同为条件），并进行化简。

这样，SVM表达式消去了 ${b}$ ，问题化简了一些。然后，再根据最小值思想，令 ${L(b,w,\alpha)}$ 对参数 ${w}$ 的梯度为零：

$${\frac{\partial L(b,w,\alpha)}{ \partial w} = 0= w - \sum_{n=1}^{N} \alpha_n y_n z_n}$$

即得到：

$${w = \sum_{n= 1}^{N} \alpha_n y_nz_n}$$

也就是说，最优解一定满足 ${w=\sum_{n=1}^{N}\alpha_n y_nz_n}$。那么，同样我们把这个条件代入并进行化简：

![Solving Lagrange Dual: Simplifications](http://ofqm89vhw.bkt.clouddn.com/226b2b2d453b30d80a9e5811081aa836.png)

这样，SVM表达式消去了 ${w}$ ，问题更加简化了。这时候的条件有 ${3}$ 个：

- all ${\alpha_n \geq 0}$。
- ${\sum_{n=1}^{N} \alpha_n y_n = 0}$。
- ${w=\sum_{n=1}^{N}\alpha_n y_n z_n}$。

SVM简化为只有\alpha_n的最佳化问题，即计算满足上述三个条件下，函数 ${-\frac{1}{2}|| \sum_{n=1}^{N} \alpha_ny_nz_n||^2+ \sum_{n=1}^{N} \alpha_n}$ 最小值时对应的 ${\alpha_n}$ 是多少。

总结一下，SVM最佳化形式转化为只与 ${\alpha_n}$ 有关：

![KKT Optimality Conditions](http://ofqm89vhw.bkt.clouddn.com/bdc1e5dd1e97c18333d9530d21778970.png)

其中，满足最佳化的条件称之为Karush-Kuhn-Tucker(KKT)：

![KKT Optimality Conditions](http://ofqm89vhw.bkt.clouddn.com/d6093668e63b2f9fdf21c6494497ced8.png)

在下一部分中，我们将利用KKT条件来计算最优化问题中的${\alpha}$，进而得到${b}$和${w}$。

## Solving Dual SVM

上面我们已经得到了dual SVM的简化版了，接下来，我们继续对它进行一些优化。首先，将 ${max}$ 问题转化为${min}$ 问题，再做一些条件整理和推导，得到：

![Dual Formulation of Support Vector Machine](http://ofqm89vhw.bkt.clouddn.com/cbd2c1b1ce7e65d9ebc6b55a6b662861.png)

显然，这是一个convex的QP问题，且有 ${N}$ 个变量 ${\alpha_n}$ ，限制条件有 ${N+1}$ 个。则根据上一节课讲的QP解法，找到${Q}$，${p}$，${A}$，${c}$对应的值，用软件工具包进行求解即可。

![Dual SVM with QP Solver](http://ofqm89vhw.bkt.clouddn.com/5f4f4e441b4a2b6852636d5bbc0ab037.png)

求解过程很清晰，但是值得注意的是，${q_{n,m} = y_n  y_m z_n^T z_m}$ ，大部分值是非零的，称为dense。当 ${N}$ 很大的时候，例如 ${N=30000}$ ，那么对应的QD的计算量将会很大，存储空间也很大。所以一般情况下，对dual SVM问题的矩阵QD，需要使用一些特殊的方法，这部分内容就不再赘述了。

![Dual SVM with Special QP Solver](http://ofqm89vhw.bkt.clouddn.com/fa0208525a85582169a8a5b829121d96.png)

得到 ${\alpha_n}$ 之后，再根据之前的KKT条件，就可以计算出 ${w}$ 和 ${b}$ 了。首先利用条件 ${w=\sum \alpha_n y_n z_n}$ 得到 ${w}$ ，然后利用条件${\alpha_n(1-y_n(w^{T}z_n+b))=0}$ ，取任一 ${\alpha_n \neq 0}$ 即 ${\alpha_n> 0}$ 的点，得到${1-y_n(w^{T}z_n+b)=0}$，进而求得 ${b=y_n - w^{T}z_n}$。

![Optimal](http://ofqm89vhw.bkt.clouddn.com/f2f46e983718de2429a630288b6f2dcf.png)

值得注意的是，计算 ${b}$ 值，${\alpha_n>0}$ 时，有${y_n(w^{T}z_n+b)=1}$ 成立。${y_n(w^{T}z_n+b)=1}$ 正好表示的是该点在SVM分类线上，即fat boundary。也就是说，满足 ${\alpha_n > 0}$ 的点一定落在fat boundary上，这些点就是Support Vector。这是一个非常有趣的特性。

## Messages behind Dual SVM

回忆一下，上一节课中，我们把位于分类线边界上的点称为support vector（candidates）。本节课前面介绍了 ${\alpha_n>0}$ 的点一定落在分类线边界上，这些点称之为support vector（注意没有candidates）。也就是说分类线上的点不一定都是支持向量，但是满足 ${\alpha_n>0}$ 的点，一定是支持向量。

SV只由 ${\alpha_n>0}$ 的点决定，根据上一部分推导的 ${w}$ 和 ${b}$ 的计算公式，我们发现，${w}$ 和${b}$ 仅由SV即 ${\alpha_n>0}$ 的点决定，简化了计算量。这跟我们上一节课介绍的分类线只由“胖”边界上的点所决定是一个道理。也就是说，样本点可以分成两类：一类是support vectors，通过support vectors可以求得fattest hyperplane；另一类不是support vectors，对我们求得fattest hyperplane没有影响。

![Support Vectors Revisited](http://ofqm89vhw.bkt.clouddn.com/09a2567ad0a1428a24a70e50480702ed.png)

回过头来，我们来比较一下SVM和PLA的 ${w}$ 公式：

![Representation of Fattest Hyperplane
](http://ofqm89vhw.bkt.clouddn.com/c554928e30e1d6235bdf567140151273.png)

我们发现，二者在形式上是相似的。${w_{SVM}}$ 由fattest hyperplane边界上所有的SV决定，${w_{PLA}}$ 由所有当前分类错误的点决定。${w_{SVM}}$ 和 ${w_{PLA}}$ 都是原始数据点 ${y_nz_n}$ 的线性组合形式，是原始数据的代表。

总结一下，本节课和上节课主要介绍了两种形式的SVM，一种是Primal Hard-Margin SVM，另一种是Dual Hard_Margin SVM。Primal Hard-Margin SVM有 ${\hat{d}+1}$ 个参数，有 ${N}$ 个限制条件。当 ${\hat{d}+1}$ 很大时，求解困难。而Dual Hard_Margin SVM有 ${N}$ 个参数，有 ${N+1}$ 个限制条件。当数据量 ${N}$ 很大时，也同样会增大计算难度。两种形式都能得到w和b，求得fattest hyperplane。通常情况下，如果 ${N}$ 不是很大，一般使用Dual SVM来解决问题。

![Summary: Two Forms of Hard-Margin SVM](http://ofqm89vhw.bkt.clouddn.com/341db30c90dd493017cf26fd1f329a47.png)

这节课提出的Dual SVM的目的是为了避免计算过程中对 ${\hat{d}}$ 的依赖，而只与 ${N}$ 有关。但是，Dual SVM是否真的消除了对 ${\hat{d}}$ 的依赖呢？其实并没有。因为在计算 ${q_{n,m}=y_n y_m z_n^T z_m}$ 的过程中，由 ${z}$ 向量引入了 ${\hat{d}}$，实际上复杂度已经隐藏在计算过程中了。所以，我们的目标并没有实现。下一节课我们将继续研究探讨如何消除对 ${\hat{d}}$ 的依赖。

![Are We Done Yet?](http://ofqm89vhw.bkt.clouddn.com/91fa2c42856ba1dcddf52beed286c07e.png)

## 总结

本节课主要介绍了SVM的另一种形式：Dual SVM。我们这样做的出发点是为了移除计算过程对 ${\hat{d}}$ 的依赖。Dual SVM的推导过程是通过引入拉格朗日因子${\alpha}$ ，将SVM转化为新的非条件形式。然后，利用QP，得到最佳解的拉格朗日因子 ${\alpha}$。再通过KKT条件，计算得到对应的 ${w}$ 和 ${b}$。最终求得fattest hyperplane。下一节课，我们将解决Dual SVM计算过程中对 ${\hat{d}}$ 的依赖问题。

## 参考

1. [台湾大学林轩田机器学习技法课程学习笔记2 -- Dual Support Vector Machine](http://blog.csdn.net/red_stone1/article/details/73822768)