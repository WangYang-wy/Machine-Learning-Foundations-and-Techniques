# Lecture 09 - Linear Regression

上节课，我们主要介绍了在有 noise 的情况下，VC Bound 理论仍然是成立的。同时，介绍了不同的 error measure 方法。本节课介绍机器学习最常见的一种算法：Linear Regression.

## 线性回归问题

在之前的 Linear Classification 课程中，讲了信用卡发放的例子，利用机器学习来决定是否给用户发放信用卡。本节课仍然引入信用卡的例子，来解决给用户发放信用卡额度的问题，这就是一个线性回归（Linear Regression）问题。

![Linear Regression Hypothesis](http://ofqm89vhw.bkt.clouddn.com/a6e3ea07eee27ceb13a6ede9f2f7d535.png)

令用户特征集为 ${d}$ 维的 ${X}$，加上常数项，维度为 ${d+1}$，与权重 ${w}$ 的线性组合即为 ${Hypothesis}$ ,记为 ${h(x)}$ 。线性回归的预测函数取值在整个实数空间，这跟线性分类不同。

$${h(x) = w^T X}$$

![Illustration of Linear Regression](http://ofqm89vhw.bkt.clouddn.com/7075837049f02952375e67983bc037cb.png)

根据上图，在一维或者多维空间里，线性回归的目标是找到一条直线（对应一维）、一个平面（对应二维）或者更高维的超平面，使样本集中的点更接近它，也就是残留误差 Residuals 最小化。

一般最常用的错误测量方式是基于最小二乘法，其目标是计算误差的最小平方和对应的权重 ${w}$，即上节课介绍的squared error：

![The Error Measure](http://ofqm89vhw.bkt.clouddn.com/ed26ac9cb24047f2a60367cd6a270f4a.png)

这里提一点，最小二乘法可以解决线性问题和非线性问题。线性最小二乘法的解是 closed-form，即 ${X=(A^TA)^{-1}A^Ty}$，而非线性最小二乘法没有 closed-form，通常用迭代法求解。本节课的解就是 closed-form 的。

## 线性回归算法

样本数据误差 ${E_{in}}$ 是权重 ${w}$ 的函数，因为 ${X}$ 和 ${y}$ 都是已知的。我们的目标就是找出合适的 ${w}$，使 ${E_{in}}$ 能够最小。那么如何计算呢？

首先，运用矩阵转换的思想，将 ${E_{in}}$ 计算转换为矩阵的形式。

![Matrix Form](http://ofqm89vhw.bkt.clouddn.com/5663350c5b9e5e51c44fe7f603b971cb.png)

然后，对于此类线性回归问题，${E_{in}(w)}$ 一般是个凸函数。凸函数的话，我们只要找到一阶导数等于零的位置，就找到了最优解。那么，我们将 ${E_w}$ 对每个 ${w_i,i=0,1,\cdots ,d}$ 求偏导，偏导为零的 ${w_i}$，即为最优化的权重值分布。

![GradientGradient](http://ofqm89vhw.bkt.clouddn.com/11eb7e45074d345ec7511a335053666b.png)

根据梯度的思想，对 ${E_w}$ 进行矩阵话求偏导处理：

![The Gradient](http://ofqm89vhw.bkt.clouddn.com/9c485e0032da25d6cebfb92c4812bcdb.png)

令偏导为零，最终可以计算出权重向量 ${w}$ 为：

![Optimal Linear Regression Weights](http://ofqm89vhw.bkt.clouddn.com/8ab8aec29a81622d1fe8c2c55f5ca5b9.png)

最终，我们推导得到了权重向量 ${w=(X^TX)^{-1}X^Ty}$，这是上文提到的 closed-form 解。其中， ${(X^TX)^{-1}X^T}$ 又称为伪逆矩阵 pseudo-inverse，记为 ${X^+}$，维度是 ${(d+1) \times N}$ 。

但是，我们注意到，伪逆矩阵中有逆矩阵的计算，逆矩阵 ${(X^TX)^{-1}}$ 是否一定存在？一般情况下，只要满足样本数量 ${N}$ 远大于样本特征维度 ${d+1}$，就能保证矩阵的逆是存在的，称之为非奇异矩阵。但是如果是奇异矩阵，不可逆怎么办呢？其实，大部分的计算逆矩阵的软件程序，都可以处理这个问题，也会计算出一个逆矩阵。所以，一般伪逆矩阵是可解的。

## 泛化问题

现在，可能有这样一个疑问，就是这种求解权重向量的方法是机器学习吗？或者说这种方法满足我们之前推导 VC Bound，即是否泛化能力强 ${E_{in} \approx E_{out}}$？

![Is Linear Regression a 'Learning Algorithm'?](http://ofqm89vhw.bkt.clouddn.com/aeff146c0da89587e090cbd11215707a.png)

有两种观点：

1. 这不属于机器学习范畴。因为这种 closed-form 解的形式跟一般的机器学习算法不一样，而且在计算最小化误差的过程中没有用到迭代。
1. 这属于机器学习范畴。因为从结果上看，${E_{in}}$和 ${E_{out}}$ 都实现了最小化，而且实际上在计算逆矩阵的过程中，也用到了迭代。

其实，只从结果来看，这种方法的确实现了机器学习的目的。下面通过介绍一种更简单的方法，证明 linear regression 问题是可以通过线下最小二乘法方法计算得到好的 ${E_{in}}$ 和 ${E_{out}}$ 的。

![Benefit of Analytic Solution](http://ofqm89vhw.bkt.clouddn.com/0db45ba2b187c703c0591a01d42e7ffc.png)

首先，我们根据平均误差的思想，把 ${E_{in}(w_{LIN})}$ 写成如图的形式，经过变换得到:

$${Ein(w_{LIN})= \frac{1}{N}||(I-XX^+)y||^2 = \frac{1}{N}||(I-H)y||^2} $$

我们称 ${XX^{+}}$ 为帽子矩阵，用 ${H}$ 表示。

下面从几何图形的角度来介绍帽子矩阵 ${H}$ 的物理意义。

![Geometric View of Hat Matrix](http://ofqm89vhw.bkt.clouddn.com/42a86d1e646904defd2522a7a2365933.png)

图中，${y}$ 是 ${N}$ 维空间的一个向量，粉色区域表示输入矩阵 ${X}$ 乘以不同权值向量 ${w}$ 所构成的空间，根据所有 ${w}$ 的取值，预测输出都被限定在粉色的空间中。向量 ${\hat{y}}$ 就是粉色空间中的一个向量，代表预测的一种。 ${y}$ 是实际样本数据输出值。

机器学习的目的是在粉色空间中找到一个 ${\hat{y}}$，使它最接近真实的 ${y}$ ，那么我们只要将 ${y}$ 在粉色空间上作垂直投影即可，投影得到的 ${\hat{y}}$ 即为在粉色空间内最接近 ${y}$ 的向量。这样即使平均误差 ${E}$ 最小。

从图中可以看出，${\hat{y}}$ 是 ${y}$ 的投影，已知 ${\hat{y}=Hy}$，那么 ${H}$ 表示的就是将 ${y}$ 投影到 ${\hat{y}}$ 的一种操作。图中绿色的箭头 ${y-\hat{y}}$ 是向量 ${y}$ 与 ${\hat{y}}$ 相减，${y-\hat{y}}$ 垂直于粉色区域。已知 ${(I-H)y = y - \hat{y}}$ 那么 ${I-H}$ 表示的就是将 ${y}$ 投影到 ${y-\hat{y}}$ 即垂直于粉色区域的一种操作。这样的话，我们就赋予了 ${H}$ 和 ${I-H}$ 不同但又有联系的物理意义。

这里 ${trace(I-H)}$ 称为 ${I-H}$ 的迹，值为 ${N-(d+1)}$ 。这条性质很重要，一个矩阵的 ${trace}$ 等于该矩阵的所有特征值(Eigenvalues) 之和。下面给出简单证明：

- ${trace(I - H)}$
- ${= trace(I) - trace(H)}$
- ${= N - trace(XX^+)= N - trace(X(X^TX)^{-1}X^T}$
- ${= N - trace(X^TX(X^TX)^{-1}) = N - trace(I_d+1)}$
- ${= N-(d+1)}$

介绍下该 ${I-H}$ 这种转换的物理意义：原来有一个有 ${N}$ 个自由度的向量 ${y}$ ，投影到一个有 ${d+1}$ 维的空间 ${x}$ （代表一列的自由度，即单一输入样本的参数，如图中粉色区域），而余数剩余的自由度最大只有 ${N-(d+1)}$ 种。

在存在 noise 的情况下，上图变为：

![An Illustrative 'Proof'](http://ofqm89vhw.bkt.clouddn.com/98c4c7fa7e8a80ba83053c656f63caef.png)

图中，粉色空间的红色箭头是目标函数 ${f(x)}$，虚线箭头是 noise，可见，真实样本输出 ${y}$ 由 ${f(x)}$ 和 noise 相加得到。由上面推导，已知向量 ${y}$ 经过 ${I-H}$ 转换为 ${y-\hat{y}}$ ，而 noise 与 ${y}$ 是线性变换关系，那么根据线性函数知识，我们推导出noise 经过 ${I-H}$ 也能转换为 ${y-\hat{y}}$ 。则对于样本平均误差，有下列推导成立：

$${Ein(w_{LIN}) = \frac{1}{N}||y-\hat{y}||^2 = \frac{1}{N}||(I-H)noise||^2 = \frac{1}{N}(N-(d+1))||noise||^2}$$

即

$${E_{in} = noiselevel \times (1 -  \frac{d+1}{N})}$$

同样，对${E_{out}}$有如下结论：

$${E_{out} = noiselevel \times (1+ \frac{d+1}{N})}$$

这个证明有点复杂，但是我们可以这样理解： ${E_{in}}$与 ${E_{out}}$形式上只差了 ${(d+1)N}$ 项，从哲学上来说， ${E_{in}}$是我们看得到的样本的平均误差，如果有 noise，我们把预测往 noise 那边偏一点，让 ${E_{in}}$好看一点点，所以减去 ${(d+1)N}$ 项。那么同时，新的样本 ${E_{out}}$是我们看不到的，如果noise在反方向，那么 ${E_{out}}$就应该加上 ${(d+1)N}$ 项。

我们把 ${E_{in}}$与 ${E_{out}}$画出来，得到学习曲线：

![The Learning Curve](http://ofqm89vhw.bkt.clouddn.com/e78417a1a1594c8cfb71b25e31db069d.png)

当 ${N}$ 足够大时， ${E_{in}}$与 ${E_{out}}$逐渐接近，满足 ${E_{in} \approx E_{out}}$，且数值保持在 noise level。这就类似 VC 理论，证明了当 ${N}$ 足够大的时候，这种线性最小二乘法是可以进行机器学习的，算法有效！

## Linear Regression for Binary Classification

之前介绍的 Linear Classification 问题使用的 Error Measure 方法用的是 0/1 error，那么 Linear Regression 的 squared error 是否能够应用到 Linear Classification 问题？

![Linear Classification vs. Linear Regression](http://ofqm89vhw.bkt.clouddn.com/fce2347511e1a14f28d97dffa83033c8.png)

下图展示了两种错误的关系，一般情况下，squared error 曲线在 0/1 error 曲线之上。即 ${err_{0/1} \leq err_{sqr}}$ 。

![Relation of Two Errors](http://ofqm89vhw.bkt.clouddn.com/4b1b4add84fc5e37d44c7255260c0a6a.png)

根据之前的 VC 理论，${E_{out}}$ 的上界满足：

![Linear Regression for Binary Classification](http://ofqm89vhw.bkt.clouddn.com/301c15cbd60aa1a907bf91799279a3ac.png)

从图中可以看出，用 ${err_{sqr}}$ 代替 ${err_{0/1}}$，${E_{out}}$ 仍然有上界，只不过是上界变得宽松了。也就是说用线性回归方法仍然可以解决线性分类问题，效果不会太差。二元分类问题得到了一个更宽松的上界，但是也是一种更有效率的求解方式。

## 总结

本节课，我们主要介绍了 Linear Regression。首先，我们从问题出发，想要找到一条直线拟合实际数据值；然后，我们利用最小二乘法，用解析形式推导了权重 ${w}$ 的 closed-form 解；接着，用图形的形式得到 ${E_{out} - E_{in} \approx \ \frac{2(N+1)}{N}}$，证明了 linear regression 是可以进行机器学习的；最后，我们证明 linear regressin这种方法可以用在 binary classification 上，虽然上界变宽松了，但是仍然能得到不错的学习方法。

## 参考

1. [台湾大学林轩田机器学习基石课程学习笔记9 -- Linear Regression](http://blog.csdn.net/red_stone1/article/details/71599034)
1. [最小二乘法和梯度下降法的一些总结](http://blog.csdn.net/red_stone1/article/details/70306403)