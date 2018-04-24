# Lecture 10 - Logistic Regression

上一节课，我们介绍了 Linear Regression 线性回归，以及用平方错误来寻找最佳的权重向量 ${w}$，获得最好的线性预测。本节课将介绍 Logistic Regression 逻辑回归问题。

## Logistic Regression Problem

一个心脏病预测的问题：根据患者的年龄、血压、体重等信息，来预测患者是否会有心脏病。很明显这是一个二分类问题，其输出 ${y}$ 只有 ${\{-1,1\}}$ 两种情况。

二元分类，一般情况下，理想的目标函数 ${f(x) > 0.5}$ ，则判断为正类 ${+1}$；若 ${f(x) < 0.5}$ ，则判断为负类 ${-1}$ 。

![Heart Attack Prediction Problem (1/2)](http://ofqm89vhw.bkt.clouddn.com/29bf12343d7b9f97a61daf339d20e4cd.png)

但是，如果我们想知道的不是患者有没有心脏病，而是到底患者有多大的几率是心脏病。这表示，我们更关心的是目标函数的值（分布在 ${(0,1)}$之间），表示是正类的概率（正类表示是心脏病）。这跟我们原来讨论的二分类问题不太一样，我们把这个问题称为软性二分类问题（'soft' binary classification）。这个值越接近 ${1}$ ，表示正类的可能性越大；越接近 ${0}$，表示负类的可能性越大。

对于软性二分类问题，理想的数据是分布在 ${[0,1]}$ 之间的具体值，但是实际中的数据只可能是 ${0}$ 或者 ${1}$ ，我们可以把实际中的数据看成是理想数据加上了噪声的影响。

如果目标函数是 ${f(x)=P(+1|x) \in [0,1]}$ 的话，我们如何找到一个好的 ${Hypothesis}$ 跟这个目标函数很接近呢？

首先，根据我们之前的做法，对所有的特征值进行加权处理。计算的结果 ${s}$ ，我们称之为 'risk score' ：

![Soft Binary Classification](http://ofqm89vhw.bkt.clouddn.com/09d61e2b5df30fbb44940ef0695fa1df.png)

但是特征加权和 ${s \in (-\infty,\ +\infty)}$ ，如何将 ${s}$ 值限定在 ${[0,1]}$ 之间呢？一个方法是使用sigmoid Function，记为 ${\theta(s)}$。那么我们的目标就是找到一个hypothesis： ${h(x) = \theta(w^{T}x)}$ 。

Sigmoid Function函数记为 ${\theta(s)= \frac{1}{1+e^{-s}}}$ ，满足 ${\theta(-\infty)=0}$，${\theta(0)= \frac{1}{2}}$，${\theta(+\infty) = 1}$ 。这个函数是平滑的、单调的S型函数。则对于逻辑回归问题，hypothesis就是这样的形式：

$${h(x)= \frac{1}{1+e^{-w^{T}x}}}$$

那我们的目标就是求出这个预测函数 ${h(x)}$ ，使它接近目标函数 ${f(x)}$。

![Logistic Function](http://ofqm89vhw.bkt.clouddn.com/14fc6be2df72596e0032bf0074fb8d64.png)

## Logistic Regression Error

现在我们将 Logistic Regression 与之前讲的 Linear Classification、Linear Regression 做个比较：

![Three Linear Models](http://ofqm89vhw.bkt.clouddn.com/4a1c6265bde976244a87dad9a8965e3c.png)

这三个线性模型都会用到线性 scoring function ${s = w^{T}x}$ 。linear classification 的误差使用的是 0/1 err；linear regression 的误差使用的是squared err。那么 logistic regression 的误差该如何定义呢？

先介绍一下“似然性”的概念。目标函数 ${f(x) = P(+1|x)}$ ，如果我们找到了 hypothesis 很接近 target function。也就是说，在所有的 hypothesis 集合中找到一个 hypothesis 与 target function 最接近，能产生同样的数据集 ${D}$ ，包含 ${y}$ 输出 label，则称这个 hypothesis 是最大似然 likelihood。

logistic function: ${h(x) = \theta(w^{T}x)}$ 满足一个性质：${1 - h(x)=h(-x)}$ 。那么，似然性 ${h}$:

$${likelihood(h) = P(x_1)h(+x_1) \times P(x_2)h(-x_2)\times \cdots P(x_N)h(-x_N)}$$

![Likelihood](http://ofqm89vhw.bkt.clouddn.com/b42c8d566e30cce790c2efa5cbb6150d.png)

因为 ${P(x_n)}$ 对所有的 ${h}$ 来说，都是一样的，所以我们可以忽略它。那么我们可以得到 logistic ${h}$ 正比于所有的 ${h(y_nx)}$ 乘积。我们的目标就是让乘积值最大化。

![Likelihood of Logistic Hypothesis](http://ofqm89vhw.bkt.clouddn.com/9bd89be6343be6cd5b7977ea52fee5b6.png)

![Cross-Entropy Error](http://ofqm89vhw.bkt.clouddn.com/7ce3c2e2d1ba6af9c8c88ae671363068.png)

如果将 ${w}$ 代入的话：

![Cross-Entropy Error](http://ofqm89vhw.bkt.clouddn.com/2d64e61289c8453f8f8ad83a376abce3.png)

为了把连乘问题简化计算，我们可以引入${\ln}$ 操作，让连乘转化为连加：

![Cross-Entropy Error](http://ofqm89vhw.bkt.clouddn.com/ca1316e24daf82d7faf3c16fbaa90ffe.png)

接着，我们将 maximize 问题转化为 minimize 问题，添加一个负号就行，并引入平均数操作 ${\frac{1}{N}}$ ,将 logistic function 的表达式带入，那么 minimize 问题就会转化为如下形式：

![Cross-Entropy Error](http://ofqm89vhw.bkt.clouddn.com/3d868e1f7b8f39a32aeb418f03f6e82b.png)

至此，我们得到了 logistic regression 的 err function，称之为 cross-entropy error 交叉熵误差：

## Gradient of Logistic Regression Error

我们已经推导了 ${E_{in}}$ 的表达式，那接下来的问题就是如何找到合适的向量 ${w}$ ，让 ${E_{in}}$ 最小。

![Minimizing](http://ofqm89vhw.bkt.clouddn.com/95866c2052563d0de1ae5b8ea7a35925.png)

Logistic Regression 的 ${E_{in}}$ 是连续、可微、二次可微的凸曲线（开口向上），根据之前 Linear Regression 的思路，我们只要计算 ${E_{in}}$ 的梯度为零时的 ${w}$ ，即为最优解。

对 ${E_{in}}$ 计算梯度，学过微积分的都应该很容易计算出来，最终得到的梯度表达式为：

![Gradient](http://ofqm89vhw.bkt.clouddn.com/0fa4cfcf0f812ff8bc608a2e9e7d5502.png)

为了计算 ${E_{in}}$ 最小值，我们就要找到让${\nabla E_{in}(w)}$ 等于 ${0}$ 的位置。

上式可以看成 ${\theta(-y_n w^T x_n)}$ 是 ${-y_n x_n}$ 的线性加权。要求 ${\theta(-y_n w^T x_n)}$ 与 ${-y_n x_n}$ 的线性加权和为 ${0}$ ，那么一种情况是线性可分，如果所有的权重 ${\theta(-y_n w^T x_n)}$ 为 ${0}$ ，那就能保证 ${\nabla E_{in}(w) = 0}$。 ${\theta(-y_n w^T x_n)}$ 是 sigmoid function，根据其特性，只要让 ${- y_n w^T x_n \ll 0}$，即 ${y_n w^T x_n \gg 0}$。${y_n w^T x_n \ll 0}$ 表示对于所有的点，${y_{n}}$ 与 ${w^{T}x_{n}}$ 都是同号的，这表示数据集 ${D}$ 必须是全部线性可分的才能成立。

然而，保证所有的权重 ${\theta(-y_n w^T x_n)}$ 为 ${\theta}$ 是不太现实的，总有不等于 ${0}$ 的时候，那么另一种常见的情况是非线性可分，只能通过使加权和为零，来求解 ${w}$。这种情况没有 closed-form 解，与 Linear Regression 不同，只能用迭代方法求解。

之前所说的 Linear Regression 有 closed-form 解，可以说是“一步登天”的；但是 PLA 算法是一步一步修正迭代进行的，每次对错误点进行修正，不断更新 ${w}$ 值。PLA 的迭代优化过程表示如下：

![PLA Revisited: Iterative Optimization](http://ofqm89vhw.bkt.clouddn.com/920a9266f3b44d20142f94c3d492f6f3.png)

${w}$ 每次更新包含两个内容：一个是每次更新的方向 ${y_{n}x_{n}}$ ，用 ${v}$ 表示，另一个是每次更新的步长 ${\eta}$。参数 ${(v,\eta)}$ 和终止条件决定了我们的迭代优化算法。

![PLA Revisited: Iterative Optimization](http://ofqm89vhw.bkt.clouddn.com/6f9ba611f1ccd014430db6b43705641a.png)

## Gradient Descent

根据上一小节 PLA 的思想，迭代优化让每次 ${w}$ 都有更新：

![Iterative Optimization](http://ofqm89vhw.bkt.clouddn.com/7a392fb63f0f09fa29560ef35b3feaca.png)

我们把 ${E_{in}(w)}$ 曲线看做是一个山谷的话，要求 ${E_{in}(w)}$ 最小，即可比作下山的过程。整个下山过程由两个因素影响：一个是下山的单位方向 ${v}$ ；另外一个是下山的步长 ${\eta}$ 。

利用微分思想和线性近似，假设每次下山我们只前进一小步，即 ${\eta}$ 很小，那么根据泰勒 ${Taylor}$ 一阶展开，可以得到：

$${E_{in} (wt+ \eta v) \approx E_{in}(wt)+ \eta v^T\nabla E_{in} (wt)}$$

迭代的目的是让 ${E_{in}}$ 越来越小，即让 ${E_{in}(wt+ \eta v) < E_{in}(wt)}$ 。${\eta}$ 是标量，因为如果两个向量方向相反的话，那么他们的内积最小（为负），也就是说如果方向 ${v}$ 与梯度 ${\nabla E_{in} (wt)}$ 反向的话，那么就能保证每次迭代 ${E_{in}(wt+ \eta v)< E_{in}(wt)}$ 都成立。则，我们令下降方向 ${v}$ 为：

$${v= - \nabla E_{in} (wt)||\nabla E_{in} (wt)||}$$

${v}$ 是单位向量，${v}$ 每次都是沿着梯度的反方向走，这种方法称为梯度下降（gradient descent）算法。那么每次迭代公式就可以写成：

$${w_{t+1} \leftarrow w_t - \eta \nabla E_{in} (w_t)||\nabla E_{in} (w_t)||}$$

![Gradient Descent](http://ofqm89vhw.bkt.clouddn.com/cc4c8bcfc8071d571683dc9672e9fe9c.png)

下面讨论一下 ${\eta}$ 的大小对迭代优化的影响： ${\eta}$ 如果太小的话，那么下降的速度就会很慢； ${\eta}$ 如果太大的话，那么之前利用 ${Taylor}$ 展开的方法就不准了，造成下降很不稳定，甚至会上升。因此， ${\eta}$ 应该选择合适的值，一种方法是在梯度较小的时候，选择小的 ${\eta}$ ，梯度较大的时候，选择大的 ${\eta}$ ，即 ${\eta}$ 正比于 ${||\nabla E_{in} (wt)||}$ 。这样保证了能够快速、稳定地得到最小值 ${E_{in}(w)}$ 。

![Choice of ${\eta}$](http://ofqm89vhw.bkt.clouddn.com/41f7f34401c872381e981652dbec012e.png)

其中：

$${\eta' = \eta||\nabla E_{in}(wt)||}$$

总结一下基于梯度下降的Logistic Regression算法步骤如下：

初始化 ${w_0}$ 计算梯度 ${\nabla E_{in}(wt) = \frac{1}{N} \sum_{n=1}^{N} \theta(-y_n w_t^T x_n)(-y_n x_n)}$ 迭代跟新 ${w_{t+1} \leftarrow w_t - \eta \nabla E_{in}(w_t)}$ 满足 ${\nabla E_{in} (w_{t+1}) \approx 0}$ 或者达到迭代次数，迭代结束。

## 总结

我们今天介绍了 Logistic Regression。首先，从逻辑回归的问题出发，将 ${P(+1|x)}$ 作为目标函数，将 ${θ(w^{T}x)}$ 作为 hypothesis。接着，我们定义了 logistic regression 的 err function，称之为 cross-entropy error 交叉熵误差。然后，我们计算 logistic regression error 的梯度，最后，通过梯度下降算法，计算 ${\nabla E_{in}(w_t)\approx 0}$ 时对应的 ${w_t}$ 值。

## 参考

1. [台湾大学林轩田机器学习基石课程学习笔记10 -- Logistic Regression](http://blog.csdn.net/red_stone1/article/details/72229903)