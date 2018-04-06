# Lecture 11 - Linear Models for Classification

上一节课，我们介绍了 Logistic Regression 问题，建立 cross-entropy error，并提出使用梯度下降算法 gradient descnt 来获得最好的 logistic hypothesis。本节课继续介绍使用线性模型来解决分类问题。

## Linear Models for Binary Classification

之前介绍几种线性模型都有一个共同点，就是都有样本特征 ${x}$ 的加权运算，我们引入一个线性得分函数 ${s}$：${s = w^T x}$。

![Linear Models Revisited](http://ofqm89vhw.bkt.clouddn.com/90397359abd8a8803e1ad4307df465ce.png)

三种线性模型，第一种是 linear classification。线性分类模型的hypothesis为 ${h(x)=sign(s)}$ ,取值范围为 ${\{-1,+1\}}$ 两个值，它的 err 是 ${0/1}$ 的，所以对应的 ${E_{in}(w)}$ 是离散的，并不好解，这是个 NP-hard 问题。第二种是 linear regression。线性回归模型的 hypothesis 为 ${h(x) = s}$ ，取值范围为整个实数空间，它的 err 是 squared 的，所以对应的 ${E_{in}(w)}$ 是开口向上的二次曲线，其解是 closed-form 的，直接用线性最小二乘法求解即可。第三种是 logistic regression。逻辑回归模型的 hypothesis 为 ${h(x)=\theta(s)}$ ，取值范围为 ${(-1,1)}$ 之间，它的 err 是 cross-entropy 的，所有对应的 ${E_{in}(w)}$ 是平滑的凸函数，可以使用梯度下降算法求最小值。

从上图中，我们发现，linear regression 和 logistic regression 的 error function 都有最小解。那么可不可以用这两种方法来求解 linear classification 问题呢？下面，我们来对这三种模型的 error function 进行分析，看看它们之间有什么联系。

![Error Functions Revisited](http://ofqm89vhw.bkt.clouddn.com/10c7f4479afae33c223c4fcf34f2958a.png)

对于 linear classification，它的 error function 可以写成：

$${err_{0/1}(s,y) = |sign(s) \neq y|=|sign(ys)\neq 1|}$$

对于 linear regression，它的 error function 可以写成：

$${err_{SQR}(s,y) = (s - y)^2 = (ys - 1)^2}$$

对于 logistic regression，它的 error function 可以写成：

$${err_{CE}(s,y) = \ln(1 + \exp(-ys))}$$

上述三种模型的 error function 都引入了 ${ys}$ 变量，那么 ${ys}$ 的物理意义是什么？${ys}$ 就是指分类的正确率得分，其值越大越好，得分越高。

下面，我们用图形化的方式来解释三种模型的 error function 到底有什么关系：

![Visualizing Error Functions](http://ofqm89vhw.bkt.clouddn.com/7997c4e413126d95486770d61782e6d3.png)

从上图中可以看出，${ys}$ 是横坐标轴，${err_{0/1}}$ 是呈阶梯状的，在 ${ys>0}$ 时，${err_{0/1}}$ 恒取最小值 ${0}$。${err_{SQR}}$ 呈抛物线形式，在 ${ys=1}$ 时，取得最小值，且在 ${ys=1}$ 左右很小区域内，${err_{0/1}}$ 和 ${err_{SQR}}$ 近似。${err_{CE}}$ 是呈指数下降的单调函数，${ys}$ 越大，其值越小。同样在 ${ys=1}$ 左右很小区域内， ${err_{0/1}}$ 和 ${err_{CE}}$ 近似。但是我们发现 ${err_{CE}}$ 并不是始终在 ${err_{0/1}}$ 之上，所以为了计算讨论方便，我们把 ${err_{CE}}$ 做幅值上的调整，引入 ${err_{SCE} = log_{2}^{(1+exp(-ys))} = \frac{1}{ln2} err_{CE}}$ ，这样能保证 ${err_{SCE}}$ 始终在 ${err_{0/1}}$ 上面，如下图所示：

由上图可以看出：

$${err_{0/1}(s,y) \leq err_{SCE}(s,y)= \frac{1}{ln_{2}^{err_{SCE(s,y)}}}}$$

$${E_{in}^{0/1}(w) \leq err_{ES} E_{in}(w) = \frac{1}{ln2} err_{EC} E_{in}(w)}$$

$${E_{out}^{0/1}(w) \leq E_{out}^{SCE}(w)= \frac{1}{ln2} E_{out}^{CE}(w)}$$

那么由VC理论可以知道：

从 0/1 出发：

$${E_{out}^{0/1}(w) \leq E_{in}^{0/1}(w) +\Omega^{0/1} \leq \frac{1}{ln2} E_{in}^{CE}(w) + \Omega^{0/1}}$$

从 CE 出发：

$${E_{out}^{0/1}(w) \leq \frac{1}{\ln 2}ECEout(w)\leq \frac{1}{\ln 2}EC E_{in}(w) +\frac{1}{\ln 2}\Omega^{CE}}$$

通过上面的分析，我们看到err 0/1是被限定在一个上界中。这个上界是由logistic regression模型的error function决定的。而linear regression其实也是linear classification的一个upper bound，只是随着sy偏离1的位置越来越远，linear regression的error function偏差越来越大。综上所述，linear regression和logistic regression都可以用来解决linear classification的问题。

![Theoretical Implication of Upper Bound](http://ofqm89vhw.bkt.clouddn.com/a956bc44bcce341ff2040176c5c5a3ef.png)

下图列举了 PLA、linear regression、logistic regression模型用来解 linear classification 问题的优点和缺点。通常，我们使用 linear regression 来获得初始化的 ${w_0}$，再用 logistic regression 模型进行最优化解。

![Regression for Classification](http://ofqm89vhw.bkt.clouddn.com/82f8e617699a2a9b462266df8ed68985.png)

## Stochastic Gradient Descent

之前介绍的 PLA 算法和 logistic regression 算法，都是用到了迭代操作。PLA 每次迭代只会更新一个点，它每次迭代的时间复杂度是 ${O(1)}$ ；而 logistic regression 每次迭代要对所有 ${N}$ 个点都进行计算，它的每时间复杂度是 ${O(N)}$ 。为了提高 logistic regression 中 gradient descent 算法的速度，可以使用另一种算法：随机梯度下降算法(Stochastic Gradient Descent)。

随机梯度下降算法每次迭代只找到一个点，计算该点的梯度，作为我们下一步更新w的依据。这样就保证了每次迭代的计算量大大减小，我们可以把整体的梯度看成这个随机过程的一个期望值。

![Logistic Regression Revisited](http://ofqm89vhw.bkt.clouddn.com/fb10331642fcff0a00a373046a0fdaa0.png)

随机梯度下降可以看成是真实的梯度加上均值为零的随机噪声方向。单次迭代看，好像会对每一步找到正确梯度方向有影响，但是整体期望值上看，与真实梯度的方向没有差太多，同样能找到最小值位置。随机梯度下降的优点是减少计算量，提高运算速度，而且便于 online 学习；缺点是不够稳定，每次迭代并不能保证按照正确的方向前进，而且达到最小值需要迭代的次数比梯度下降算法一般要多。

对于 logistic regression 的 SGD，它的表达式为：

$${w_{t+1} \leftarrow w_t +\eta \ \theta(-y_n w^T tx_n)(y_n x_n)}$$

我们发现，SGD 与 PLA 的迭代公式有类似的地方，如下图所示：

我们把 SGD logistic regression称之为 'soft' PLA，因为 PLA 只对分类错误的点进行修正，而 SGD logistic regression 每次迭代都会进行或多或少的修正。另外，当 ${\eta=1}$ ，且 ${w_t^T x_n}$ 足够大的时候，PLA 近似等于 SGD。

![PLA Revisited](http://ofqm89vhw.bkt.clouddn.com/eaf5edb4fd8c0af7bf20d861abbd3628.png)

除此之外，还有两点需要说明：

1. SGD的终止迭代条件。没有统一的终止条件，一般让迭代次数足够多；
1. 学习速率 ${\eta}$。${\eta}$的取值是根据实际情况来定的，一般取值 ${0.1}$ 就可以了。

## Multiclass via Logistic Regression

之前我们一直讲的都是二分类问题，本节主要介绍多分类问题，通过 linear classification 来解决。假设平面上有四个类，分别是正方形、菱形、三角形和星形，如何进行分类模型的训练呢？

首先我们可以想到这样一个办法，就是先把正方形作为正类，其他三种形状都是负类，即把它当成一个二分类问题，通过 linear classification 模型进行训练，得出平面上某个图形是不是正方形，且只有 ${\{-1,+1\}}$两种情况。然后再分别以菱形、三角形、星形为正类，进行二元分类。这样进行四次二分类之后，就完成了这个多分类问题。

但是，这样的二分类会带来一些问题，因为我们只用 ${\{-1, +1\}}$ 两个值来标记，那么平面上某些可能某些区域都被上述四次二分类模型判断为负类，即不属于四类中的任何一类；也可能会出现某些区域同时被两个类甚至多个类同时判断为正类，比如某个区域又判定为正方形又判定为菱形。那么对于这种情况，我们就无法进行多类别的准确判断，所以对于多类别，简单的 binary classification 不能解决问题。

针对这种问题，我们可以使用另外一种方法来解决：soft 软性分类，即不用 ${\{-1, +1\}}$ 这种 binary classification，而是使用 logistic regression，计算某点属于某类的概率、可能性，去概率最大的值为那一类就好。

![Multiclass Prediction: Combine Soft Classifiers](http://ofqm89vhw.bkt.clouddn.com/30e7dba12b35822d4cbf667818a97578.png)

soft classification 的处理过程和之前类似，同样是分别令某类为正，其他三类为负，不同的是得到的是概率值，而不是 ${\{-1, +1\}}$ 。最后得到某点分别属于四类的概率，取最大概率对应的哪一个类别就好。效果如下图所示：

这种多分类的处理方式，我们称之为 One-Versus-All(OVA) Decomposition。这种方法的优点是简单高效，可以使用 logistic regression 模型来解决；缺点是如果数据类别很多时，那么每次二分类问题中，正类和负类的数量差别就很大，数据不平衡 unbalanced，这样会影响分类效果。但是，OVA 还是非常常用的一种多分类算法。

![One-Versus-All (OVA) Decomposition](http://ofqm89vhw.bkt.clouddn.com/7ec681f12a058ea8b76ca22f315c5ad5.png)

## Multiclass via Binary Classification

上一节，我们介绍了多分类算法 OVA，但是这种方法存在一个问题，就是当类别k很多的时候，造成正负类数据 unbalanced，会影响分类效果，表现不好。现在，我们介绍另一种方法来解决当k很大时，OVA 带来的问题。

这种方法呢，每次只取两类进行 binary classification，取值为 ${\{-1, +1\}}$ 。假如 ${k=4}$，那么总共需要进行 ${C_{2}^{4}=6}$ 次 binary classification。那么，六次分类之后，如果平面有个点，有三个分类器判断它是正方形，一个分类器判断是菱形，另外两个判断是三角形，那么取最多的那个，即判断它属于正方形，我们的分类就完成了。这种形式就如同 ${k}$ 个足球对进行单循环的比赛，每场比赛都有一个队赢，一个队输，赢了得 ${1}$ 分，输了得 ${0}$ 分。那么总共进行了 ${C_k^2}$次的比赛，最终取得分最高的那个队就可以了。

![One-versus-one (OVO) Decomposition](http://ofqm89vhw.bkt.clouddn.com/794997420789a63f67544abe6a72807b.png)

这种区别于 OVA 的多分类方法叫做 One-Versus-One(OVO)。这种方法的优点是更加高效，因为虽然需要进行的分类次数增加了，但是每次只需要进行两个类别的比较，也就是说单次分类的数量减少了。而且一般不会出现数据 unbalanced 的情况。缺点是需要分类的次数多，时间复杂度和空间复杂度可能都比较高。

## 总结

本节课主要介绍了分类问题的三种线性模型：linear classification、linear regression 和 logistic regression。首先介绍了这三种 linear models 都可以来做 binary classification。然后介绍了比梯度下降算法更加高效的 SGD 算法来进行 logistic regression 分析。最后讲解了两种多分类方法，一种是 OVA，另一种是 OVO。这两种方法各有优缺点，当类别数量 ${k}$ 不多的时候，建议选择 OVA，以减少分类次数。

## 参考

1. [台湾大学林轩田机器学习基石课程学习笔记11 -- Linear Models for Classification](http://blog.csdn.net/red_stone1/article/details/72453273)