# Lecture 04 - Soft-Margin Support Vector Machine

上节课我们主要介绍了Kernel SVM。先将特征转换和计算内积这两个步骤合并起来，简化计算、提高计算速度，再用Dual SVM的求解方法来解决。Kernel SVM不仅能解决简单的线性分类问题，也可以求解非常复杂甚至是无限多维的分类问题，关键在于核函数的选择，例如线性核函数、多项式核函数和高斯核函数等等。但是，我们之前讲的这些方法都是Hard-Margin SVM，即必须将所有的样本都分类正确才行。这往往需要更多更复杂的特征转换，甚至造成过拟合。本节课将介绍一种Soft-Margin SVM，目的是让分类错误的点越少越好，而不是必须将所有点分类正确，也就是允许有noise存在。这种做法很大程度上不会使模型过于复杂，不会造成过拟合，而且分类效果是令人满意的。

## Motivation and Primal Problem

上节课我们说明了一点，就是 SVM 同样可能会造成 overfit。原因有两个，一个是由于我们的 SVM 模型（即kernel）过于复杂，转换的维度太多，过于 powerful 了；另外一个是由于我们坚持要将所有的样本都分类正确，即不允许错误存在，造成模型过于复杂。如下图所示，左边的图 ${\Phi_1}$是线性的，虽然有几个点分类错误，但是大部分都能完全分开。右边的图 ${\Phi_4}$ 是四次多项式，所有点都分类正确了，但是模型比较复杂，可能造成过拟合。直观上来说，左边的图是更合理的模型。

![Cons of Hard-Margin SVM](http://ofqm89vhw.bkt.clouddn.com/ad6f9e761580f8e19b28aafa218cd8ea.png)

如何避免过拟合？方法是允许有分类错误的点，即把某些点当作是noise，放弃这些 noise 点，但是尽量让这些 noise 个数越少越好。回顾一下我们在《机器学习基石》笔记中介绍的 pocket 算法，pocket 的思想不是将所有点完全分开，而是找到一条分类线能让分类错误的点最少。而 Hard-Margin SVM 的目标是将所有点都完全分开，不允许有错误点存在。为了防止过拟合，我们可以借鉴 pocket 的思想，即允许有犯错误的点，目标是让这些点越少越好。

为了引入允许犯错误的点，我们将 Hard-Margin SVM 的目标和条件做一些结合和修正，转换为如下形式：

![Give Up on Some Examples](http://ofqm89vhw.bkt.clouddn.com/5d2e92a5b98fd1774c0202718dc93af7.png)

修正后的条件中，对于分类正确的点，仍需满足 ${y_n(w^Tz_n+b) \geq 1}$，而对于 noise 点，满足 ${y_n(w^Tz_n+b) \geq -\infty}$，即没有限制。修正后的目标除了 ${\frac{1}{2} w^Tw}$ 项，还添加了 ${y_n \neq sign(w^Tz_n+b)}$ ，即 noise 点的个数。参数C的引入是为了权衡目标第一项和第二项的关系，即权衡 large margin 和 noise tolerance 的关系。

我们再对上述的条件做修正，将两个条件合并，得到：

![Soft-Margin SVM](http://ofqm89vhw.bkt.clouddn.com/d26a38ce88b69cc077b450233e3bb966.png)

这个式子存在两个不足的地方。首先，最小化目标中第二项是非线性的，不满足 ${QP}$ 的条件，所以无法使用 dual 或者 kernel SVM 来计算。然后，对于犯错误的点，有的离边界很近，即 error 小，而有的离边界很远，error 很大，上式的条件和目标没有区分 small error 和 large error。这种分类效果是不完美的。

![Soft-Margin SVM](http://ofqm89vhw.bkt.clouddn.com/6dd709c0782188cccb184f03ce69ae26.png)

为了改正这些不足，我们继续做如下修正：

![Soft-Margin SVM](http://ofqm89vhw.bkt.clouddn.com/01d0bbc025a5538fa12267c69b6c3888.png)

修正后的表达式中，我们引入了新的参数 ${\xi_n}$ 来表示每个点犯错误的程度值，${\xi_n \geq 0}$。通过使用 error 值的大小代替是否有 error，让问题变得易于求解，满足 ${QP}$ 形式要求。这种方法类似于我们在机器学习基石笔记中介绍的 0/1 error 和 squared error。这种 soft-margin SVM 引入新的参数 ${\xi}$。

至此，最终的Soft-Margin SVM的目标为：

$${min(b,w,\xi)  \frac{1}{2} w^Tw+C \cdot \sum _{n=1}^{N} \xi_n}$$

条件是：

$${y_n(w^Tz_n+b) \geq 1 - \xi_n}$$

$${\xi_n \geq 0}$$

其中， ${\xi_n}$ 表示每个点犯错误的程度，${\xi_n=0}$，表示没有错误，${\xi_n}$ 越大，表示错误越大，即点距离边界（负的）越大。参数 ${C}$ 表示尽可能选择宽边界和尽可能不要犯错两者之间的权衡，因为边界宽了，往往犯错误的点会增加。${large \ C}$ 表示希望得到更少的分类错误，即不惜选择窄边界也要尽可能把更多点正确分类；${small \ C}$ 表示希望得到更宽的边界，即不惜增加错误点个数也要选择更宽的分类边界。

与之对应的QP问题中，由于新的参数 ${\xi_n}$ 的引入，总共参数个数为 ${\hat{d} +1+N}$，限制条件添加了 ${\xi_n \geq 0}$，则总条件个数为 ${2N}$。

![Soft-Margin SVM](http://ofqm89vhw.bkt.clouddn.com/1f3e489ae84b0cbefcc867970f9c2e37.png)

## Dual Problem

接下来，我们将推导 Soft-Margin SVM 的对偶 dual 形式，从而让 ${QP}$ 计算更加简单，并便于引入 kernel 算法。首先，我们把 Soft-Margin SVM 的原始形式写出来：

![primal](http://ofqm89vhw.bkt.clouddn.com/49282e3e5083beaeee446c923792c06e.png)

然后，跟我们在第二节课中介绍的 Hard-Margin SVM 做法一样，构造一个拉格朗日函数。因为引入了 ${\xi_n}$，原始问题有两类条件，所以包含了两个拉格朗日因子 ${\alpha_n}$ 和 ${\beta_n}$ 。拉格朗日函数可表示为如下形式：

![Lagrange function with Lagrange multipliers](http://ofqm89vhw.bkt.clouddn.com/c2528e6a5066af76a587c0b5b7071847.png)

接下来，我们跟第二节课中的做法一样，利用 Lagrange dual problem，将 Soft-Margin SVM 问题转换为如下形式：

![Simplify](http://ofqm89vhw.bkt.clouddn.com/f5d7b89a0653dfd1ab7306db0ca97196.png)

根据之前介绍的 ${KKT}$ 条件，我们对上式进行简化。上式括号里面的是对拉格朗日函数 ${L(b,w, \xi , \alpha , \beta)}$ 计算最小值。那么根据梯度下降算法思想：最小值位置满足梯度为零。

我们先对 ${\xi_n}$ 做偏微分：

$${\frac{\partial L}{\partial \xi_n}=0 = C - \alpha_n - \beta_n}$$

根据上式，得到 ${\beta_n=C - \alpha_n}$，因为有 ${\beta_n \geq 0}$，所以限制 ${0 \leq \alpha_n \leq C}$。将 ${\beta_n=C− \alpha_n}$ 代入到dual形式中并化简，我们发现 ${\beta_n}$ 和 ${\xi_n}$ 都被消去了：

![Other Simplifications](http://ofqm89vhw.bkt.clouddn.com/0911f28f69c71e8e0056eaa956b15054.png)

这个形式跟 Hard-Margin SVM 中的 dual 形式是基本一致的，只是条件不同。那么，我们分别令拉个朗日函数 ${L}$ 对 ${b}$ 和 ${w}$ 的偏导数为零，分别得到：

$${\sum_{n=1}^{N} \alpha_n y_n=0}$$

$${w= \sum_{n=1}^{N} \alpha_n y_n z_n}$$

经过化简和推导，最终标准的 Soft-Margin SVM 的 Dual 形式如下图所示：

![Standard Soft-Margin SVM Dual](http://ofqm89vhw.bkt.clouddn.com/cc4ab51230514612abfd044081afcbb4.png)

Soft-Margin SVM Dual 与 Hard-Margin SVM Dual 基本一致，只有一些条件不同。Hard-Margin SVM Dual 中 ${\alpha_n \geq 0}$，而 Soft-Margin SVM Dual 中 ${0 \leq  \alpha_n \leq C}$，且新的拉格朗日因子 ${\beta_n=C - \alpha_n}$。在 ${QP}$ 问题中，Soft-Margin SVM Dual 的参数 ${\alpha_n}$ 同样是 ${N}$ 个，但是，条件由 Hard-Margin SVM Dual 中的 ${N+1}$ 个变成${2N+1}$ 个，这是因为多了 ${N}$ 个 ${\alpha_n}$ 的上界条件。

## Messages behind Soft-Margin SVM

推导完 Soft-Margin SVM Dual 的简化形式后，就可以利用 ${QP}$，找到 ${Q}$，${p}$，${A}$，${c}$ 对应的值，用软件工具包得到 ${\alpha_n}$ 的值。或者利用核函数的方式，同样可以简化计算，优化分类效果。Soft-Margin SVM Dual 计算 ${\alpha_n}$ 的方法过程与Hard-Margin SVM Dual的过程是相同的。

![Kernel Soft-Margin SVM](http://ofqm89vhw.bkt.clouddn.com/5c974d4f7273783969a9495d6ed989d9.png)

但是如何根据 ${\alpha_n}$ 的值计算 ${b}$ 呢？在Hard-Margin SVM Dual中，有complementary slackness条件：${\alpha_n(1 - y_n(w^Tz_n+b))=0}$，找到 ${SV}$，即 ${\alpha s > 0}$ 的点，计算得到 ${b=ys - w^T z_s}$。

那么，在 Soft-Margin SVM Dual 中，相应的 complementary slackness 条件有两个（因为两个拉格朗日因子 ${\alpha_n}$ 和 ${\beta_n}$）：

$${\alpha_n(1 - \xi_n - y_n(w^Tz_n+b))=0}$$

$${\beta_n \xi_n =(C -  \alpha_n) \xi =0}$$

找到 ${SV}$，即 ${\alpha s>0}$ 的点，由于参数 ${\xi_n}$ 的存在，还不能完全计算出 ${b}$ 的值。根据第二个 complementary slackness 条件，如果令 ${C -  \alpha_n \neq 0}$ ，即 ${\alpha_n \neq C}$ ，则一定有 ${\xi_n=0}$，代入到第一个 complementary slackness 条件，即可计算得到 ${b=ys - w^T z_s}$ 。我们把 ${0< \alpha s<C}$ 的点称为 ${free \ SV}$ 。引入核函数后，${b}$ 的表达式为：

$${b=ys -  \sum_{SV} \alpha_n y_n K(x_n,x_s)}$$

上面求解 ${b}$ 提到的一个假设是 ${\alpha s<C}$，这个假设是否一定满足呢？如果没有 ${free \ SV}$ ，所有 ${\alpha s}$ 大于零的点都满足 ${\alpha s=C}$ 怎么办？一般情况下，至少存在一组 ${SV}$ 使 ${\alpha s<C}$ 的概率是很大的。如果出现没有 ${free \ SV}$ 的情况，那么 ${b}$ 通常会由许多不等式条件限制取值范围，值是不确定的，只要能找到其中满足 ${KKT}$ 条件的任意一个 ${b}$ 值就可以了。这部分细节比较复杂，不再赘述。

![Solving for b](http://ofqm89vhw.bkt.clouddn.com/647a20a41b79199637dc3ba2b3926ea5.png)

接下来，我们看看 ${C}$ 取不同的值对margin的影响。例如，对于 Soft-Margin Gaussian SVM，${C}$ 分别取${1}$，${10}$，${100}$ 时，相应的 margin 如下图所示：

![Soft-Margin Gaussian SVM in Action](http://ofqm89vhw.bkt.clouddn.com/942427c037be0f113178422e252994b7.png)

从上图可以看出，${C=1}$ 时，margin 比较粗，但是分类错误的点也比较多，当 ${C}$ 越来越大的时候，margin 越来越细，分类错误的点也在减少。正如前面介绍的，${C}$ 值反映了 margin 和分类正确的一个权衡。${C}$ 越小，越倾向于得到粗的 margin，宁可增加分类错误的点；${C}$ 越大，越倾向于得到高的分类正确率，宁可 margin 很细。我们发现，当 ${C}$ 值很大的时候，虽然分类正确率提高，但很可能把 noise 也进行了处理，从而可能造成过拟合。也就是说 Soft-Margin Gaussian SVM 同样可能会出现过拟合现象，所以参数${(\gamma ,C)}$ 的选择非常重要。

我们再来看看 ${\alpha_n}$ 取不同值是对应的物理意义。已知 ${0 \leq  \alpha_n \leq C}$ 满足两个complementary slackness条件：

$${\alpha_n(1 - \xi_n - y_n(w^Tz_n+b))=0}$$

$${\beta_n \xi_n=(C - \alpha_n) \xi =0}$$

若 ${\alpha_n=0}$，得 ${\xi_n=0}$。${\xi_n=0}$ 表示该点没有犯错，${\alpha_n=0}$ 表示该点不是 ${SV}$。所以对应的点在 margin 之外（或者在 margin 上），且均分类正确。

若 ${0< \alpha_n<C}$，得 ${\xi_n=0}$，且 ${y_n(w^Tz_n+b)=1}$。${\xi_n=0}$ 表示该点没有犯错，${y_n(w^Tz_n+b)=1}$ 表示该点在 margin 上。这些点即${free \ SV}$，确定了 ${b}$ 的值。

若 ${\alpha_n=C}$，不能确定 ${\xi_n}$ 是否为零，且得到 ${1 - y_n(w^Tz_n+b)= \xi_n}$ ，这个式表示该点偏离margin的程度，${\xi_n}$ 越大，偏离margin的程度越大。只有当 ${\xi_n=0}$ 时，该点落在 margin 上。所以这种情况对应的点在 margin 之内负方向（或者在 margin 上），有分类正确也有分类错误的。这些点称为 bounded ${SV}$。

所以，在 Soft-Margin SVM Dual 中，根据 ${\alpha_n}$ 的取值，就可以推断数据点在空间的分布情况。

![Physical Meaning of ${\alpha_n}$](http://ofqm89vhw.bkt.clouddn.com/c1bfed5722470f24801e68a904e955ab.png)

## Model Selection

在Soft-Margin SVM Dual中，kernel 的选择、${C}$ 等参数的选择都非常重要，直接影响分类效果。例如，对于 Gaussian SVM，不同的参数${(C,\gamma)}$ ，会得到不同的 margin，如下图所示。

![Practical Need: Model Selection](http://ofqm89vhw.bkt.clouddn.com/cf5ab7159af8d16bf338b0fd639d06c0.png)

其中横坐标是 ${C}$ 逐渐增大的情况，纵坐标是 ${\gamma}$ 逐渐增大的情况。不同的 ${(C,\gamma)}$ 组合，margin 的差别很大。那么如何选择最好的 ${(C,\gamma)}$ 等参数呢？最简单最好用的工具就是 validation。

validation 我们在《机器学习基石》课程中已经介绍过，只需要将由不同 ${(C,\gamma)}$ 等参数得到的模型在验证集上进行 cross validation，选取 ${E_{cv}}$ 最小的对应的模型就可以了。例如上图中各种 ${(C,\gamma)}$ 组合得到的 ${E_{cv}}$ 如下图所示：

![Selection by Cross Validation](http://ofqm89vhw.bkt.clouddn.com/8638d9719cb280ffcf3571895e44cdf4.png)

因为左下角的 ${E_{cv}(C,\gamma)}$ 最小，所以就选择该${(C,\gamma)}$ 对应的模型。通常来说，${E_{cv}(C,\gamma)}$ 并不是 ${(C,\gamma)}$ 的连续函数，很难使用最优化选择（例如梯度下降）。一般做法是选取不同的离散的 ${(C,\gamma)}$ 值进行组合，得到最小的 ${E_{cv}(C,\gamma)}$ ，其对应的模型即为最佳模型。这种算法就是我们之前在机器学习基石中介绍过的 ${V}$-Fold cross validation，在SVM 中使用非常广泛。

${V}$-Fold cross validation 的一种极限就是 Leave-One-Out ${CV}$，也就是验证集只有一个样本。对于 SVM 问题，它的验证集 ${Error}$ 满足：

$${E_{loocv} \leq \frac{SV}{N}}$$

也就是说留一法验证集 ${Error}$ 大小不超过支持向量 ${SV}$ 占所有样本的比例。下面做简单的证明。令样本总数为 ${N}$ ，对这 ${N}$ 个点进行 SVM 分类后得到 margin，假设第 ${N}$ 个点 ${(x_N,y_N)}$ 的 ${\alpha N=0}$，不是 ${SV}$ ，即远离 margin（正距离）。这时候，如果我们只使用剩下的 ${N-1}$ 个点来进行 SVM 分类，那么第 ${N}$ 个点 ${(x_N,y_N)}$ 必然是分类正确的点，所得的SVM margin跟使用 ${N}$ 个点的到的是完全一致的。这是因为我们假设第 ${N}$ 个点是 ${non-SV}$ ，对 ${SV}$ 没有贡献，不影响 margin 的位置和形状。所以前 ${N-1}$ 个点和 ${N}$ 个点得到的 margin 是一样的。

那么，对于 non-SV 的点，它的 ${g^{-} =g}$ ，即对第 ${N}$ 个点，它的 Error 必然为零：

$${e_{non-SV} = err(g^{-},non-SV)=err(g,non-SV)=0}$$

另一方面，假设第 ${N}$ 个点 ${\alpha N \neq 0}$ ，即对于 ${SV}$ 的点，它的 ${Error}$ 可能是 ${0}$ ，也可能是 ${1}$ ，必然有：

$${e_{SV} \leq 1}$$

综上所述，即证明了 ${E_{loocv} \leq \frac{SV}{N}}$ 。这符合我们之前得到的结论，即只有 ${SV}$ 影响 margin，non-SV 对 margin 没有任何影响，可以舍弃。

${SV}$ 的数量在 SVM 模型选择中也是很重要的。一般来说，${SV}$ 越多，表示模型可能越复杂，越有可能会造成过拟合。所以，通常选择 ${SV}$ 数量较少的模型，然后在剩下的模型中使用 cross-validation，比较选择最佳模型。

## 总结

本节课主要介绍了Soft-Margin SVM。我们的出发点是与Hard-Margin SVM不同，不一定要将所有的样本点都完全分开，允许有分类错误的点，而使margin比较宽。然后，我们增加了 ${\xi_n}$ 作为分类错误的惩罚项，根据之前介绍的Dual SVM，推导出了Soft-Margin SVM的QP形式。得到的 ${\alpha_n}$ 除了要满足大于零，还有一个上界 ${C}$。接着介绍了通过 ${\alpha_n}$ 值的大小，可以将数据点分为三种：${non-SVs}$ ，${free \ SVs}$，${bounded \ SVs}$，这种更清晰的物理解释便于数据分析。最后介绍了如何选择合适的SVM模型，通常的办法是cross-validation和利用SV的数量进行筛选。

## 参考

1. [台湾大学林轩田机器学习技法课程学习笔记4 -- Soft-Margin Support Vector Machine](http://blog.csdn.net/red_stone1/article/details/74279607)