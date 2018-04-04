# Lecture 01 - The Learning Problem

- When Can Machine Learn ?
- Why Can Machine Learn ?
- How Can Machine Learn ?
- How Can Machine Learn Better ?

## What is Machine Learning

什么是“学习”？学习就是人类通过观察、积累经验，掌握某项技能或能力。就好像我们从小学习识别字母、认识汉字，就是学习的过程。而 **`机器学习`（Machine Learning），顾名思义，就是让机器（计算机）也能向人类一样，通过观察大量的数据和训练，发现事物规律，获得某种分析问题、解决问题的能力。**

![From Learning to Machine Learning](http://ofqm89vhw.bkt.clouddn.com/bde751135d696e30e0c8fe8fb3b606a5.png)

机器学习可以被定义为：**Improving some performance measure with experence computed from data.** 也就是机器从数据中总结经验，从数据中找出某种规律或者模型，并用它来解决实际问题。

![A More Concrete Definition](http://ofqm89vhw.bkt.clouddn.com/9d23f42fa9d4f5a09e58018c1ccfbe76.png)

什么情况下会使用机器学习来解决问题呢？其实，目前机器学习的应用非常广泛，基本上任何场合都能够看到它的身影。其应用场合大致可归纳为三个条件：

![Key Essence of Machine Learning](http://ofqm89vhw.bkt.clouddn.com/35dab4d0e3e334c560edf4fd05cc2d26.png)

- 事物本身存在某种潜在规律。
- 某些问题难以使用普通编程解决。
- 有大量的数据样本可供使用。

## Applications of Machine Learning

机器学习在我们的`衣`、`食`、`住`、`行`、`教育`、`娱乐`等各个方面都有着广泛的应用，我们的生活处处都离不开机器学习。比如，打开购物网站，网站就会给我们自动推荐我们可能会喜欢的商品；电影频道会根据用户的浏览记录和观影记录，向不同用户推荐他们可能喜欢的电影等等，到处都有机器学习的影子。

## Components of Machine Learning

本系列的课程对机器学习问题有一些基本的术语需要注意一下：

- 输入 ${x}$。
- 输出 ${y}$。
- 目标函数 ${f}$ ，即最接近实际样本分布的规律。
- 训练样本 ${data}$ 。
- 假设 ${hypothesis}$ ，一个机器学习模型对应了很多不同的 ${hypothesis}$ ，通过演算法 ${A}$ ，选择一个最佳的 ${hypothesis}$ 对应的函数称为 ${g}$ ，${g}$ 能最好地表示事物的内在规律，也是我们最终想要得到的模型表达式。

![Basic Notations](http://ofqm89vhw.bkt.clouddn.com/6e05f57b4090d304eeea9a0496dd52b4.png)

对于理想的目标函数 ${f}$ ，我们是不知道的，我们手上拿到的是一些训练样本 ${D}$ ，假设是监督式学习，其中有输入 ${x}$ ，也有输出 ${y}$ 。机器学习的过程，就是根据先验知识选择模型，该模型对应的 ${hypothesis\ set}$（用 ${H}$ 表示），${H}$ 中包含了许多不同的 ${hypothesis}$ ，通过演算法 ${A}$ ，在训练样本 ${D}$ 上进行训练，选择出一个最好的 ${hypothes}$ ，对应的函数表达式 ${g}$ 就是我们最终要求的。一般情况下， ${g}$ 能最接近目标函数 ${f}$ ，这样，机器学习的整个流程就完成了。

![Learning Flow](http://ofqm89vhw.bkt.clouddn.com/8dad5d8697d58894760718af5a976294.png)

## Machine Learning and Other Fields

与机器学习相关的领域有：

- 数据挖掘（Data Mining）
- 人工智能（Artificial Intelligence）
- 统计（Statistics）

其实，机器学习与这三个领域是相通的，基本类似，但也不完全一样。机器学习是这三个领域中的有力工具，而同时，这三个领域也是机器学习可以广泛应用的领域，总得来说，他们之间没有十分明确的界线。

## 总结

本节课主要介绍了什么是机器学习，什么样的场合下可以使用机器学习解决问题，然后用流程图的形式展示了机器学习的整个过程，最后把机器学习和数据挖掘、人工智能、统计这三个领域做个比较。本节课的内容主要是概述性的东西，比较简单，所以笔记也相对比较简略。

## 参考

1. [台湾大学林轩田机器学习基石课程学习笔记1 -- The Learning Problem](http://blog.csdn.net/red_stone1/article/details/72899485)