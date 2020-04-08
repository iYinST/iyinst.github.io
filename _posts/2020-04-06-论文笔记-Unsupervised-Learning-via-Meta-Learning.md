---
layout: post
title: "论文笔记《Unsupervised Learning via Meta-Learning》"
subtitle: '无监督元学习'
author: "iYinST"
header-style: text
# header-img: "img/post-bg-alibaba.jpg"
tags:
  - 论文笔记
  - 无监督
  - 元学习
  - meta learning
---

# 论文笔记《*Unsupervised Learning via Meta-Learning*》

《Unsupervised Learning via Meta-Learning》发表于ICLR 2019，作者是多伦多大学的Kyle Hsu和加州伯克利的Sergey Levine和Chelsea Finn。

《Unsupervised Learning via Meta-Learning》提出了一种**无监督元学习方法**，可以优化在少量数据集上学习各种任务的能力。实验构建在四个图像数据集上，实验表明这种无监督元学习方法无需下游分类任务的label，即可获得一种学习算法。

![image-20200406144342750](/img/in-post/image-20200406143613190.png)

## 介绍

无监督学习是一类基本但尚未解决的问题，无监督学习主要应用场景是在无监督表示的基础上进行下游任务。然而由于下游目标需要监督，因此用于无监督学习的目标只是下游目标的粗略替代。如果无监督学习的主要目标是好的数据表示，是否可以将如何使用表示结果作为学习目标？

在下游任务中使用无监督表示与元学习技术的目标紧密相关：找到比从头开始训练更有效的学习算法。与无监督学习不同的是元学习方法需要标记的数据集和手工指定的任务分布。这些约束是小样本分类广泛应用的主要障碍。

为了开始解决这些问题，本文提出了一种无监督的元学习方法：该方法旨在在无监督的情况下学习如何学习的过程，这对于解决各种新的，人类指定的任务很有用。仅使用原始的未标记的数据，模型的目标是学习有用的先验知识，以便在元训练之后，该模型可以转移其先验经验以有效地学习执行新知识任务。

要无监督的元学习需要**根据未标记的数据自动构造任务**。一个好的任务分配应该是多样的，但也不应该太困难：用于任务生成的简单随机方法所产生的任务包含的规则性不足以实现有用的元学习。为此，本文的方法首先通过利用先前的无监督学习算法来学习输入数据的embedding来提出任务，然后对数据集进行完全的划分以构造多分类。

本文的核心思想是**利用无监督的embedding为元学习算法提出任务**，从而产生无监督的元学习算法，该算法对于下游任务的预训练特别有效。



## 无监督元学习

本文的目标是利用未标记的数据来有效地学习一系列人为指定的下游任务。假设元学习的未标记的数据$\mathcal{D} = \lbrace  x_i \rbrace $。

无监督embedding算法$\mathcal{\epsilon}$将未标记的数据集$D = \lbrace  x_i \rbrace $作为输入，输出从$ \lbrace  x_i \rbrace $到嵌入$\lbrace  z_i \rbrace $的映射。这些embedding点通常是低维的，并且距离对应于输入之间的有意义的差异，而不是原始输入（例如图像像素）之间的距离。

一个$M$-way $K$-shot的分类任务中每个类别$\tau$包含K个训练数据点和标签$\lbrace ( x_k,l_k )\rbrace $，和$Q$个查询（query）数据点和标签（验证）。也就是说在一个任务中，$M$类别中每个类别都有$K + Q = R$个数据点和标签。

本文基于两种元学习算法：模型不可知元学习（MAML）和原型网络（ProtoNets）。 MAML旨在学习深度网络的初始参数，以便接下来在一个或几个梯度步骤内可以得到有效的模型。 ProtoNets旨在通过元学习来学习一种表示形式，可以通过其原型有效地识别出一个类，该原型定义在元学习空间中的训练样本的平均值; $\mathcal{F}$是这些类原型的计算，并且$\mathcal{f}$是一个线性分类器， 预测在欧氏距离中query表示最近的类。

为了构造一个$N$-way分类任务$\mathcal{T}$（假设N不大于label的数量），可以对采样$N$个类，对每个类别采样$R$个数据点
$$
\left\{\mathbf{x}_{r}\right\}_{n}
$$
，并采样N个不同的one-hot向量$$\left(\ell_{n}\right)$$的排列，以用作N个采样类的标签。任务定义为
$$
\mathcal{T}=\left\{\left(\mathbf{x}_{n, r}, \ell_{n}\right)| \mathbf{x}_{n, r} \in\left\{\mathbf{x}_{r}\right\}_{n}\right\}
$$
。此过程仅适用于有标签的数据。

从元学习的角度解决问题，将问题归结为如何从未标记的数据中获取可转换为人工设计的任务，尤其是旨在从未标记的数据中构建分类任务和然后再学习如何有效地学习分类任务。如果从未标记的数据中构建出的任务具有足够的多样性和结构性，那么经过元学习后模型应该可以快速学习新的、人工提供的任务。那么关键的问题就是如何根据未标记的$\mathcal{D} = \lbrace  x_i \rbrace $自动构建此样的任务。通过对所有数据点分配标签$y_c$，可以将整个数据集$\mathcal{D} = \lbrace  x_i \rbrace $进行划分$\mathcal{P} = \lbrace  \mathcal{C}_\mathcal{c} \rbrace $。一旦有了数据集的划分，任务生成变得简单，因此可以将问题从由未标记的数据集$\mathcal{D} = \lbrace  x_i \rbrace $构造任务简化为由未标记的数据集$\mathcal{D} = \lbrace  x_i \rbrace $构造划分。接下来就是如何构建划分。

最简便的方法是随机分割$\mathcal{D}$。尽管随机划分引入了多样性的任务，但这些任务没有结构性。也就是说任务的训练数据和查询数据之间没有一致性（完全随机），因此在每个任务期间都不会学到任何东西，更不用说跨任务学习了。

如果要构造任务，并且任务具有类似于人类指定label的结构，就需要将数据集依据显著特征划分为一致且不同的子集，因此考虑$k$-means聚类。将$k$-means划分的$\mathcal{P} = \lbrace  \mathcal{C}_\mathcal{c} \rbrace $作为高斯混合模型$p(x\|c)p(c)$的简化。如果聚类可以表现真实类别的条件生成分布$p(x\|c)p(c)$的，则将聚类视为类别划分的任务就可以用于元学习的训练。但是$k$-means的聚类结果严格取决于度量空间，像素空间中的聚类不吸引人的原因有两个：（1）像素空间中的距离与语义的相关性很差；（2）原始图像维数过高导致聚类困难。实验表明由像素空间聚类的任务进行的元学习失败了。

受在语义空间中常用距离聚类的启发，使用最先进的无监督学习方法来生成有效的embedding空间。尽管embedding空间可能不直接适合于高效地学习新任务，但仍然可以利用它来构建结构化的任务。

因此首先使用out-of-the-box的无监督embedding算法$\mathcal{\epsilon}$，然后将数据$$\left\{\mathbf{x}_{i}\right\}$$映射到embedding空间$\mathcal{Z} = \lbrace  z_i \rbrace$。为了产生多样化的任务集，通过运行$P$次聚类算法来产生$P$个划分$\lbrace \mathcal{P}_p \rbrace $，每次聚类时随机缩放$\mathcal{Z}$的维度以便引入不同的度量，随机缩放由对角矩阵$A$来表示。$\mu _ c$表示学习到的聚类$\mathcal{C} _c$的中心，每次聚类可以简单表示为
$$
\mathcal{P},\left\{\boldsymbol{\mu}_{c}\right\}=\underset{\left\{\mathcal{C}_{c}\right\},\left\{\boldsymbol{\mu}_{c}\right\}}{\arg \min } \sum_{c=1}^{k} \sum_{\mathbf{z} \in \mathcal{C}_{c}}\left\|\mathbf{z}-\boldsymbol{\mu}_{c}\right\|_{\mathbf{A}}^{2}
$$

为了避免聚类簇不平衡构造的任务，选择不对
$$
p(c) \propto\left|\mathcal{C}_{c}\right|
$$
进行采样，而是对每个任务进行均匀采样N个簇。

在$$\left\{\mathbf{Z}_{i}\right\}$$上构建划分之后，还需要考虑：是否应该对embedding或图像进行meta-learning吗？作者认为为了在元测试阶段成功解决新任务，将embedding作为输入的学习过程$$\mathcal{F}$$取决于embedding函数将推广到分布外观测的能力。另外通过对图像的元学习，$$\mathcal{F}$$可以从最原始的表示形式分别将$$f$$适应每个评估任务。因此作者选择对图像进行元学习。算法细节如下。![image-20200408143748240](/img/in-post/image-20200408143748240.png)

