# 从 SNE 到 t-SNE 再到 LargeVis

- [从 SNE 到 t-SNE 再到 LargeVis](#%e4%bb%8e-sne-%e5%88%b0-t-sne-%e5%86%8d%e5%88%b0-largevis)
- [0x00 前言](#0x00-%e5%89%8d%e8%a8%80)
- [0x01 预备知识](#0x01-%e9%a2%84%e5%a4%87%e7%9f%a5%e8%af%86)
  - [降维](#%e9%99%8d%e7%bb%b4)
  - [流形学习](#%e6%b5%81%e5%bd%a2%e5%ad%a6%e4%b9%a0)
  - [t 分布](#t-%e5%88%86%e5%b8%83)
  - [kNN 图](#knn-%e5%9b%be)
  - [k-d 树与随机投影树](#k-d-%e6%a0%91%e4%b8%8e%e9%9a%8f%e6%9c%ba%e6%8a%95%e5%bd%b1%e6%a0%91)
  - [LINE](#line)
  - [负采样](#%e8%b4%9f%e9%87%87%e6%a0%b7)
- [0x02 从 SNE 说起](#0x02-%e4%bb%8e-sne-%e8%af%b4%e8%b5%b7)
- [0x03 从 SNE 到 t-SNE——小小的 t 分布，迈进了一大步](#0x03-%e4%bb%8e-sne-%e5%88%b0-t-sne%e5%b0%8f%e5%b0%8f%e7%9a%84-t-%e5%88%86%e5%b8%83%e8%bf%88%e8%bf%9b%e4%ba%86%e4%b8%80%e5%a4%a7%e6%ad%a5)
  - [对称 SNE](#%e5%af%b9%e7%a7%b0-sne)
  - [拥挤问题(The Crowding Problem)](#%e6%8b%a5%e6%8c%a4%e9%97%ae%e9%a2%98the-crowding-problem)
  - [神奇的长尾——t 分布](#%e7%a5%9e%e5%a5%87%e7%9a%84%e9%95%bf%e5%b0%bet-%e5%88%86%e5%b8%83)
- [0x04 t-SNE 的改进——各种树算法轮番上阵](#0x04-t-sne-%e7%9a%84%e6%94%b9%e8%bf%9b%e5%90%84%e7%a7%8d%e6%a0%91%e7%ae%97%e6%b3%95%e8%bd%ae%e7%95%aa%e4%b8%8a%e9%98%b5)
  - [用 kNN 图表示高维空间中点的相似性](#%e7%94%a8-knn-%e5%9b%be%e8%a1%a8%e7%a4%ba%e9%ab%98%e7%bb%b4%e7%a9%ba%e9%97%b4%e4%b8%ad%e7%82%b9%e7%9a%84%e7%9b%b8%e4%bc%bc%e6%80%a7)
  - [梯度中斥力的近似](#%e6%a2%af%e5%ba%a6%e4%b8%ad%e6%96%a5%e5%8a%9b%e7%9a%84%e8%bf%91%e4%bc%bc)
  - [从点-区域到区域-区域](#%e4%bb%8e%e7%82%b9-%e5%8c%ba%e5%9f%9f%e5%88%b0%e5%8c%ba%e5%9f%9f-%e5%8c%ba%e5%9f%9f)
- [0x05 从 t-SNE 再到 LargeVis——厚积薄发](#0x05-%e4%bb%8e-t-sne-%e5%86%8d%e5%88%b0-largevis%e5%8e%9a%e7%a7%af%e8%96%84%e5%8f%91)
  - [高效 kNN 图构建算法](#%e9%ab%98%e6%95%88-knn-%e5%9b%be%e6%9e%84%e5%bb%ba%e7%ae%97%e6%b3%95)
  - [低维空间可视化算法](#%e4%bd%8e%e7%bb%b4%e7%a9%ba%e9%97%b4%e5%8f%af%e8%a7%86%e5%8c%96%e7%ae%97%e6%b3%95)
- [0x06 总结](#0x06-%e6%80%bb%e7%bb%93)
- [0x07 参考资料](#0x07-%e5%8f%82%e8%80%83%e8%b5%84%e6%96%99)
  - [SNE](#sne)
  - [t-SNE](#t-sne)
  - [LINE&LargeVis](#linelargevis)
  - [word2vec](#word2vec)
  - [其他(引用了图或者例子)](#%e5%85%b6%e4%bb%96%e5%bc%95%e7%94%a8%e4%ba%86%e5%9b%be%e6%88%96%e8%80%85%e4%be%8b%e5%ad%90)

# 0x00 前言

本文谢绝转载，如有需要请联系 bindog@outlook.com

数据可视化是大数据领域非常倚重的一项技术，但由于业内浮躁的大环境影响，这项技术的地位渐渐有些尴尬。尤其是在诸如态势感知、威胁情报等应用中，简陋的可视化效果太丑，过于华丽的可视化效果只能忽悠忽悠外行，而给内行的感觉就是刻意为之、华而不实。

曾几何时，可视化技术不过是一种数据分析的手段罢了。惭愧的说就是我们的算法还不够智能，必须依靠人类的智慧介入分析。所以，需要通过可视化技术把高维空间中的数据以二维或三维的形式展示给我们这样的低维生物看，展示的效果如何也就直接决定着我们分析的难度。

抛开浮躁的大环境，在数据可视化领域还是有人踏踏实实做研究的，比如深度学习大牛 Hinton(SNE)、Maaten(t-SNE)还有[唐建大神](http://research.microsoft.com/en-us/people/jiatang/)(LargeVis，新鲜出炉，*WWW’16*最佳论文提名)，下面言归正传，我们从简单的基础知识开始。

# 0x01 预备知识

说实在的，想要彻底搞清楚这些算法的原理并不轻松，需要长时间关注和积累。这里我把所需要知识和资料简要列出，供大家有针对性的了解。

## 降维

降维顾名思义就是把数据或特征的维数降低，一般分为线性降维和非线性降维，比较典型的如下：

- 线性降维：PCA(Principal Components Analysis)、LDA(Linear Discriminant Analysis)、MDS(Classical Multidimensional Scaling)
- 非线性降维：Isomap(Isometric Mapping)、LLE(Locally Linear Embedding)、LE(Laplacian Eigenmaps)

大家可能对线性降维中的一些方法比较熟悉了，但是对非线性降维并不了解，非线性降维中用到的方法大多属于流形学习方法。

## 流形学习

流形学习(Manifold Learning)听名字就觉得非常深奥，涉及微分流行和黎曼几何等数学知识。当然，想要了解流形学习并不需要我们一行一行的去推导公式，通过简单的例子也能够有一个直观的认识。关于流行学习的科普文章首推 pluskid 写的[《浅谈流行学习》](http://blog.pluskid.org/?p=533)，里面有很多通俗易懂的例子和解释。

简单来说，地球表面就是一个典型的流形，在流形上计算距离与欧式空间有所区别。例如，计算南极与北极点之间的距离不是从地心穿一个洞计算直线距离，而是沿着地球表面寻找一条最短路径，这样的一条路径称为**测地线**。如下面所示的三幅图

![流形与测地线](http://lc-cf2bfs1v.cn-n1.lcfile.com/6a0a667a8081616f.png)

其中第一张图为原始数据点分布，红色虚线是欧式距离，蓝色实线是沿着流形的真实测地线距离。第二张图是在原始数据点的基础上基于欧式距离构造的 kNN 图（灰色线条，下面还会具体介绍 kNN 图），红色实线表示 kNN 图中两点之间的最短路径距离。第三张图是将流形展开后的效果，可以看到，kNN 图中的最短路径距离（红色实线）要略长于真实测地线距离（蓝色实线）。

在实际应用中，真实测地距离较难获得，一般可以通过构造 kNN 图，在 kNN 图中寻找最短路径距离作为真实测地线距离的近似。

## t 分布

大家在概率与统计课程中都接触过$t$分布的概念，从正态总体中抽取容量为$N$的随机样本，若该正态总体的均值为$\mu$，方差为${\sigma}^2$。随机样本均值为$\bar{x}$，方差为$s^2=\frac{1}{N-1}\sum_{i=1}^{N}(x_i-\bar{x})^2$，随机变量$t$可表示为：
$t=\frac{\bar{x}-\mu}{s/\sqrt{N}}$
此时我们称$t$服从自由度为$n-1$的$t$分布，即$t \sim t(n-1)$

下图展示了不同自由度下的$t$分布形状与正态分布对比，其中自由度为 1 的$t$分布也称为柯西分布。自由度越大，$t$分布的形状越接近正态分布。

![正态分布与t分布](http://lc-cf2bfs1v.cn-n1.lcfile.com/823840b0a7738efb.png)

从图中还可以看出，$t$分布比正态分布要“胖”一些，尤其在尾部两端较为平缓。$t$分布是一种典型的长尾分布。实际上，在[稳定分布](https://en.wikipedia.org/wiki/Stable_distribution)家族中，除了正态分布，其他均为长尾分布。长尾分布有什么好处呢？在处理小样本和一些异常点的时候作用就突显出来了。下文介绍 t-sne 算法时也会涉及到 t 分布的长尾特性。

## kNN 图

kNN 图(k-Nearest Neighbour Graph)实际上是在经典的 kNN(k-Nearest Neighbor)算法上增加了一步构图过程。假设空间中有$n$个节点，对节点$v_i$，通过某种距离度量方式（欧式距离、编辑距离）找出距离它最近的$k$个邻居${v_1,v_2,\cdots,v_k}$，然后分别将$v_i$与这$k$个邻居连接起来，形成$k$条有向边。对空间中所有顶点均按此方式进行，最后就得到了 kNN 图。

当然，为方便起见，在许多场景中我们往往将 kNN 图中的有向边视为无向边处理。如下图是一个二维空间中以欧式距离为度量的 kNN 图。

![kNN图样例](http://lc-cf2bfs1v.cn-n1.lcfile.com/94568c30f8b7698c.png)

kNN 图的一种用途上文已经提到过：在计算流形上的测地线距离时，可以构造基于欧式距离的 kNN 图得到一个近似。原因很简单，我们可以把一个流形在很小的局部邻域上近似看成欧式的，也就是局部线性的。这一点很好理解，比如我们所处的地球表面就是一个流形，在范围较小的日常生活中依然可以使用欧式几何。但是在航海、航空等范围较大的实际问题中，再使用欧式几何就不合适了，使用黎曼几何更加精确。

kNN 图还可用于异常点检测。在大量高维数据点中，一般正常的数据点会聚集为一个个簇，而异常数据点与正常数据点簇的距离较远。通过构建 kNN 图，可以快速找出这样的异常点。

## k-d 树与随机投影树

刚才说到 kNN 图在寻找流形的过程中非常有用，那么如何来构建一个 kNN 图呢？常见的方法一般有三类：第一类是空间分割树(space-partitioning trees)算法，第二类是局部敏感哈希(locality sensitive hashing)算法，第三类是邻居搜索(neighbor exploring techniques)算法。其中 k-d 树和随机投影树均属于第一类算法。

很多同学可能不太熟悉随机投影树(Random Projection Tree)，但一般都听说过 k-d 树。k-d 树是一种分割 k 维数据空间的数据结构，本质上是一棵二叉树。主要用于多维空间关键数据的搜索，如范围搜索、最近邻搜索等。那么如何使用 k-d 树搜索 k 近邻，进而构建 kNN 图呢？我们以二维空间为例进行说明，如下图所示：

![k-d树示意图](http://lc-cf2bfs1v.cn-n1.lcfile.com/1bd1eadb508a9986.png)

上图是一个二维空间的 k-d 树，构建 k-d 树是一个递归的过程，根节点对应区域内所有点，将空间按某一维划分为左子树和右子树之后，重复根结点的分割过程即可得到下一级子节点，直到 k-d 树中所有叶子节点对应的点个数小于某个阈值。

有了 k-d 树之后，我们寻找 k 近邻就不用挨个计算某个点与其他所有点之间的距离了。例如寻找下图中红点的 k 近邻，只需要搜索当前子空间，同时不断回溯搜索父节点的其他子空间，即可找到 k 近邻点。

![k-d树找k近邻](http://lc-cf2bfs1v.cn-n1.lcfile.com/f41ba87492c96da2.png)

当然，搜索过程还有一些缩小搜索范围的方法，例如画圆判断是否与父节点的分割超平面相交等等，这里就不展开讨论了。

不过 k-d 树最大的问题在于其划分空间的方式比较死板，是严格按照坐标轴来的。对高维数据来说，就是将高维数据的每一维作为一个坐标轴。当数据维数较高时，k-d 树的深度可想而知，**维数灾难**问题也不可避免。相比之下，随机投影树划分空间的方式就比较灵活，还是以二维空间为例，如下图所示：

![随机投影树示意图](http://lc-cf2bfs1v.cn-n1.lcfile.com/459e3b29b5065043.png)

随机投影树的基本思路还是与 k-d 树类似的，不过划分空间的方式不是按坐标轴了，而是按随机产生的单位向量。有的同学说，这样就能保证随机投影树的深度不至于太深吗？随机产生的单位向量有那么靠谱吗？这里需要注意的是，我们所分析的数据处于一个流形上的，并非是杂乱无章的，因此从理论上讲，随机投影树的深度并不由数据的维数决定，而取决于数据所处的流形维数。(此处可参考 Freund 等人的论文《Learning the structure of manifolds using random projections》)

那么如何使用随机投影树寻找 k 近邻呢？当然可以采用和 k-d 树类似回溯搜索方法。但是当我们对 k 近邻的精确度要求不高时，可以采用一个更加简单巧妙的方式，充分利用随机投影树的特性。简单来说，我们可以并行的构建多个随机投影树，由于划分的单位向量都是随机产生的，因此每棵随机投影树对当前空间的划分都是不相同的，如下图所示

![随机投影树找k近邻](http://lc-cf2bfs1v.cn-n1.lcfile.com/644b8aaa16d0b2a5.png)

例如我们想搜索红点的 k 近邻，只需要在不同的随机投影树中搜索其所处的子空间(或者仅回溯一层父结点)，最后取并集即可。这样做虽然在构建随机投影树的过程中较为耗时耗空间，但是在搜索阶段无疑是非常高效的。

## LINE

LINE，即 Large-scale Information Network Embedding，是唐建大神 2015 年的一项工作(_www’15_)。内容依旧很好很强大，而且代码是开源的。

一句话概括，LINE 是“Embed Everything”思想在网络表示中的发扬光大。自从 Mikolov 开源 word2vec 以来，词向量(word embedding)的概念在 NLP 界可谓是火的一塌糊涂，embedding 的概念更是快速渗透到其他各研究领域。entity embedding、relation embedding…等如雨后春笋般涌现，更是有人在 Twitter 上犀利的吐槽：

![twitter吐槽](http://lc-cf2bfs1v.cn-n1.lcfile.com/87a9088866752f77.png)

当然，这里完全没有贬低 LINE 的意思，事实上 LINE 的工作是非常出色的，主要有两大突出贡献：一是能够适应各种类型(无向边或有向边、带权值不带权值的)的大规模(百万级节点、十亿级边)网络，而且能够很好的捕获网络中的一阶和二阶相似性；二是提出了非常给力的**边采样算法**(edge-sampling algorithm)，大幅降低了 LINE 的时间复杂度，使用边采样算法后时间复杂度与网络中边的数量呈线性关系。LargeVis 的高效也得益于 LINE 及其边采样算法。

其中一阶相似性指的是网络中两个节点之间的点对相似性，具体为节点之间边的权重(如果点对不存在边，则其一阶相似性为 0)；二阶相似性指的是若节点间共享相似的邻居节点，那么两者就趋于相似。比如下图展示的这种情况，边的权值大小用粗细表示：

![一阶二阶相似度](http://lc-cf2bfs1v.cn-n1.lcfile.com/f233ea2102c812f3.png)

其中节点 8 与节点 9 之间的一阶相似性为较高，因为其直接连接边的权值较高。节点 1 与节点 7 有着绝大多数相同的邻居，因此两者的二阶相似性非常高。

边采样算法的思路来源于 Mikolov 在 word2vec 中使用的**负采样**优化技术。既提高了训练的效率，也解决了网络表示中带权值边在训练过程中造成的梯度剧增问题，具体的边采样算法在下文涉及的地方进行介绍。

## 负采样

了解 word2vec 的同学一定对负采样(Negative sampling)不陌生，Mikolov 在 word2vec 中集成了 CBOW 和 Skip-gram 两种词向量模型，在训练过程中使用到了多项优化技术，负采样正是其中一种优化技术。我们以 Skip-gram 模型为例进行说明，Skip-gram 模型的思路是从目标词预测上下文，用一个上下文窗口限定文本范围，如下图所示：

![Skip-gram](http://lc-cf2bfs1v.cn-n1.lcfile.com/f6397e4873f8c449.png)

Skip-gram 模型需要最大化“做了”“一点”“的”“贡献”等词语出现在目标词“微小”周围的概率，即最大化$p(c\mid w)=\sum{p(w_i\mid w)}$。出现在目标词周围上下文窗口中的词$w_i \in c$构成一个正样本$(w_i,w)$，未出现在目标词周围的词$w_j \in D$构成负样本$(w_j,w)$。我们在训练过程中要最大化正样本出现的概率，同时也要减小负样本出现的概率。为什么要减小负样本出现的概率呢，只提高正样本出现的概率不就可以了吗？举个不恰当的例子，这就好比垄断一样，为了达到最终目的不择手段，一方面肯定要加强自身产品的竞争力，另一方面竞争对手也在发展壮大，发展水平并不比我们差，所以必须使用些手段，打压消灭竞争对手，负采样也是这么个道理。

由于负样本数量众多(上下文窗口之外的词基本都可以构成负样本)，直接考虑所有的负样本显然是不现实的，所以我们用采样的方式选一部分负样本出来即可。那么负采样具体如何采样呢？在语料中有的词语出现频率高，有的词语出现频率低，直接从词表中随机抽取负样本显然是不科学的。word2vec 中使用的是一种**带权采样**策略，即根据词频进行采样，高频词被采样的概率较大，低频词被采样的概率较小。

那么具体如何带权采样呢？看下面这张图，词$w_i$的词频用$f_{w_i}$表示

![带权采样](http://lc-cf2bfs1v.cn-n1.lcfile.com/cd91f11ea9c8d27f.png)

上面那根线段是按词频进行分割的，词频越高线段较长，下面的线段是等距离分割。我们往下方的线段中随机打点(均匀分布)，根据点所落在的区间对应到上方的线段，即可确定所采样的词。直观来看，采用这种方式词频较高的词被采样到的概率更大，词频较低的词被采样到的概率更低。

加入负采样优化之后，目标函数的形式变为
$\log \sigma (v*{w_c}^T \cdot v_w)+\sum*{i=1}^k\ _{E_{w*i}\sim P_n(f)}[\log \sigma(-v*{w_i}^T\cdot v_w)]$
其中$w$表示目标词，$w_c$表示目标词周围上下文窗口中的词(正样本)，$w_i$表示未出现在上下文窗口中的词(负样本)，$k$表示抽取的负样本个数，$P_n(f)$是用于负样本生成的噪声分布，$f$表示词频，$P_n(f)\propto f^{0.75}$，不要问我 0.75 怎么来的，Mikolov 做实验得出来的，直接用原始词频效果不好，加个 0.75 次幂效果较好。

word2vec 里面还有许多有意思的细节，感兴趣的同学可以去看看 peghoty 写的《[word2vec 中的数学原理](http://blog.csdn.net/itplus/article/details/37998797)》

# 0x02 从 SNE 说起

了解完预备知识后，我们可以从 SNE 开始本趟可视化算法之旅了。SNE 即 stochastic neighbor embedding，是 Hinton 老人家 2002 年提出来的一个算法，出发点很简单：在高维空间相似的数据点，映射到低维空间距离也是相似的。常规的做法是用欧式距离表示这种相似性，而 SNE 把这种距离关系转换为一种条件概率来表示相似性。什么意思呢？考虑高维空间中的两个数据点$x_i$和$x_j$，$x_i$以条件概率$p_{j\mid i}$选择$x_j$作为它的邻近点。考虑以$x_i$为中心点的高斯分布，若$x_j$越靠近$x_i$，则$p_{j\mid i}$越大。反之，若两者相距较远，则$p_{j\mid i}$极小。因此，我们可以这样定义$p_{j\mid i}$：
$p*{j|i}=\frac{\exp (-\left \| x_i-x_j \right \|^2/2 \sigma*{i}^2)}{\sum*{k \neq i}\exp (-\left \| x_i-x_k \right \|^2/2 \sigma*{i}^2)}$
其中$\sigma_{i}$表示以$x_i$为中心点的高斯分布的方差。由于我们只关心不同点对之间的相似度，所以设定$p_{i\mid i}=0$。

当我们把数据映射到低维空间后，高维数据点之间的相似性也应该在低维空间的数据点上体现出来。这里同样用条件概率的形式描述，假设高维数据点$x_i$和$x_j$在低维空间的映射点分别为$y_i$和$y_j$。类似的，低维空间中的条件概率用$q_{j\mid i}$表示，并将所有高斯分布的方差均设定为$\frac{1}{\sqrt 2}$，所以有：
$q*{j|i}=\frac{\exp (-\left \| y_i-y_j \right \|^2)}{\sum*{k \neq i}\exp (-\left \| y*i-y_k \right \|^2)}$
同理，设定\$q*{i\mid i}=0$。此时就很明朗了，若$y*i$和$y*j$真实反映了高维数据点$x_i$和$x_j$之间的关系，那么条件概率$p*{j\mid i}$与$q*{j\mid i}$应该完全相等。这里我们只考虑了$x_i$与$x_j$之间的条件概率，若考虑$x_i$与其他所有点之间的条件概率，则可构成一个条件概率分布$P_i$，同理在低维空间存在一个条件概率分布$Q_i$且应该与$P_i\$一致。如何衡量两个分布之间的相似性？当然是用经典的 KL 距离(Kullback-Leibler Divergence)，SNE 最终目标就是对所有数据点最小化这个 KL 距离，我们可以使用梯度下降算法最小化如下代价函数：
\$C=\sum*{i}KL(P*i||Q_i)=\sum_i \sum_j p*{j|i} \log \frac{p*{j|i}}{q*{j|i}}$
似乎到这里问题就漂亮的解决了，你看我们代价函数都写出来了，剩下的事情就是利用梯度下降算法进行训练了。但事情远没有那么简单，因为 KL 距离是一个非对称的度量。最小化代价函数的目的是让\$p*{j\mid i}$和$q*{j\mid i}$的值尽可能的接近，即低维空间中点的相似性应当与高维空间中点的相似性一致。但是从代价函数的形式就可以看出，当$p*{j\mid i}$较大，$q*{j\mid i}$较小时，代价较高；而$p*{j\mid i}$较小，$q\_{j\mid i}\$较大时，代价较低。什么意思呢？很显然，高维空间中两个数据点距离较近时，若映射到低维空间后距离较远，那么将得到一个很高的惩罚，这当然没问题。反之，高维空间中两个数据点距离较远时，若映射到低维空间距离较近，将得到一个很低的惩罚值，这就有问题了，理应得到一个较高的惩罚才对。换句话说，SNE 的代价函数更关注局部结构，而忽视了全局结构。

SNE 代价函数对$y_i$求梯度后的形式如下：
$\frac{\delta C}{\delta y*i}=2\sum*{j}(p*{j|i}-q*{j|i}+p*{i|j}-q*{i|j})(y*i-y_j)$
这个梯度还有一定的物理意义，我们可以用分子之间的引力和斥力进行解释。低维空间中点$y_i$的位置是由其他所有点对其作用力的合力所决定的。其中某个点$y_j$对其作用力是沿着$y_i-y_j$方向的，具体是引力还是斥力占主导就取决于$y_j$与$y_i$之间的距离了，其实就与\$(p*{j\mid i}-q*{j\mid i}+p\*{i\mid j}-q\_{i\mid j})\$这一项有关。

SNE 算法中还有一个细节是关于高维空间中以点$x_i$为中心的正态分布方差$\sigma_i$的选取，这里不展开讲了，有兴趣的同学可以去看看论文。

最后，我们来看一下 SNE 算法的效果图。将 SNE 算法用在 UPS database 的手写数字数据集上(五种数字，01234)，效果如下：

![sne效果图](http://lc-cf2bfs1v.cn-n1.lcfile.com/5eddecaad72424c3.png)

从图中可以看出，SNE 的可视化效果还算可以，同一类别的数据点映射到二维空间后基本都能聚集在一起，但是不同簇之间的边界过于模糊。老实说，如果不是这个图上把不同类别用不同颜色和符号标识出来，根本没法把边界处的数据点区分开来，做可视化分析也非常不方便。这个问题下面我们还会详细分析。

# 0x03 从 SNE 到 t-SNE——小小的 t 分布，迈进了一大步

SNE 算法的思路是不错的，但是它的可视化效果大家也看到了，存在很大改进空间。如何改进它呢？我们一步一步来，先看看如何解决 SNE 中的不对称问题。

## 对称 SNE

在原始的 SNE 中，$p_{i\mid j}$与$p_{j\mid i}$是不相等的，低维空间中$q_{i\mid j}$与$q_{j\mid i}$也是不相等的。所以如果能得出一个更加通用的联合概率分布更加合理，即分别在高维和低维空间构造联合概率分布$P$和$Q$，使得对任意$i,j$，均有$p_{ij}=p_{ji},q_{ij}=q_{ji}$。

在低维空间中，我们可以这样定义$q_{ij}$：
$q*{ij}=\frac{\exp (-\left \| y_i-y_j \right \|^2)}{\sum*{k \neq l}\exp (-\left \| y*k-y_l \right \|^2)}$
在高维空间呢？是不是可以想当然的写出：
$p*{ij}=\frac{\exp (-\left \| x*i-x_j \right \|^2/2\sigma^2)}{\sum*{k \neq l}\exp (-\left \| x*k-x_l \right \|^2/2\sigma^2)}$
但是如果这样定义\$p*{ij}$又会遭遇刚才的问题，考虑一个离群点$x*i$，它与所有结点之间的距离都较大，那么对所有$j$，$p*{ij}$的值均较小，所以无论该离群点在低维空间中的映射点$y*i$处在什么位置，惩罚值都不会太高，这显然也不是我们希望看到的。所以这里采用一种更简单直观的方式定义$p*{ij}$：
$p*{ij}=\frac{p*{j|i}+p\_{i|j}}{2n}$
其中$n$为数据点的总数，这样定义即满足了对称性，又保证了$x*i\$的惩罚值不会过小。此时可以利用 KL 距离写出如下代价函数：
$C=KL(P||Q)=\sum_i \sum_j p*{ij}\log \frac{p*{ij}}{q*{ij}}$
梯度变为：
$\frac{\delta C}{\delta y*i}=4\sum*{j}(p*{ij}-q*{ij})(y_i-y_j)$
相比刚才定义的公式，这个梯度更加简化，计算效率更高。但是别高兴的太早，虽然我们解决了 SNE 中的不对称问题，得到了一个更为简单的梯度公式，但是 Maaten 指出，对称 SNE 的效果只是略微优于原始 SNE 的效果，依然没有从根本上解决问题。

## 拥挤问题(The Crowding Problem)

所谓拥挤问题，顾名思义，看看 SNE 的可视化效果，不同类别的簇挤在一起，无法区分开来，这就是拥挤问题。有的同学说，是不是因为 SNE 更关注局部结构，而忽略了全局结构造成的？这的确有一定影响，但是别忘了使用对称 SNE 时同样存在拥挤问题。实际上，拥挤问题的出现与某个特定算法无关，而是由于高维空间距离分布和低维空间距离分布的差异造成的。

我们生活在一个低维的世界里，所以有些时候思维方式容易受到制约。比如在讨论流形学习问题的时候，总喜欢拿一个经典的“Swiss roll”作为例子，这只不过是把一个简单的二维流形嵌入到三维空间里而已。实际上真实世界的数据形态远比“Swiss roll”复杂，比如一个 10 维的流形嵌入到更高维度的空间中，现在我们的问题是把这个 10 维的流形找出来，并且映射到二维空间上可视化。在进行可视化时，问题就来了，在 10 维流形上可以存在 11 个点且两两之间距离相等。在二维空间中呢？我们最多只能使三个点两两之间距离相等，想将高维空间中的距离关系完整保留到低维空间是不可能的。

这里通过一个实验进一步说明，假设一个以数据点$x_i$为中心，半径为$r$的$m$维球(二维空间就是圆，三维空间就是球)，其体积是按$r^m$增长的，假设数据点是在$m$维球中均匀分布的，我们来看看其他数据点与$x_i$的距离随维度增大而产生的变化。

代码如下所示：

```python
    # -*- coding: utf-8 -*-
    importmatplotlib.pyplotaspltimportnumpyasnpfromnumpy.linalgimportnormnpoints=1000# 抽取1000个m维球内均匀分布的点
    plt.figure(figsize=(20,4))fori,minenumerate((2,3,5,8)):# 这里模拟m维球中的均匀分布用到了拒绝采样，即先生成m维立方中的均匀分布，再剔除m维球外部的点
    accepts=[]whilelen(accepts)<1000:points=np.random.rand(500,m)accepts.extend([dfordinnorm(points,axis=1)ifd<=1.0])# 拒绝采样
    accepts=accepts[:npoints]ax=plt.subplot(1,4,i+1)ax.set_xlabel('distance')# x轴表示点到圆心的距离
    ifi==0:ax.set_ylabel('count')# y轴表示点的数量
    ax.hist(accepts,bins=np.linspace(0.,1.,50),color='green')ax.set_title('m={0}'.format(str(m)),loc='left')plt.show()
```

结果如下图所示：

![高维空间距离分布](http://lc-cf2bfs1v.cn-n1.lcfile.com/11bf4eea600c0fb6.png)

从图中可以看到，随着维度的增大，大部分数据点都聚集在$m$维球的表面附近，与点$x_i$的距离分布极不均衡。如果直接将这种距离关系保留到低维，肯定会出现拥挤问题。如何解决呢？这个时候就需要请出$t$分布了。

## 神奇的长尾——t 分布

刚才预备知识部分说到，像$t$分布这样的长尾分布，在处理小样本和异常点时有着非常明显的优势，例如下面这个图：

![异常点与t分布](http://lc-cf2bfs1v.cn-n1.lcfile.com/119b8b66f320da19.png)

从图中可以看到，在没有异常点时，$t$分布与高斯分布的拟合结果基本一致。而在第二张图中，出现了部分异常点，由于高斯分布的尾部较低，对异常点比较敏感，为了照顾这些异常点，高斯分布的拟合结果偏离了大多数样本所在位置，方差也较大。相比之下，$t$分布的尾部较高，对异常点不敏感，保证了其鲁棒性，因此其拟合结果更为合理，较好的捕获了数据的整体特征。
那么如何利用$t$分布的长尾性来改进 SNE 呢？我们来看下面这张图，注意这个图并不准确，主要是为了说明$t$分布是如何发挥作用的。

![神奇的长尾](http://lc-cf2bfs1v.cn-n1.lcfile.com/48c85252fba44ba5.png)

图中有高斯分布和$t$分布两条曲线，表示点之间的相似性与距离的关系，高斯分布对应高维空间，$t$分布对应低维空间。**那么对于高维空间中相距较近的点，为了满足$p_{ij}=q_{ij}$，低维空间中的距离需要稍小一点；而对于高维空间中相距较远的点，为了满足$p_{ij}=q_{ij}$，低维空间中的距离需要更远。**这恰好满足了我们的需求，即同一簇内的点(距离较近)聚合的更紧密，不同簇之间的点(距离较远)更加疏远。我们使用自由度为 1 的$t$分布重新定义$q_{ij}$：
$q*{ij}=\frac{ (1 + \left \| y_i-y_j \right \|^2)^{-1}}{\sum*{k \neq l} (1 + \left \| y*k-y_l \right \|^2)^{-1}}$
依然用 KL 距离衡量两个分布之间的相似性，此时梯度变为
$\frac{\delta C}{\delta y_i}=4\sum*{j}(p*{ij}-q*{ij})(y_i-y_j)(1 + \left \| y_i-y_j \right \|^2)^{-1}$
再利用随机梯度下降算法训练即可。

这就是所谓的 t-SNE 算法，总结一下其实就是在 SNE 的基础上增加了两个改进：一是把 SNE 变为对称 SNE，二是在低维空间中采用了$t$分布代替原来的高斯分布，高维空间不变。最后来看看 t-SNE 在可视化效果上是如何完虐其他算法的。(ps:Maaten 还是比较厚道的，没有在论文中展示 SNE 的可视化效果图，不然也是赤果果的打 Hinton 脸)

![t-SNE可视化效果-Olivetti faces](http://lc-cf2bfs1v.cn-n1.lcfile.com/181958b03d47f6c5.png)

![t-SNE可视化效果-COIL-20](http://lc-cf2bfs1v.cn-n1.lcfile.com/033629c26aba22de.png)

其他算法真是被虐的体无完肤……想看更多图的可以去看原文，这里再给大家展示一个 t-SNE 算法过程中的点簇之间距离变化的一个动画，使用的是经典的 MNIST 数据集。

![t-sne动画](http://lc-cf2bfs1v.cn-n1.lcfile.com/fe6782ce2f1b7875.gif)

# 0x04 t-SNE 的改进——各种树算法轮番上阵

2014 年的时候，Maaten 又写了一篇论文对 t-SNE 算法进行了改进，使用了各种基于树的算法，具体包括两部分内容：一是采用了 kNN 图来表示高维空间中点的相似性；二是优化了梯度的求解过程，将梯度计算分为引力和斥力两部分，同样使用了一些优化技巧。下面我们简单看看这两项优化的基本思路，具体用到的树算法这里只是简单提一下，不作深入介绍了，感兴趣的同学直接到文末的参考资料中找对应的论文即可。

## 用 kNN 图表示高维空间中点的相似性

在 t-SNE 中，我们用高斯分布描述了高维空间中整体的距离分布关系，注意表达式是这样的一个形式：
$p*{j|i}=\frac{\exp (-\left \| x_i-x_j \right \|^2/2 \sigma*{i}^2)}{\sum*{k \neq i}\exp (-\left \| x_i-x_k \right \|^2/2 \sigma*{i}^2)}p*{i|i}=0p*{ij}=\frac{p*{j|i}+p*{i|j}}{2n}$
对每一个数据点都包含$\sum_{k \neq i}\exp (-\left | x_i-x_k \right |^2/2 \sigma_{i}^2)$这样一项，当数据量较大时，计算量无疑是非常大的。但实际上，两个相距较远的点互为邻居的概率$p_{ij}$是非常小的，几乎可以忽略。因此，在高维空间对一个点构建距离相似性关系时，不必考虑图中的每一个节点，只需考虑与其相近的若干个节点即可。这里我们考虑与点$x_i$最近的$\lfloor {3u} \rfloor$个点，其中$u$为点$x_i$的周围条件概率分布的 perplexity，将这些点的集合表示为$N_i$，可得到
$p*{j|i}=\frac{\exp (-\left \| x_i-x_j \right \|^2/2 \sigma*{i}^2)}{\sum*{k \in N_i}\exp (-\left \| x_i-x_k \right \|^2/2 \sigma*{i}^2)}p*{i|i}=0p*{ij}=\frac{p*{j|i}+p*{i|j}}{2n}$
这样就大大降低了计算量，但也引入了一个新的问题，我们必须先构建一个高维空间的 kNN 图。Maaten 采用了 VP 树(vantage-point tree)来构建这个 kNN 图，可以在$O(uNlogN)$的时间复杂度内得到一个精确的 kNN 图。

## 梯度中斥力的近似

在介绍 t-SNE 梯度的物理意义时，提到可以将梯度视为所有点对$y_i$的合力作用，这里我们利用这个特点对梯度进行变换，令$Z=\sum_{k \ne l}(1 + \left | y_k-y_l \right |^2)^{-1}$
$\frac{\delta C}{\delta y*i}=4\sum*{j}(p*{ij}-q*{ij})(y*i-y_j)(1 + \left \| y_i-y_j \right \|^2)^{-1}=4\sum*{j}(p*{ij}-q*{ij})q*{ij}Z(y_i-y_j)\frac{\delta C}{\delta y_i}=4(F*{attr}+F*{rep})=4(\sum*{j \ne i} p*{ij}q*{ij}Z(y*i-y_j)-\sum*{j \ne i} q*{ij}^2Z(y_i-y_j))$
可以看到，将梯度分解为引力和斥力两部分后，其中引力部分计算起来较为容易，因为\$q\*{ij}Z=(1+\left | y_i-y_j \right |^2)^{-1}$，我们可以采用刚才的近似，只考虑最近邻的若干个节点，忽略较远的节点，时间复杂度为$O(uN)$。但是计算斥力依然有较大的计算量，时间复杂度约为$O(N^2)$因为$P$和$Q\$的分布是不一样的，不过我们依然可以使用一些优化技巧来简化这个计算。

看下面这个图，有三个点$y_i,y_j,y_k$，当$y_i-y_j \approx y_i-y_k \gt \gt y_j-y_k$时，我们可以认为点$y_j$和点$y_k$对$y_i$的斥力是近似相等的。

![斥力近似](http://lc-cf2bfs1v.cn-n1.lcfile.com/216c1183c1eeb4e7.png)

事实上，这种情况在低维空间中是很常见的，甚至某一片区域中每个点对$y_i$的斥力均可用同一个值来近似。现在问题来了，任意给定一个点，我们需要一个方法快速搜索出符合条件的区域，并用刚才的优化方法计算出每块区域的总斥力。如下图所示：

![点-区域](http://lc-cf2bfs1v.cn-n1.lcfile.com/ddab5376643f1065.png)

对上图的情形，假设区域$A$中的 5 个点对$y_i$产生的斥力都是近似相等的，那么我们可以计算这 5 个点的中心点(虚构的点)产生的斥力$F_{A_c}$，区域$A$产生的总斥力为$5F_{A_c}$。Maaten 使用了四叉树来完成区域搜索任务，并用该区域中心点产生的斥力作为整个区域的代表值，当然，并非所有区域都满足这个近似条件，这里使用 Barnes-Hut 算法搜索并验证符合近似条件的点-区域对。

## 从点-区域到区域-区域

在上面的近似中，我们考虑的是一个点与一个区域之间斥力的近似，事实上，我们可以更进一步的优化，考虑一个区域与另一个区域之间斥力的近似。如下图这种情形，$A$和$B$两个区域之间任意两个节点产生的斥力均可用$F_{AB_c}$近似。

![区域-区域](http://lc-cf2bfs1v.cn-n1.lcfile.com/e1f825520462f9bb.png)

同样，也需要判断两个区域之间的斥力是否满足近似的条件，Maaten 采用了 Dual-tree 算法搜索并验证符合近似条件的区域-区域对。

# 0x05 从 t-SNE 再到 LargeVis——厚积薄发

虽然 t-SNE 算法和它的改进算法都得到广泛应用，但存在两个不足：一是处理大规模高维数据时，t-SNE 的效率显著降低(包括改进后的算法)；二是 t-SNE 中的参数对不同数据集较为敏感，我们辛辛苦苦的在一个数据集上调好了参数，得到了一个不错的可视化效果，却发现不能在另一个数据集上适用，还得花费大量时间寻找合适的参数。

唐建提出的 LargeVis 基本思路与 t-SNE 改进算法大致相同，如下图所示：

![LargeVis思路](http://lc-cf2bfs1v.cn-n1.lcfile.com/eecf348a369a5512.png)

但是 LargeVis 用到了几个非常漂亮的优化技巧，包括他之前在 LINE 上的工作。我们分别来看看这些另人耳目一新的优化技术。

## 高效 kNN 图构建算法

在 t-SNE 的改进算法中，高维空间距离相似性我们只考虑与$x_i$最接近的若干个邻居点，这实质上就是一个构建 kNN 图的过程。Maaten 使用了 VP 树来构建一个精确的 kNN 图，但是效率依然堪忧。而 LargeVis 采用了一种更巧妙的方式，不追求一步到位，先近似再提高准确率。

在预备知识部分我们说到，构建 kNN 图一般有三类方法，LargeVis 的做法是将第一类方法和第三类方法相结合。具体来说，第一步先利用随机投影树得到一个空间划分，在此基础上寻找每个点的$k$近邻，得到一个初步 kNN 图，这个 kNN 图不要求完全准确。第二步根据“邻居的邻居可能也是我的邻居”的思想，利用邻居搜索算法寻找潜在的邻居，计算邻居与当前点、邻居的邻居与当前点的距离并放入一个小根堆之中，取距离最小的 k 个节点作为 k 近邻，最终得到一个精确的 kNN 图。

## 低维空间可视化算法

在低维空间可视化过程中，t-SNE 的思路是保证高维空间的距离分布 P 与低维空间的距离分布 Q 尽可能接近，用 KL 距离写出代价函数并求梯度。但是效率问题也很突出，不然 Maaten 也不会花那么大力气，利用各种树算法优化这个梯度求解的过程。

由于我本人是研究自然语言处理的，所以看到 LargeVis 在这一块的优化思路，联想到 word2vec 中类似的优化思路，很有一种拍案叫绝的感觉。这里我也重点从 word2vec 的角度进行阐释。

首先我们知道 word2vec 中用到了两个模型，CBOW 和 Skip-gram 模型，此外还有众多优化技术，其中一个叫负采样，这个预备知识中强调过了，不记得的同学再回到文章开头看一下。那么在网络中其实也是相类似的，我们可以把当前中心点视为目标词，其邻居节点视为上下文窗口中出现的词，那么中心点和其邻居节点即构成一个正样本，而中心点与非邻居点构成一个负样本。

![网络正负样本](http://lc-cf2bfs1v.cn-n1.lcfile.com/2bd602a6982ea1bf.png)

如上面展示的 kNN 图中的一小部分，若蓝点为中心点，则每个绿点可与蓝点构成正样本，而每个黄点与蓝点构成负样本。如何把这个 kNN 图中的结构关系映射到低维空间呢？直观来看，在低维空间应该是正样本中的节点对聚合在一起，而负样本中的节点对分散的较远。
我们先考虑无权值网络的情况，用$y_i$和$y_j$表示低维空间中的两个点，两个点在 kNN 图中有一条二元边$e_{ij}=1$(权值为 1 的边)的概率为：
$P(e\_{ij}=1)=f(\left \| y_i-y_j \right \|^2)$
其中$f(\cdot)$类似的用到了 t-SNE 里的$t$分布，$f(x)=\frac{1}{1+x^2}$，若$y_i$和$y_j$之间的距离越小，两点在 kNN 图中有二元边的概率较大，反之若$y_i$和$y_j$之间的距离越大，则两点在 kNN 图中有二元边的概率越小。

LargeVis 还考虑了有权值网络的情况，定义边权值为$w_{ij}$的概率为：
$P(e*{ij}=w*{ij})=P(e*{ij}=1)^{w*{ij}}$
现在我们假定正样本集合为$E$，负样本集合为$\bar E$，正样本和负样本的集合都可以直接通过 kNN 图获得。优化目标可以写为：
$O=\prod*{(i,j) \in E}p(e*{ij}=1)^{w*{ij}} \prod*{(i,j) \in \bar E}(1-p(e*{ij}=1))^{\gamma}$
整个优化目标很好理解，就是最大化正样本的节点对在 kNN 图中有连接边的概率，最小化负样本的节点对在 kNN 图中有连接边的概率，其中$\gamma$是我们统一为负样本边设定的权值。这里再取一个对数，优化目标变为：
$O=\sum*{(i,j) \in E}w*{ij}p(e*{ij}=1) \sum*{(i,j) \in \bar E}\gamma(1-p(e*{ij}=1))$
那么是不是到这里又可以用梯度下降开始训练了呢？很遗憾，还是不行，这里我们使用的是所有负样本$\bar E$，计算量太大，是时候请出负采样算法了。对每一个点$i$，我们根据一个噪声分布$P_n(j)$随机选取$M$个点与$i$构成负样本，这个噪声分布采用与 Mikolov 等人使用的噪声分布类似的形式，即$P_n(j)\propto d_j^{0.75}$，其中$d_j$为点$j$的度(这里用的是节点的度，word2vec 用的是词的词频)。此时我们可重新定义目标函数：
$O=\sum*{(i,j) \in E}w*{ij}(p(e*{ij}=1)+\sum*{k=1}^M\ _{E_{j*k}\sim P_n(j)}\gamma \log(1-p(e*{ij*k}=1)))$
看这个目标函数，是不是和使用了负采样技术的 Skip-gram 模型的目标函数非常类似？我们再仔细观察上面的目标函数，求出梯度后，边权值\$w*{ij}$仍是作为乘积的一项出现的，这就带来一个问题，网络中边的权值$w*{ij}$变化范围是很大的，所以受$w\*{ij}\$的影响，梯度的变化也会较大，这对训练是非常不利的，也就是所谓的**梯度剧增与消失问题**(gradient explosion and vanishing problem)。

到了这一步就可以用 LINE 里面的边采样技术了，其实原理和负采样是一致的，只不过用在了正样本中。若正样本中两个点之间边的权值为$w_{ij}$，我们可以将其转换为$w_{ij}$个重叠的二元边，就像下图这种形式，假如权值是 5，就转换成 5 条二元边。

![边采样](http://lc-cf2bfs1v.cn-n1.lcfile.com/1551769952654dea.png)

若存在多个较大的带权值边(上百甚至上千)，转换成二元边之后，总边数也是非常多的，全部考虑的话同样影响效率。因此，这里将所有带权值边全部转换为二元边之后(相当于做了等距离分割)，再随机从这些二元边中进行采样(相当于带权采样)。边采样算法的优势在于：一方面，由于采样到的边是二元边，因此权值都是一样的，解决了梯度变化范围大的问题；另一方面，采样过程实质遵循了带权采样策略，因为权值较大的边转换得到的二元边更多，被采样到的概率也就越大，保证了正确性与合理性。

利用负采样和边采样优化之后，LargeVis 还用到了异步随机梯度下降来进行训练，这项技术在稀疏图上是非常有效的，因为不同线程采样的边所连接的两个节点很少有重复的，不同线程之间几乎不会产生冲突。从时间复杂度上来看，每一轮随机梯度下降的时间复杂度为$O(sM)$，其中$M$是负样本个数，$s$是低维空间的维数(2 或 3)，随机梯度的步数通常又与点节数量$N$成正比，因此总的时间复杂度为$O(sMN)$。从这里可以知道，LargeVis 的时间复杂度是与网络中的节点数量呈线性关系的。

从可视化效果上看，LargeVis 与 t-SNE 是差不多的，在某些数据集上略胜一筹，但是从训练时间上看，LargeVis 比 t-SNE 高效太多。

![LargeVis与t-SNE对比效果](http://lc-cf2bfs1v.cn-n1.lcfile.com/f2263169bc403d83.png)

下面是准确率与时间的对比，LargeVis 果然不是浪得虚名。

![LargeVis与t-SNE对比准确率和时间](http://lc-cf2bfs1v.cn-n1.lcfile.com/e174418deebd3c69.png)

不过到目前为止，LargeVis 的代码还没有放出来。还请大家多点耐心，拭目以待。

# 0x06 总结

从 SNE 到 t-SNE 再到 LargeVis，SNE 奠定了一个非常牢靠的基础，却遗留了一个棘手的拥挤问题；t-SNE 用$t$分布巧妙的解决了拥挤问题，并采用了多种树算法改进算法效率；LargeVis 在 t-SNE 改进算法的基础上，参考了近年来较为新颖的优化技巧，如随机投影树、负采样、边采样(实质也是负采样)等，直接将训练的时间复杂度降至线性级。在表示学习和深度学习如此火热的年代，任何一种经典的模型或方法都有可能在其他领域发挥不可思议的妙用。word2vec 中的 Skip-gram 模型和负采样优化技术在 LargeVis 中的应用就是很好的证明。

值得一提的是，Maaten 提出 t-SNE 的时间是 2008 年，进一步改进 t-SNE 的时间是 2014 年，唐建提出 LINE 和 LargeVis 的时间分别是 2015 年和 2016 年。从这个角度看，t-SNE 还是一个非常经典的算法，毕竟傲视群雄了这么多年……不过从另一个角度看，科研之路漫漫，一项值得称道的技术或改进不是一蹴而就的，是要经过长时间积累和沉淀的。

# 0x07 参考资料

## SNE

- [Hinton G E, Roweis S T. **Stochastic neighbor embedding**[C]//Advances in neural information processing systems. 2002: 833-840.](http://machinelearning.wustl.edu/mlpapers/paper_files/AA45.pdf)

## t-SNE

- [Van der Maaten L, Hinton G. **Visualizing data using t-SNE**[J]. Journal of Machine Learning Research, 2008, 9(2579-2605): 85.](http://siplab.tudelft.nl/sites/default/files/vandermaaten08a.pdf)
- [Van Der Maaten L. **Accelerating t-sne using tree-based algorithms**[J]. The Journal of Machine Learning Research, 2014, 15(1): 3221-3245.](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

## LINE&LargeVis

- [Tang J, Qu M, Wang M, et al. **Line: Large-scale information network embedding**[C]//Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2015: 1067-1077.](http://arxiv.org/pdf/1503.03578.pdf)
- [Tang J, Liu J, Zhang M, et al. **Visualizing Large-scale and High-dimensional Data**[C]//Proceedings of the 25th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2016: 287-297.](http://arxiv.org/pdf/1602.00370.pdf)

## word2vec

- [Mikolov T, Sutskever I, Chen K, et al. **Distributed representations of words and phrases and their compositionality**[C]//Advances in neural information processing systems. 2013: 3111-3119.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- [Goldberg Y, Levy O. **word2vec explained: Deriving mikolov et al.’s negative-sampling word-embedding method**[J]. arXiv preprint arXiv:1402.3722, 2014.](http://arxiv.org/pdf/1402.3722.pdf)

## 其他(引用了图或者例子)

- [Heylen R, Parente M, Gader P. **A review of nonlinear hyperspectral unmixing methods**[J]. Selected Topics in Applied Earth Observations and Remote Sensing, IEEE Journal of, 2014, 7(6): 1844-1868.](https://www.researchgate.net/profile/Rob_Heylen/publication/264564339_A_Review_of_Nonlinear_Hyperspectral_Unmixing_Methods/links/53f485700cf2888a7491048d.pdf)
- [Freund Y, Dasgupta S, Kabra M, et al. **Learning the structure of manifolds using random projections**[C]//Advances in Neural Information Processing Systems. 2007: 473-480.](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2007_133.pdf)
- [Characterizing a Distribution](http://work.thaslwanter.at/Stats/html/statsDistributions.html#t-distribution)
- [An illustrated introduction to the t-SNE algorithm](https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm)
- [浅谈流形学习](http://blog.pluskid.org/?p=533)
- [word2vec 中的数学原理详解（五）基于 Negative Sampling 的模型](http://blog.csdn.net/itplus/article/details/37998797)
