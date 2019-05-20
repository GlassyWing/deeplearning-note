# Sampled softmax

原文：https://www.tensorflow.org/extras/candidate_sampling.pdf

假设有一个单分类问题。训练集中的每个样本$(x_i, \{t_i\})$包含一个上下文和一个目标类。将给定上下文$x$时，目标类为$y$写作概率$P(y|x)$

我们使用函数$F(x,y)$产生softmax logits，即对数化的概率：

$$ F(x, y) \leftarrow \log (P(y | x))+K(x) $$

其中$K(x)$是一个不依赖$y$的函数。

在完整的softmax训练中，对于每个训练样本$(x_i, \{t_i\})$，都需要对所有的分类$y \in L$计算$F(x_i,y)$。如果分类集L非常大，该操作将变得非常昂贵。

而在“Sampled Softmax”中，对于每个训练样本$(x_i, \{t_i\})$，我们根据一个选定的抽样函数$Q(y|x)$来选择一个小的采样分类集$S_{i} \subset L$，其中的每一个分类$ y \in L$都以概率$Q(y|x_i)$独立的存在：

$$
P\left(S_{i}=S | x_{i}\right)=\prod_{y \in S} Q\left(y | x_{i}\right) \prod_{y \in(L-S)}\left(1-Q\left(y | x_{i}\right)\right)
$$

我们创建一个候选集$C_i$，联合了目标分类和采样的分类集：

$$
C_{i}=S_{i} \cup\left\{t_{i}\right\}
$$

我们的训练任务是计算出，在给定候选集$C_i$的条件下，$C_i$中的哪一个分类是目标分类。

对于$C_i$中的每一个分类$y$，我们想要计算出当给定$x_i$和$C_i$时，$y$的后验概率，记作$P\left(t_{i}=y | x, C_{i}\right)$:

应用贝叶斯法则：
$$
\begin{array}{l}
P\left(t_{i}=y | x_{i}, C_{i}\right)
\\ = \frac{P\left(t_{i}=y, C_{i} | x_{i}\right)}{P\left(C_{i} | x_{i}\right)} 
\\  {= \frac{P\left(t_{i}=y | x_{i}\right) P\left(C_{i} | t_{i}=y, x_{i}\right)}{P\left(C_{i} | x_{i}\right)} }
\\ = \frac{P\left(y | x_{i}\right) P\left(C_{i} | t_{i}=y, x_{i}\right)}{P\left(C_{i} | x_{i}\right)} 
\end{array}
$$

现在，来计算$P\left(C_{i} | t_{i}=y, x_{i}\right)$，我们注意到要使这发生，$y$可能在也可能不在$S_i$中，$S_i$一定包含$C_i$中所有的其它（除$y$外）元素，并且不包含任何不在$C_i$中的元素，
因此：

$$
\begin{array}{l}
P\left(C_{i} | t_{i}=y, x_{i}\right)
\\ =\prod_{y^{\prime} \in C_{i}-\{y\}} Q\left(y^{\prime} | x_{i}\right) \prod_{y^{\prime} \in\left(L-C_{i}\right)}\left(1-Q\left(y^{\prime} | x_{i}\right)\right)
\\ = \frac{1}{Q(y|x_i)} \prod_{y^{\prime} \in C_{i}} Q\left(y^{\prime} | x_{i}\right) \prod_{y^{\prime} \in\left(L-C_{i}\right)}\left(1-Q\left(y^{\prime} | x_{i}\right)\right)
\end{array}
$$

$$
\begin{array}{l}
P\left(t_{i}=y | x_{i}, C_{i}\right)
\\ = \frac{P\left(y | x_{i}\right) P\left(C_{i} | t_{i}=y, x_{i}\right)}{P\left(C_{i} | x_{i}\right)} 
\\ = \frac{P(y|x_i)}{Q(y|x_i)} \prod_{y^{\prime} \in C_{i}} Q\left(y^{\prime} | x_{i}\right) \prod_{y^{\prime} \in\left(L-C_{i}\right)}\left(1-Q\left(y^{\prime} | x_{i}\right)\right) / P(C_i|x_i)
\\ = \frac{P(y|x_i)}{Q(y|x_i)} / K(x_i, C_i)
\end{array}
$$

$K(x_i, C_i)$是一个不依赖于y的函数，所以：
$$
\log \left(P\left(t_{i}=y | x_{i}, C_{i}\right)\right)=\log \left(P\left(y | x_{i}\right)\right)-\log \left(Q\left(y | x_{i}\right)\right)+K^{\prime}\left(x_{i}, C_{i}\right)
$$

这些是应该输入softmax分类器的相对logits，用于预测$C_i$中的哪一个候选类才是真正的分类。

既然我们试图训练函数$F(x, y)$来拟合$log(P(y|x))$，我们用神经网络中的层输出表示$F(x, y)$，然后减去$log(Q(y|x))$，将结果传入一个softmax分类器来预测哪个候选是真正的分类。

$$
Training Softmax Input=F(x, y)-\log (Q(y | x)
$$

从分类输出中反向传播梯度到F，这就是我们所要的。