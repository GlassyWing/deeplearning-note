# Transformer-xl

## 循环机制

训练阶段，每个隐层接收两个输入

1. 该段下层隐藏层的输出，与原始Transformer相同
2. 前段下层隐藏层的输出，使其建模长期依赖关系

$$ \widetilde{h}_{\tau + 1} ^ {n-1} = [SG(h_{\tau}^{n-1}) \circ h_{\tau + 1} ^{n-1}] \tag{extened context}  $$

$$
\mathbf{q}_{\tau+1}^{n}, \mathbf{k}_{\tau+1}^{n}, \mathbf{v}_{\tau+1}^{n}=\mathbf{h}_{\tau+1}^{n-1} \mathbf{W}_{q}^{\top}, \widetilde{\mathbf{h}}_{\tau+1}^{n-1} \mathbf{W}_{k}^{\top}, \widetilde{\mathbf{h}}_{\tau+1}^{n-1} \mathbf{W}_{v}^{\top}
$$

$$
\mathbf{h}_{\tau+1}^{n}=\text { Transformer-Layer }\left(\mathbf{q}_{\tau+1}^{n}, \mathbf{k}_{\tau+1}^{n}, \mathbf{v}_{\tau+1}^{n}\right)
$$

## 相对位置编码

若是每个段继续使用相同的位置编码，比如段1的编码[0, 1, 2]，段2的编码也是[0, 1, 2]，则组合后，位置编码变成了[0, 1, 2, 0, 1, 2]，而每个位置的语义在整个序列中应当是不一致的。

在原Transformer中，计算查询$q_i^T$与键$k_j$之间的注意力方式为：

$$
\mathbf{A}_{i, j}^{\mathrm{abs}}=q_{i}^{\top} k_{j} \\ =\left(W_{q}\left(E_{x_{i}}+U_{i}\right)\right)^{T} \cdot\left(W_{k}\left(E_{x_{j}}+U_{j}\right)\right)
\\ =\underbrace{\mathbf{E}_{x_{i}}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k} \mathbf{E}_{x_{j}}}_{(a)}+\underbrace{\mathbf{E}_{x_{i}}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k} \mathbf{U}_{j}}_{(b)}+\underbrace{\mathbf{U}_{i}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k} \mathbf{E}_{x_{j}}}_{(c)}+\underbrace{\mathbf{U}_{i}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k} \mathbf{U}_{j}}_{(d)}
$$

其中，$E_{x_i}$是词$i$的词向量，$U_i$是对应的位置向量。

而在Transformer-XL中

$$
\mathbf{A}_{i, j}^{\mathrm{rel}}=\underbrace{\mathbf{E}_{x_{i}}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k, E} \mathbf{E}_{x_{j}}}_{(a)}+\underbrace{\mathbf{E}_{x_{i}}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k, R} \mathbf{R}_{i-j}}_{(b)}+\underbrace{u^{\top} \mathbf{W}_{k, E} \mathbf{E}_{x_{j}}}_{(c)}+\underbrace{v^{\top} \mathbf{W}_{k, R} R_{i-j}}_{(d)}
$$

对比来看，主要有三点变化：

* 在(b)和(d)这两项中，将所有绝对位置向量$U_j$都转为相对位置向量$R_{i-j}$，与Transformer一样，这是一个固定的编码向量，不需要学习。
* 在(c)这一项中，将查询的$U_i^TW_q^T$向量转为一个需要学习的参数向量$u$，因为在考虑相对位置的时候，不需要查询的绝对位置$i$，因此对于任意的$i$，都可以采用同样的向量。同理，在(d)这一项中，也将查询的$U_i^TW_q^T$向量转为另一个需要学习的参数向量$v$。
* 将键的权重$W_k$变换矩阵转为$W_{k,E}$和$W_{k,R}$分别作为content-based key vectors和location-based key vectors。
从另一个角度来解读这个公式的话，可以将attention的计算分为如下四个部分：

a. 基于内容的“寻址”，即没有添加原始位置编码的原始分数。
b. 基于内容的位置偏置，即相对于当前内容的位置偏差。
c. 全局的内容偏置，用于衡量key的重要性。
d. 全局的位置偏置，根据query和key之间的距离调整重要性。

## 整体计算过程

$$
For \ n=1, \ldots, N : \quad \tilde{\mathbf{h}}_{\tau}^{n-1}=\left[\mathrm{SG}\left(\mathbf{m}_{\tau}^{n-1}\right) \circ \mathbf{h}_{\tau}^{n-1}\right]
$$

$$
\mathbf{q}_{\tau}^{n}, \mathbf{k}_{\tau}^{n}, \mathbf{v}_{\tau}^{n}=\mathbf{h}_{\tau}^{n-1} \mathbf{W}_{q}^{n \top}, \widetilde{\mathbf{h}}_{\tau}^{n-1} \mathbf{W}_{k, E}^{n}, \mathbf{\tilde { h }}_{\tau}^{n-1} \mathbf{W}_{v}^{n \top}
$$

$$
\mathbf{A}_{\tau, i, j}^{n}={\mathbf{q}_{\tau, i}^{n}}^T\mathbf{k}_{\tau, j}^{n}+{\mathbf{q}_{\tau, i}^{n}}^T\mathbf{W}_{k, R}^{n} \mathbf{R}_{i-j}+u^{\top} \mathbf{k}_{\tau, j}+v^{\top} \mathbf{W}_{k, R}^{n} \mathbf{R}_{i-j}
$$

$$
\mathbf{a}_{\tau}^{n}=\operatorname{Masked}-\operatorname{Softmax}\left(\mathbf{A}_{\tau}^{n}\right) \mathbf{v}_{\tau}^{n}
$$

$$
\mathbf{o}_{\tau}^{n}=\text { LayerNorm (Linear }\left(\mathbf{a}_{\tau}^{n}\right)+\mathbf{h}_{\tau}^{n-1} )
$$

$$
\mathbf{h}_{\tau}^{n}=\text { Positionwise-Feed-Forward }\left(\mathbf{o}_{\tau}^{n}\right)
$$