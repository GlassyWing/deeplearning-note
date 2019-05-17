# 提升神经网络

## 正则化

### L2正则化

代价函数：

$$J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} $$

反向传播时需加上正则项：

$$ d_{W} = ... + \underbrace{\frac{\lambda}{m} * W}_\text{L2 regularization} $$

### Dropout 正则化

前向传播与反向传播都必须使用相同的掩码，同时都需除以keep_prob

## 小批量处理

### 定义

为了应对内存限制而采取的措施，每次训练数据集中的一小部分，以`batch_size`表示：

1. batch_size == m：

    此时就是批量处理

2. batch_size == 1:

    此时变为随机梯度下降，无法利用向量化的优势，且一般无法达到最优解

常用的`batch_size`大小为$2^n$，视计算机内存大小而定

### 滑动平均

采用小批量处理时，每次迭代的梯度并不是依次下降的，但具有下降的趋势，有波动性。为了加快梯度下降，使用滑动平均算法。

## 各种优化算法

### 动量梯度下降

初始化$ V_{d_w} = 0, V_{d_b} = 0 $：
每次迭代：

 $$ V_{d_w} = \beta V_{d_w} + (1 - \beta) d_w, V_{d_b} = \beta V_{d_b} + (1 - \beta) d_b $$

 $$ W = W - \alpha V_{d_w}, b = b - \alpha V_{d_b} $$

该方法类似于滑动平均，适用于小批量处理（但也适用于批量处理）。

### RMSprop

初始化 $ S_{d_w} = 0, S_{d_b} = 0 $

每次迭代：

$$ S_{d_w} = \beta S_{d_w} + ( 1 - \beta) d_w^2, S_{d_b} = \beta S_{d_b} + (1 - \beta) d_b^2 $$

$$ W = W - \alpha \frac{d_w}{\sqrt{S_{d_w} + \epsilon}}, b = b - \alpha \frac{d_b}{\sqrt{S_{d_b}+ \epsilon}} $$

该方法本质上进一步降低了参数W和b的噪声

### Adam (自适应距估计)

该方法融入了动量下降和RMSprop的优点。

初始化：$V_{d_w} =0, S_{d_w} = 0, V_{d_b} = 0, S_{d_b} = 0$

每次迭代：

$$ V_{d_w} = \beta_{1} V_{d_w} + (1 - \beta_{1}) d_w, V_{d_b} = \beta_{1} V_{d_b} + (1 - \beta_{1}) d_b $$

$$ S_{d_w} = \beta_{2} S_{d_w} + (1 - \beta_{2}) d_w^2, S_{d_b} = \beta_{2} S_{d_b} + (1 - \beta_{2}) d_b^2 $$

$$ V_{d_w}^{corrected} = \frac{V_{d_w}}{1-\beta_{1}^t}, V_{d_b}^{corrected} = \frac{V_{d_b}}{1-\beta_{1}^t} $$

$$ S_{d_w}^{corrected} = \frac{S_{d_w}}{1-\beta_{2}^t}, s_{d_b}^{corrected} = \frac{S_{d_b}}{1 - \beta_{2}^t} $$

$$ W = W - \alpha \frac{V_{d_w}^{corrected}}{\sqrt{S_{d_w}^{corrected} + \epsilon}}, b = b - \alpha \frac{ V_{d_b}^{corrected}}{\sqrt{s_{d_b}^{corrected}+ \epsilon}} $$

惯用参数值：

$ \alpha: $ 需要调整
$ \beta_{1}: 0.9 $
$ \beta_{2}: 0.999$
$ \epsilon: 10^{-8}$

## 学习率衰减

在训练开始时使用较大的学习率，而在训练后期慢慢减小学习率，使最终的结果不会离最优解漂离太远。

常用衰减公式：

1. :
  
    $$ \alpha = \frac{1}{1 + decay\_rate * epoch\_num} \alpha_{0} $$

2. :
  
   $$ \alpha = decay\_rate^{epoch\_num} {\alpha_{0}} $$

## 超参数调优

### 超参数优先级

| 参数                | 优先级1 | 优先级2 | 优先级3 | 默认  |
| ------------------- | ------- | ------- | ------- | ----- |
| $ \alpha $          | ✓       |         |         |       |
| $ \beta $           |         | ✓       |         |       |
| $ \beta_{1} $       |         |         |         | 0.9   |
| $ \beta_{2} $       |         |         |         | 0.999 |
| $ \epsilon $        |         |         | ✓       |       |
| hidden_units        |         | ✓       |         |       |
| mini-batch size     |         | ✓       |         |       |
| layers              |         |         | ✓       |       |
| learning rate decay |         |         | ✓       |       |

### 超参数选择

1. 对超参数空间随机采样
2. 限定超参数空间区域，在该区域进行密集采样

### 参数采样

#### $ \alpha $的采样

对于layers这样的参数使用均匀随机抽样很合理，但对于$ \alpha $，有可能想要从`0.0001 - 1`之间进行抽样，这样将会有90%的可能抽取的参数位于`0.1 - 1`间。对于这样的参数，采用对数尺度进行搜索：

设：我们希望平均的分配概率在区间`0.0001--0.001, 0.001--0.01, 0.01--0.1, 0.1--1`上,使用python:

```python
r = -4 * np.random.rand() # [-4, 0)
alpha = 10 ** r # [10 ** -4, 1)
```

更一般的：$$ a = \log_{10}^{0.0001} = -4, b = \log_{10}^1 = 0 $$

#### $ \beta $的采样

设$ \beta \in [0.9, 0.999] $
若$ \beta = 0.9 $,则相当于对最近10个数据取平均，而若$ \beta = 0.999 $, 则相当于对最近1000个数据取平均，那么使用线性尺度进行采样并不合理，处理方式如下。

$ 1 - \beta \in [0.001, 0.1] $

再次使用对$ \alpha $ 抽样相似的方式进行抽样：

```python
r = -2 * np.random.rand() -1 # [-3, -1)
# 1 - beta = 10 ** r
beta = 1 - 10 ** r
```

## 批量归一化

对每一层的激活量进行归一化。
单层归一化公式：

$$ \mu = \frac{1}{m} \sum_{i}z^{(i)} $$

$$ \sigma^2 = \frac{1}{m} \sum_{i}(z^{(i)} - \mu)^2 $$

$$ z_{norm}^{(i)} = \frac{z^{(i) - \mu}}{\sqrt{\sigma^2 + \epsilon}} $$

$$ \widetilde{z}^{(i)} = \gamma z_{norm}^{(i)} + \beta $$

**注意**：此处的$ \beta $ 与 动量梯度下降算法中的 $ \beta$ 不同。