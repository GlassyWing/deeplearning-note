# 注意力模型

通过对时间步分配不同的权重来进行注意力的分配。该模型对于长序列有着更高的鲁棒性。

## 引入注意力机制的机器翻译模型

## 模型属性

* 有两层分离的LSTM，因为在一层双向LSTM后紧接着注意力机制，这一层LSTM称为前注意力Bi-LSTM。最后一层LSTM在注意力机制之后，称为后LSTM层

## 模型架构

<img src="../imgs/attn_model.png"></img>
<img src="../imgs/attn_mechanism.png"></img>

## 计算步骤

使用前Bi-LSTM层得到a：

$$ a: (m, T_x, 2N_a) $$

将$ s^{<t-1>} $重复$T_x$次：

$$ s^{<t-1>}: (m, N_s) \xrightarrow{Repeat} (m, T_x, N_s) $$

组合变量a和$ s^{<t-1>} $得到e:

$$ e: (m, T_x, 2N_a + N_s) $$

使用一个小型神经网络计算注意力$ \alpha $:

1. 使用Dense层，激活函数使用tanh，包含10个隐藏单元

2. 使用Dense层，激活函数使用relu，包含1个隐藏单元

3. 使用Softmax层，计算维度选为1

$$ \alpha: (m, T_x, 1) $$

将$ \alpha $与 $ a $按照维度1进行点积操作得到Context，注意点积顺序：

$$ context=\alpha \cdot a = \alpha^Ta: (m, 1, 2N_a) $$

将以上步骤重复$T_y$次，得到最终输出：

$$ out: (m, T_y, 2N_a) $$