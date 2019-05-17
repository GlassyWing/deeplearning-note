# 卷积神经网络

## RNN 网络

$$ a^{<t>} = g_1(W_{aa}a^{<t-1>} + W_{ax}x^{<t>} + b_a) \tag{1} $$

$$ \hat{y}^{<t>} = g_2(W_{ya}a^{<t>} + b_y) \tag{2} $$

以上公式简化为：

$$ a^{<t>} = g_1(W_a\theta + b_a) \tag{1} $$

$$ \hat{y}^{<t>} = g_2(W_{y}a^{<t>} + b_y) \tag{2} $$

其中：

$$ W_a = \begin{bmatrix} W_{aa} & W_{ax} \end{bmatrix} $$

$$ \theta = \begin{bmatrix} a^{<t-1>} \\ x^{<t>}\end{bmatrix} $$

## GRU单元

C名为记忆细胞，门控单元可以有效抑制梯度消失。

设 $ c^{<t>} = a^{<t>} $

$$ \Gamma_{r} = \sigma(W_r \theta + b_r) $$

$$ \widetilde{c}^{<t>} = tanh(W_c [\Gamma_{r} * c^{<t-1>}, x^{<t>}] + b_c)  $$

$$ \Gamma_{u} = \sigma(W_u \theta + b_u) $$

$$ c^{<t>} = \Gamma_{u} * \widetilde{c}^{<t>} + (1 - \Gamma_{u}) * c^{<t-1>} $$

其中：

$$ \theta = \begin{bmatrix} c^{<t-1>} \\ x^{<t>}\end{bmatrix} $$

## LSTM单元

$$ \widetilde{c}^{<t>} = tanh(W_c \theta + b_c)  $$

$$ \Gamma_{u} = \sigma(W_u \theta + b_u) \tag{更新门} $$

$$ \Gamma_{f} = \sigma(W_f \theta + b_r) \tag{遗忘门} $$

$$ \Gamma_{o} = \sigma(W_o \theta + b_o) \tag{输出门} $$

$$ c^{<t>} = \Gamma_{u} * \widetilde{c}^{<t>} + \Gamma_{f} * c^{<t-1>} $$

$$ a^{<t>} = \Gamma_{o} * tanh(c^{<t>}) $$

其中：

$$ \theta = \begin{bmatrix} a^{<t-1>} \\ x^{<t>}\end{bmatrix} $$

## 各参数维度说明

### 标准1

批次数在后，即m在后：

$ W_a : (n_a, n_a) $
$ W_x : (n_a, n_x) $
$ W_y : (n_y, n_a) $
$ a: (n_a, m) $
$ X: (n_x, m, T_x) $
$ Y: (n_y, m, T_x) $

### 标准2

批次数在前，即m在前

$ W_a : (n_a, n_a) $
$ W_x : (n_a, n_x) $
$ W_y : (n_y, n_a) $
$ a: (m, n_a) $
$ X: (m, T_x, n_x) $
$ Y: (m, T_x, n_y) $

**注意**：此时执行计算时，参数W应该进行转置