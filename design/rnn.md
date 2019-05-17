# RNN 模型设计经验

## 与Dropout的结合

1. 在每层LSTM之后加上一层Dropout，dropout_rate 从0.3左右调整，可以防止局部过拟合，加快收敛速度。
2. 在注意力层前后加入Dropout，效果不大
3. 在多层LSTM之间加入Dropout，效果不大

## 参数

1. LSTM单元数量一般从128左右进行调整

## 经验

1. 总是从较大的模型开始调整，并测试不同的Dropout率