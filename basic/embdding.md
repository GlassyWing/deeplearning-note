# 词嵌入

## 如何学习

词嵌入实际上是一个矩阵，假设有10000个词，每个词有300维的特征，那么词嵌入矩阵E的维度为 (10000, 300) 或 (300, 10000)，这里以后者为例：

1. 通过前面几个词来预测后一个词
   
   将前面几个词的词向量堆叠到一起并输入到Dense层中，最后使用softmax输出预测值进行训练。具体需要几个词可以通过设置窗口值大小确定。

2. 通过前面和后面几个词预测中间的词
   
   方法和1类似，只是选择的词不同

3. Word2Vec算法
   
   随机选择一个词，并预测这个词正负s个词距内的其它词

4. 通过分词任务学习词嵌入
   
   通过目前最先进的分词模型可以学到很好的语法特性