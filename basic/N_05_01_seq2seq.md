# seq2seq模型

## 定向搜索误差分析

为了确定是RNN网络本身问题还是束搜索算法问题：

1. 获得人类翻译结果和算法翻译结果，例如：
    - Human: Jane visits Africa in September. $ y^* $
    - Algorithm: Jane visited Africa last September. $ \hat{y} $

2. 使用RNN（不用束方法）计算出人类翻译结果和算法翻译结果的概率：
   - $ P(y^*|x) $
   - $ P{(\hat{y}|x)} $
  
3. 比较两个概率值的大小
   - $ P(y^*|x) > P{(\hat{y}|x)} $ 说明束方法出错了
   - $ P(y^*|x) \leq P{(\hat{y}|x)} $ RNN 网络问题

多次执行1 - 3 步骤，找出两种错误所占比重，确定最终的问题

## Bleu得分
