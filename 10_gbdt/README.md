# 10 GBDT 实现指引

## 目标

从零实现一个基础 GBDT 回归模型，理解加法模型、残差拟合和梯度提升过程。

## 建议文件

- `gbdt.py`：GBDT 回归模型。
- `regression_tree.py`：GBDT 使用的回归树。
- `demo.py`：在简单回归数据上验证模型。
- `test_gbdt.py`：验证残差、预测累加和损失下降。

## 建议接口

```python
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        ...

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...
```

## 实现顺序

1. 使用目标值的均值初始化模型预测。
2. 计算当前预测与真实值之间的残差。
3. 使用一棵回归树拟合残差。
4. 按照学习率将回归树预测加入当前模型。
5. 重复计算残差并训练新的回归树。
6. 将初始预测和所有树的预测相加。

## 最小验收

- 每棵树都拟合上一轮产生的残差。
- 训练过程中均方误差整体下降。
- `predict` 能返回形状为 `(n_samples,)` 的结果。
