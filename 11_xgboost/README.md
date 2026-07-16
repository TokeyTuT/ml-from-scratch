# 11 XGBoost 实现指引

## 目标

在理解 GBDT 的基础上，实现一个简化版 XGBoost 回归模型，理解一阶梯度、二阶梯度和正则化叶子权重。

## 建议文件

- `xgboost.py`：简化版 XGBoost 回归模型。
- `xgb_tree.py`：使用梯度信息进行分裂的回归树。
- `demo.py`：在简单回归数据上验证模型。
- `test_xgboost.py`：验证梯度、分裂增益和预测结果。

## 建议接口

```python
class XGBoostRegressor:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        reg_lambda=1.0,
        gamma=0.0,
    ):
        ...

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...
```

## 实现顺序

1. 先完成并理解基础 GBDT 回归模型。
2. 根据当前预测计算损失函数的一阶梯度和二阶梯度。
3. 使用梯度统计量计算叶子节点的预测值。
4. 计算候选切分带来的增益。
5. 使用正则化参数限制树的复杂度。
6. 按照学习率累加每棵树的预测结果。

## 最小验收

- 能正确计算平方误差的一阶梯度和二阶梯度。
- 只有分裂增益大于阈值时才创建子节点。
- 训练过程中损失整体下降。
