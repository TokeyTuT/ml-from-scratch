# 09 AdaBoost 实现指引

## 目标

从零实现二分类 AdaBoost，理解样本权重更新、弱分类器权重和加权投票。

## 建议文件

- `adaboost.py`：AdaBoost 分类器。
- `decision_stump.py`：决策树桩弱分类器。
- `demo.py`：在简单二分类数据上验证模型。
- `test_adaboost.py`：验证权重更新和预测结果。

## 建议接口

```python
class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        ...

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...
```

## 实现顺序

1. 将二分类标签转换为 `-1` 和 `+1`。
2. 初始化所有样本的权重。
3. 根据当前样本权重训练决策树桩。
4. 计算弱分类器的加权错误率和投票权重。
5. 更新并归一化样本权重。
6. 对多个弱分类器的结果进行加权投票。

## 最小验收

- 每轮更新后样本权重之和为 1。
- 错分样本的权重能够提高。
- 在简单二分类数据上，组合模型优于单个决策树桩。
