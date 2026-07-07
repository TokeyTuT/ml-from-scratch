# 04 Decision Tree 实现指引

## 目标

从零实现分类决策树，理解递归划分、纯度指标和树结构预测。

## 建议文件

- `decision_tree.py`：核心模型类和节点结构。
- `demo.py`：用简单二维数据训练树并预测。
- `test_decision_tree.py`：验证划分、叶子节点、预测路径。

## 建议接口

```python
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        ...

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...
```

## 建议数据结构

```python
class Node:
    feature_index = ...
    threshold = ...
    left = ...
    right = ...
    value = ...
```

叶子节点只需要保存 `value`，内部节点保存划分特征、阈值和左右子树。

## 核心概念

Gini impurity：

```text
Gini = 1 - sum(p_k^2)
```

划分收益：

```text
gain = parent_gini - weighted_child_gini
```

## 实现步骤

1. 实现多数类函数，用于叶子节点预测。
2. 实现 Gini 计算。
3. 枚举每个特征和候选阈值，寻找最佳划分。
4. 递归构建左右子树。
5. 设置停止条件：
   - 达到最大深度。
   - 样本数小于 `min_samples_split`。
   - 当前节点所有标签相同。
   - 找不到有效划分。
6. 预测时，从根节点按阈值一路走到叶子节点。

## 注意点

- 一开始只做分类树，不做回归树。
- 阈值可以先直接用特征中出现过的唯一值。
- 递归时注意空子集。
- 树很容易过拟合，`max_depth` 很重要。

## 最小验收

- 能在简单逻辑规则数据上学出正确划分。
- 预测结果数量和输入样本数量一致。
- 单类别数据能直接生成叶子节点。
