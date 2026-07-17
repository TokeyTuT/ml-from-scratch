# 03 Logistic Regression 实现指引

## 目标

从零实现二分类逻辑回归，理解线性打分、Sigmoid、交叉熵和分类阈值。

## 建议文件

- `logistic_regression.py`：核心模型类。
- `demo.py`：构造二分类数据并预测。
- `test_logistic_regression.py`：验证概率范围、分类结果、损失下降。

## 建议接口

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        ...

    def fit(self, X, y):
        ...

    def predict_proba(self, X):
        ...

    def predict(self, X):
        ...
```

## 核心公式

线性部分：

```text
z = Xw + b
```

Sigmoid：

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

概率：

```text
p = sigmoid(z)
```

二元交叉熵：

```text
loss = -mean(y * log(p) + (1 - y) * log(1 - p))
```

梯度：

```text
dw = 1 / n * X.T @ (p - y)
db = 1 / n * sum(p - y)
```

## 实现步骤

1. 初始化权重和偏置。
2. 实现 Sigmoid 函数。
3. 在训练循环中计算概率。
4. 根据 `p - y` 计算梯度。
5. 更新参数。
6. `predict_proba` 返回概率。
7. `predict` 使用阈值 `0.5` 返回 0 或 1。

## 注意点

- `log(0)` 会导致数值问题，可以给概率加一个很小的 `epsilon`。
- 标签先只支持 0 和 1。
- 特征尺度差异很大时，最好先标准化。

## 最小验收

- `predict_proba` 的结果必须在 0 到 1 之间。
- 在线性可分数据上，训练后准确率应明显高于随机猜测。
