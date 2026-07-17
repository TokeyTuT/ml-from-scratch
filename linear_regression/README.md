# 02 Linear Regression 实现指引

## 目标

从零实现线性回归，理解线性模型、均方误差和梯度下降。

## 建议文件

- `linear_regression.py`：核心模型类。
- `demo.py`：拟合一条直线或多维线性数据。
- `test_linear_regression.py`：验证参数收敛和预测结果。

## 建议接口

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        ...

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...
```

## 核心公式

预测函数：

```text
y_hat = Xw + b
```

均方误差：

```text
MSE = mean((y_hat - y)^2)
```

梯度：

```text
dw = 2 / n * X.T @ (y_hat - y)
db = 2 / n * sum(y_hat - y)
```

参数更新：

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

## 实现步骤

1. 初始化权重 `w` 为 0，偏置 `b` 为 0。
2. 循环训练 `epochs` 次。
3. 每轮计算预测值。
4. 计算误差和梯度。
5. 更新权重和偏置。
6. 实现 `predict` 返回连续值预测。

## 注意点

- `X` 建议统一处理成二维矩阵。
- `y` 建议处理成一维向量。
- 学习率过大可能发散。
- 可以记录 `loss_history` 观察训练过程。

## 最小验收

- 用 `y = 2x + 1` 生成数据，训练后参数应接近 `w=2, b=1`。
- 预测值和真实值的误差应随着训练下降。
