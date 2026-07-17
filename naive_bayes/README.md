# 05 Naive Bayes 实现指引

## 目标

从零实现 Gaussian Naive Bayes，理解贝叶斯公式、类先验和条件独立假设。

## 建议文件

- `naive_bayes.py`：核心模型类。
- `demo.py`：用连续特征数据做分类。
- `test_naive_bayes.py`：验证均值方差、先验概率、预测结果。

## 建议接口

```python
class GaussianNaiveBayes:
    def fit(self, X, y):
        ...

    def predict(self, X):
        ...
```

## 核心公式

贝叶斯分类：

```text
argmax_y P(y) * P(x_1 | y) * P(x_2 | y) * ... * P(x_d | y)
```

高斯概率密度：

```text
P(x_i | y) = 1 / sqrt(2 * pi * var) * exp(-(x_i - mean)^2 / (2 * var))
```

实际实现中建议用 log 防止连乘下溢：

```text
log posterior = log prior + sum(log likelihood)
```

## 实现步骤

1. 找出所有类别。
2. 对每个类别计算先验概率。
3. 对每个类别、每个特征计算均值和方差。
4. 预测时分别计算样本属于每个类别的 log posterior。
5. 选择 log posterior 最大的类别。

## 注意点

- 方差为 0 时要加一个很小的平滑值。
- 先只支持连续特征。
- 类别标签可以是数字或字符串。

## 最小验收

- 能保存每个类别的均值、方差、先验概率。
- 能对多个样本批量预测。
- 在两个高斯分布明显分开的数据上分类正确率较高。
