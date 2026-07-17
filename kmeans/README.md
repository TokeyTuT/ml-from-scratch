# 06 K-Means 实现指引

## 目标

从零实现 K-Means 聚类，理解无监督学习中的质心更新和迭代收敛。

## 建议文件

- `kmeans.py`：核心模型类。
- `demo.py`：生成二维点并可视化聚类结果。
- `test_kmeans.py`：验证标签数量、质心形状、收敛行为。

## 建议接口

```python
class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        ...

    def fit(self, X):
        ...

    def predict(self, X):
        ...

    def fit_predict(self, X):
        ...
```

## 核心流程

1. 随机选择 `k` 个样本作为初始质心。
2. 计算每个样本到所有质心的距离。
3. 把每个样本分配到最近的质心。
4. 对每个簇重新计算均值，作为新质心。
5. 如果质心变化很小，或者达到最大迭代次数，则停止。

## 注意点

- `n_clusters` 不能大于样本数量。
- 某个簇可能没有样本，需要处理空簇。
- 可以用 `random_state` 固定随机结果，方便调试。
- K-Means 对初始点敏感。

## 最小验收

- `centroids` 的形状应为 `(n_clusters, n_features)`。
- `predict` 返回每个样本的簇编号。
- 对明显分成几团的二维数据，聚类结果应合理。
