# 07 PCA 实现指引

## 目标

从零实现 PCA，理解数据中心化、协方差矩阵、特征值分解和线性降维。

## 建议文件

- `pca.py`：核心降维类。
- `demo.py`：把二维或三维数据降到低维并可视化。
- `test_pca.py`：验证输出形状、均值中心化、主成分方向。

## 建议接口

```python
class PCA:
    def __init__(self, n_components):
        ...

    def fit(self, X):
        ...

    def transform(self, X):
        ...

    def fit_transform(self, X):
        ...
```

## 核心步骤

1. 计算每个特征的均值。
2. 对数据做中心化：

   ```text
   X_centered = X - mean
   ```

3. 计算协方差矩阵。
4. 对协方差矩阵做特征值分解。
5. 按特征值从大到小排序。
6. 取前 `n_components` 个特征向量作为主成分。
7. 用中心化后的数据乘以主成分矩阵完成降维。

## 注意点

- PCA 是无监督方法，不需要标签 `y`。
- `n_components` 不能大于特征数量。
- 特征值越大，表示该方向解释的方差越多。
- 可以额外实现 `explained_variance_ratio_`。

## 最小验收

- 输入 `(n_samples, n_features)`，输出 `(n_samples, n_components)`。
- `fit` 后保存 `mean_` 和 `components_`。
- 在强线性相关数据上，第一个主成分应解释主要方差。
