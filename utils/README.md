# Utils 实现指引

## 目标

`utils/` 用于存放多个算法都会用到的通用工具。目前包含：

- 模型的顶级父类: `BaseEstimator`, 位于 `base_estimator.py` 

- 数据集划分：`train_test_split`
- 分类评估：`accuracy_score`
- 回归评估：`mean_absolute_error`、`mean_squared_error`、`root_mean_squared_error`、`r2_score`
- 特征归一化：`MinMaxScaler`
- 特征标准化：`StandardScaler`

## 数据集划分

实现文件：[model_selection.py](model_selection.py)

核心函数：

```python
train_test_split(X, y, test_size=0.2, shuffle=True, random_seed=None)
```

参数说明：

- `X`：二维特征矩阵，形状为 `(n_samples, n_features)`。
- `y`：一维标签或目标值数组，形状为 `(n_samples,)`。
- `test_size`：测试集比例，取值范围为 `[0, 1]`。
- `shuffle`：是否在划分前打乱样本。
- `random_seed`：随机种子，用于复现实验结果。

## Min-Max 归一化

实现文件：[min_max_scaler.py](min_max_scaler.py)

Min-Max 归一化会把原始数据映射到指定区间，默认区间为 `[0, 1]`。

$$
X' = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

如果目标区间为 `[a, b]`，则继续做线性变换：

$$
X'' = X' \times (b - a) + a
$$

注意点：

- 该方法容易受到最大值和最小值影响。
- 当某个特征的 `max - min = 0` 时，代码会把分母替换为 1，避免除零错误。

## Z-Score 标准化

实现文件：[standard_scaler.py](standard_scaler.py)

标准化会把每个特征转换为均值为 0、标准差为 1 的形式。

$$
X' = \frac{X - mean}{\sigma}
$$

其中，`mean` 是当前特征列的均值，`\sigma` 是当前特征列的标准差。

适用场景：

- KNN、线性回归、逻辑回归、PCA、神经网络等对特征尺度敏感的算法。
- 特征量纲不同、数值范围差异较大的数据集。

## 模型评估

实现文件：[metrics.py](metrics.py)

分类指标：

- `accuracy_score(y_true, y_pred)`：准确率。

回归指标：

- `mean_absolute_error(y_true, y_pred)`：平均绝对误差。
- `mean_squared_error(y_true, y_pred)`：均方误差。
- `root_mean_squared_error(y_true, y_pred)`：均方根误差。
- `r2_score(y_true, y_pred)`：决定系数。

常用公式：

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

$$
RMSE = \sqrt{MSE}
$$

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

## 交叉验证与网格搜索

交叉验证和网格搜索通常配合使用：

1. 网格搜索枚举多组超参数。
2. 每组超参数通过交叉验证评估。
3. 选择平均验证得分最高的一组参数。

### 交叉验证
这里采用经典的 KFlod 方法来实现交叉验证
> KFold 就是 K 折交叉验证 里的“切数据方法”。
> 它的核心思想是：
> 不要只做一次训练集/验证集划分，而是把数据平均分成 K 份，每次拿其中 1 份做验证集，剩下 K-1 份做训练集，重复 K 次。
见  `model_selection.py` 中的 `KFlod` 方法
### 网格搜索
