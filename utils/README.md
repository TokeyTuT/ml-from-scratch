# UTILS 实现指引

 ## 目标
   总体目标：实现机器学习中几个常用的工具函数：
   * 数据集划分
   * 模型评估（模型准确度）
   * 归一化函数
   * 数据标准化函数
   * 交叉验证的函


## 数据集划分
位于文件 `train_test_split.py` 下, 方法为：`train_test_split(X,y,test_size = 0.2,shuffle = True,random_seed = None)` 
实现细节见注释。

## 数据的归一化
数据的预处理很重要的一步就是**数据的归一化**，目的是把原始数据映射到\[min,max\]之间

参考公式
$$
X'=\frac {x - min} {max - min}
$$
$$
               X''=X'(max - min) + min
$$

*   $X'$–>基于公式算出来的结果

*   $X''$–>最终的结果

*   *min*–> 区间最小值

*   *max*–> 区间最大值

**弊端**：
容易受到最大值和最小值的影响，所以他一般用于处理 **小数据集**


见文件  `min_max_scaler.py` 

代码鲁棒性：
* 这里加了判断 $max - min$ 是否等于 0，如果等于 0，那么我们把他置为 1，防止相除的时候出现异常值。

## 数据标准化

通过对原始数据进行标准化，转换为均值为 0 和标准差为 1 的标准正态分布的的数据
$$
X'=\frac{x-mean}{\sigma}
$$
*mean*为当前列的平均值，*σ* 代表当前列的标准差

场景： 
	适用于 **大数据集** 的处理


## 数据的评估 `metrics.py`
* 对于分类模型，采用 `accucacy_score(y_true,y_pred)` 对模型打分
* 对于回归模型，采用决定系数 $R^2$ `r2_score(y_true,y_pred)` 对模型打分
* 此外，提供了 MAE、MSE、RMSE 的实现

1. 均方误差 (MSE, Mean Squared Error)
  $$
  MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$
2. 平均绝对误差 (MAE, Mean Absolute Error
   )
   $$
   MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
   $$
3. 均方根误差 (RMSE, Root Mean Squared Error)
   $$
   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{MSE}
   $$
4. 决定系数 ($R^2$, Coefficient of Determination)
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$