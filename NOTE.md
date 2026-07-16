# 学习笔记

## `axis` 参数

在 NumPy 和 Pandas 中，`axis` 表示操作沿着哪条轴进行：

- `axis=0`：沿行方向聚合，通常得到“每一列”的结果。
- `axis=1`：沿列方向聚合，通常得到“每一行”的结果。

以二维数组为例：

```text
shape = (n_samples, n_features)
axis=0 -> 对样本维度做聚合，结果长度通常等于 n_features
axis=1 -> 对特征维度做聚合，结果长度通常等于 n_samples
```

## 数据标准化与归一化

### 为什么需要处理特征尺度

很多机器学习算法会直接使用特征之间的距离或梯度。如果不同特征的数值范围差异很大，量纲较大的特征会主导模型。

以 KNN 为例，假设用“年龄”和“收入”衡量用户相似度：

```text
d = sqrt((年龄差)^2 + (收入差)^2)
```

如果收入的数值范围远大于年龄，距离几乎会被收入决定，年龄特征的影响会被削弱。因此，在 KNN、线性模型、PCA、神经网络等算法中，通常需要先处理特征尺度。

### Min-Max 归一化

Min-Max 归一化会把数据映射到指定区间，常见区间是 `[0, 1]`。

$$
x_{new} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

特点：

- 输出有明确上下界。
- 对最大值和最小值敏感。
- 如果存在极端异常值，普通样本可能被压缩到很小的区间。

适用场景：

- 图像像素值缩放。
- 需要固定取值范围的特征。
- 数据异常值较少的小型数据集。

实现文件：[utils/min_max_scaler.py](utils/min_max_scaler.py)

### Z-Score 标准化

Z-Score 标准化会把数据转换为均值为 0、标准差为 1 的形式。

$$
x_{new} = \frac{x - \mu}{\sigma}
$$

其中，$\mu$ 表示均值，$\sigma$ 表示标准差。

特点：

- 不强制限制输出范围。
- 比 Min-Max 归一化更不依赖极值。
- 常用于线性回归、逻辑回归、SVM、PCA 和神经网络。

实现文件：[utils/standard_scaler.py](utils/standard_scaler.py)

## 交叉验证

交叉验证是一种模型评估方法。它会把数据划分为 `n` 份，每次取其中一份作为验证集，其余 `n - 1` 份作为训练集，重复训练和评估后取平均分。

基本流程：

1. 第一次：第 1 份作为验证集，其余数据作为训练集。
2. 第二次：第 2 份作为验证集，其余数据作为训练集。
3. 重复直到每一份都做过验证集。
4. 计算多次验证得分的平均值，作为模型的整体评估结果。

## 网格搜索

网格搜索用于选择超参数。超参数是训练前需要人为指定的参数，例如 KNN 中的 `k`。

基本流程：

1. 枚举多组超参数组合。
2. 对每一组超参数进行交叉验证。
3. 比较平均验证得分。
4. 选择得分最好的超参数组合。

网格搜索适合参数空间较小、希望系统比较不同配置的场景。


### 梯度下降法(*Gradient descent*)

*   什么是梯度

    *   单变量函数中，梯度就是某一点的切线斜率：有方向：函数增长最快的方向
    *   多变量函数中，梯度就是某一个点的偏导数；有方向：偏导数分量的向量方向

*   梯度下降公式:

    循环迭代求当前点的梯度，更新当前的权重参数
    $$
    \theta_{i}:=\theta_{i}-\alpha\frac {\partial}{\partial\theta_i}J(\theta)
    $$
    其中：

    *   $\alpha$ 为学习率（步长）不能太大，也不能太小，机器学习中：$[0.001,0.01]$
    *   梯度是上升最快的方向，我们需要的是下降最快的方向，所以需要加负号

    **这里我我给出一元回归的梯度下降推导过程，对于多元不必掌握**：

    >   设目标函数为 $f(x)$，我们的目的就是为了寻找一个 $x$，使得$f(x)$ 的值取到最小。（注意这里的 x 并不是特征值，而是我们所说的权重，特征值已经被看作常数项或者常系数了）
    >
    >   假设我们当前在 $x_0$ 点，想移动一个微小的距离 $\Delta x$ 到达 $x_0 + \Delta x$。根据一阶泰勒展开公式：
    >   $$
    >   f(x_0 + \Delta x) \approx f(x_0) + f'(x_0) \Delta x
    >   $$
    >   我们的目标是每一次移动$\Delta x$后$f(x)$都更加小也就是$f(x + \Delta x) < f(x)$。
    >
    >   也就是说，$f'(x_0) \Delta x < 0$每一次都要成立。
    >
    >   为了确保这个不等式永远成立，最简单的办法就是令 $\Delta x$ 的符号与导数 $f'(x_0)$ 相反：
    >
    >   -   如果导数 $f'(x_0) > 0$（函数在上升），我们就让 $\Delta x$ 为负（向左走）。
    >   -   如果导数 $f'(x_0) < 0$（函数在下降），我们就让 $\Delta x$ 为正（向右走）。
    >
    >   如果我们令 ：
    >   $$
    >   \Delta x=-\alpha f'(x)
    >   $$
    >   这里的 $\alpha$（学习率）是一个大于 0 的微小常数，用来控制步长
    >
    >   这样，我们就得到了一元函数梯度下降的公式：
    >   $$
    >   x_{nex} = x_{old} - \Delta x = x_{old} - \alpha f'(x)
    >   $$
    >   如果当$f'(x) = 0$ 时，意味着到达了极值点，x 不会再更新，算法停止收敛。
    >
    >   证明完毕。
    >
    >   对于多元线性回归，其实类似，只是把$x$ 换成了向量$w$

![](https://cdn.jsdelivr.net/gh/TokeyTuT/my-image-storage@main/img/image-20260425163633317.png)

![image-20260425163825868](https://cdn.jsdelivr.net/gh/TokeyTuT/my-image-storage@main/img/image-20260425163825868.png)

*   有关学习率：
    *   学习率太小：下降速率太慢
    *   学习率太大：容易错过最低点、产生下降过程中的震荡、甚至梯度爆炸

但是，我们不可能一开始就知道损失函数，那么我们如何确定函数呢？

---

以二元线性回归为例：

假设我们有 $m$ 个样本数据。对于第 $i$ 个样本，特征为 $x_1^{(i)}$ 和 $x_2^{(i)}$，真实值为 $y^{(i)}$。我们的多元线性回归模型（预测函数 $h$）为：
$$
h(x^{(i)}) = w_1 x_1^{(i)} + w_2 x_2^{(i)} + b
$$
由此我们可以导出损失函数$J$，为了后续求导时抵消掉平方项产生的系数 2，我们通常会在前面乘上一个 $\frac{1}{2}$：
$$
J(w_1, w_2, b) = \frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2
$$
现在，我们的目标就是：找到一组 $(w_1, w_2, b)$，使得函数 $J(w_1, w_2, b)$ 的值达到全局最小。

**对 $w_1$ 求偏导：**
$$
\frac{\partial J}{\partial w_1} = \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)}
$$
**对 $w_2$ 求偏导：**

$$
\frac{\partial J}{\partial w_2} = \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)}
$$

**对 $b$ 求偏导：**
$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})
$$
拿到每一组变量的偏导之后，根据梯度下降的规则：$\theta_{i+1}=\theta_{i}-\alpha\frac {\partial}{\partial\theta_i}J(\theta)$

每一次迭代的更新规则如下：

$$
w_1 := w_1 - \alpha \frac{\partial J}{\partial w_1}
$$

$$
w_2 := w_2 - \alpha \frac{\partial J}{\partial w_2}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

初始时，我们不知道$(w_1,w_2,b)$，我们可以任取一个合理的数即可，这对元组会在迭代之后自动收敛

>   我们考虑这个案例：
>
>   假设我们想要预测一个大学生的 **GPA**（目标变量 $y$）。影响 GPA 的因素有很多，我们挑出两个最关键的特征（这就是“多元”的由来）：
>
>   -   $x_1$：每周复习专业课的小时数
>   -   $x_2$：每周在 LeetCode 上刷题的数量
>
>   我们假设它们之间是一个“线性”的关系，也就是可以用一个公式来表达：
>   $$
>   y_{预测} = w_1 x_1 + w_2 x_2 + b
>   $$
>   这符合上面的二元线性回归，现在我们对上面的求到的偏导进行验证。
>
>   假设我们有两个学生的数据（样本量 $m=2$）：
>
>   -   **样本 1**：$x_1^{(1)}=2$（复习 2 小时），$x_2^{(1)}=1$（刷 1 题），真实 GPA $y^{(1)}=3.5$
>   -   **样本 2**：$x_1^{(2)}=3$（复习 3 小时），$x_2^{(2)}=2$（刷 2 题），真实 GPA $y^{(2)}=4.0$
>
>   我们依然使用瞎猜的初始参数和学习率 $\alpha$：
>
>   -   $w_1 = 0.5$
>   -   $w_2 = 0.5$
>   -   $b = 1.0$
>   -   $\alpha = 0.1$
>
>   ------
>
>   ### 代入偏导数公式
>
>   现在，我们把这 2 个样本的误差和特征值，代入到上一回提到的偏导数（梯度）公式中
>
>   **计算 $w_1$ 的偏导数：**
>
>   $$\begin{align*} \frac{\partial J}{\partial w_1} = -1.75 \end{align*}$$
>
>   **计算 $w_2$ 的偏导数：**
>
>   $$\begin{align*} \frac{\partial J}{\partial w_2} &= -1.0 \end{align*}$$
>
>   **计算 $b$ 的偏导数：**
>
>   *(注意：$b$ 后面没有特征 $x$，所以直接对误差求和取平均即可)*
>
>   $$\begin{align*} \frac{\partial J}{\partial b}  &= -0.75 \end{align*}$$
>
>   **偏导数的意义：** 这里算出来的 $-1.75$、$-1.0$ 和 $-0.75$ 就是当前的梯度向量。它综合了两个样本的情况，告诉我们：为了让整体误差变小，$w_1$ 需要向上调整的力度最大，其次是 $w_2$，最后是 $b$。
>
>   ------
>
>   ### 更新参数
>
>   利用梯度下降的更新规则 $w := w - \alpha \frac{\partial J}{\partial w}$，代入学习率 $0.1$：
>
>   -   $w_1 := 0.5 - 0.1 \times (-1.75) = 0.5 + 0.175 = \mathbf{0.675}$
>   -   $w_2 := 0.5 - 0.1 \times (-1.0) = 0.5 + 0.1 = \mathbf{0.6}$
>   -   $b := 1.0 - 0.1 \times (-0.75) = 1.0 + 0.075 = \mathbf{1.075}$
>
>   ------
>
>   ###  检验效果
>
>   我们用刚刚更新好的新参数 $(0.675, 0.6, 1.075)$，重新对这两个样本预测一次，看看误差是不是真的变小了：
>
>   **对于样本 1 (真实值 3.5)：**
>
>   -   新预测值：$0.675 \times 2 + 0.6 \times 1 + 1.075 = 1.35 + 0.6 + 1.075 = \mathbf{3.025}$
>   -   绝对误差从 $1.0$ 缩小到了 $0.475$。
>
>   **对于样本 2 (真实值 4.0)：**
>
>   -   新预测值：$0.675 \times 3 + 0.6 \times 2 + 1.075 = 2.025 + 1.2 + 1.075 = \mathbf{4.3}$
>   -   绝对误差从 $0.5$ 缩小到了 $0.3$。
>
>   **结论：** 经过仅仅一次严谨的偏导数计算和参数更新，**所有样本的预测误差都显著缩小了**。在实际应用中，计算机就是通过高效的矩阵运算，把这套流程在一秒钟内跑上几万次，最终找到那组让误差无限趋近于最小值的最优解。



---

于是，我们可以导出多元线性回归方程的梯度下降法求解过程：

如果我们有 $n$ 个特征值，一共有 $m$ 个样本数的情况下

为了计算方便、让过程具有统一性，我们定义偏差 $b$，那一项代表$1*b$，记$x^{(0)} = 1$

对于**第 $i$ 个样本**，它的预测公式为：

$$
h(x^{(i)}) = w_1 x_1^{(i)} + w_2 x_2^{(i)} + \dots + w_n x_n^{(i)} + b
$$
对应的误差函数：
$$
J(w_0, w_1, \dots, w_n) = \frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2
$$

如果我们对$w_j$求偏导
$$
\frac{\partial J}{\partial w_j}=\frac 1m\sum_{i=1}^m(h(x^{(i)})-y^{(i)})x^{(i)}_j
$$
根据梯度下降的更新规则：
$$
w_j:=w_j-\alpha\frac{\partial}{\partial w_j}J(w_0, w_1, \dots, w_n)
$$
将样本代入，不断迭代，就可以求到对应最优解。



## 模型评估方法

*   均方误差(Mean Squared Error MSE)：
    $$
    MSE=\frac 1n\sum_{i=1}^n (y_i - \hat y_i)^2
    $$

     

*   平均绝对值误差(Mean Absolute Error MAE)：
    $$
    MAE =\frac1n\sum_{i=1}^n|y_i-\hat y_i|
    $$

*   均方根误差(Root Mean Squered Error RMSE):
    $$
    RMSE=\sqrt{\frac 1n\sum_{i=1}^n (y_i - \hat y_i)^2}
    $$

绝大多数情况下：$RMSE > MAE$，因为 RMSE 有一个平方项，会放大误差

*   MAE 和 RMSE 都能对模型进行评估
*   RMSE 对异常值更敏感


## 欠拟合和过拟合

![Code_Generated_Image](https://cdn.jsdelivr.net/gh/TokeyTuT/my-image-storage@main/img/Code_Generated_Image.png)

观察上图：

*   当模型简单的时候，测试误差和训练误差都很大 —— 这叫欠拟合
*   当模型越来越复杂的时候，模型在训练集上表示不好，在测试集上表现好 —— 这叫过拟合
*   欠拟合和过拟合在训练集误差和都比较大

具体案例：

```python
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# 准备数据

# 指定随机种子
np.random.seed(115)

x = np.random.uniform(-3, 3, 100) # 模拟生成一百个数据
y  = 0.5 ** x ** 2+ x + 2 + np.random.normal(0, 1,100) # 添加噪声 e



estimator = LinearRegression()
estimator.fit(x.reshape(-1, 1), y)

y_predict = estimator.predict(x.reshape(-1, 1))

# 模型评估
print(f'均方误差{mean_squared_error(y, y_predict)}')

# 绘制散点图
plt.scatter(x, y)
plt.plot(x, y_predict, color='r') # 以折线图的形式模拟预测值
plt.show()
```

生成的图像如下：

![image-20260426183854371](https://cdn.jsdelivr.net/gh/TokeyTuT/my-image-storage@main/img/image-20260426183854371.png)

可以发现，拟合出来的数据直线与原数据相差比较大，说明这个模型对测试集拟合不好。这就体现了**欠拟合**

*   如何优化模型呢？
    *   实际上，我们上面的线性回归只用了 ***x* 的一次方这一个特征**，但是我们的函数实际上是一个二次函数，我们可以考虑给测试集添加一列特征 ——  $x^2$这一个特征，进行多元线性回归

```python
def dm2():
    # 1. 准备数据
    np.random.seed(115)
    x = np.random.uniform(-3, 3, 100)
    # 模拟一个二次方程关系：y = 0.5x^2 + x + 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

    # 2. 增加特征维度 (重要：LinearRegression 要求输入是二维矩阵)
    # 使用 np.column_stack 将 x 和 x^2 组合，形状变为 (100, 2)
    x_features = np.column_stack([x, x ** 2])

    # 3. 训练模型
    estimator = LinearRegression()
    estimator.fit(x_features, y)

    # 4. 预测
    y_predict = estimator.predict(x_features)

    # 5. 模型评估
    print(f'均方误差 (MSE): {mean_squared_error(y, y_predict):.4f}')

    # 6. 绘制结果
    plt.scatter(x, y, alpha=0.6, label='Original Data')

    # 注意：为了让绘出的回归线平滑，需要根据 x 的大小对预测值进行排序
    sort_idx = np.argsort(x)
    plt.plot(x[sort_idx], y_predict[sort_idx], color='r', label='Polynomial Fit')

    plt.title("Polynomial Regression (Degree 2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
```

![image-20260426185510648](https://cdn.jsdelivr.net/gh/TokeyTuT/my-image-storage@main/img/image-20260426185510648.png)

**注意**：

*   增加一列特征时，由于我们要传给模型的参数时一个(100,2)形状的数组，所以我们需要通过`np.column_stack([x, x ** 2])`对 x 进行预处理，这个函数可以直接添加一列并且将 x 转为形状为*(100,2)*的数组

*   在画图前，要对数据进行索引排序 —— 因为 plt 在绘制折线图时，是直接按照(x,y)的顺序进行连线，而不会根据他的大小进行连线。

至于这个排序的语法，详细讲解如下

>   理解这个排序的逻辑其实非常重要，因为在绘图（Matplotlib）和数据处理（NumPy/Pandas）中，这种“通过索引排序”的操作非常常见。
>
>   简单来说，`np.argsort(x)` 并不直接对 `x` 进行排序，而是返回**“如果要把 `x` 排好序，索引应该是什么样子”**。
>
>   我们可以通过一个简单的例子拆解一下：
>
>   ### 1. 拆解步骤
>
>   假设我们有以下数据：
>
>   Python
>
>   ```
>   x = np.array([10, 30, 20])
>   y_predict = np.array([1, 3, 2]) # 假设这是对应的预测值
>   ```
>
>   #### 第一步：获取排序索引 (`np.argsort`)
>
>   Python
>
>   ```
>   sort_idx = np.argsort(x) 
>   # 结果是 [0, 2, 1]
>   # 解释：x 最小的是 10（索引0），其次是 20（索引2），最大是 30（索引1）
>   ```
>
>   #### 第二步：利用索引重新排列
>
>   当你把这个 `sort_idx` 传回数组时，它会按照指定的顺序提取元素：
>
>   Python
>
>   ```
>   x_sorted = x[sort_idx]         # 得到 [10, 20, 30]
>   y_sorted = y_predict[sort_idx] # 得到 [1, 2, 3]
>   ```
>
>   ------
>
>   ### 2. 为什么要这么做？（绘图逻辑）
>
>   Matplotlib 的 `plt.plot()` 是一个**“连线抽样器”**。它会按照你给出的点序列，一个接一个地连线。
>
>   -   **如果不排序**：如果你的 `x` 是随机的（比如 `[3, -1, 2]`），Matplotlib 会先连到 `x=3` 的点，再猛地拉回到 `x=-1`，再跳到 `x=2`。这就会导致画面上出现杂乱交叉的直线，像乱麻一样。
>   -   **如果排序后**：点是按照从左到右（$x$ 轴从小到大）的顺序排列的，线段就会平滑地连接起来，形成我们想要的曲线。
>
>   ------
>
>   ### 3. 代码里的语法细节
>
>   Python
>
>   ```
>   plt.plot(x[sort_idx], y_predict[sort_idx], color='r')
>   ```
>
>   -   `x[sort_idx]`：确保 $x$ 轴是从坐标系左边画到右边。
>   -   `y_predict[sort_idx]`：确保每一个 $y$ 值依然对应它原本的那个 $x$。



上面的代码是正好拟合，那么过拟合的图像应该长什么样子呢？

```python
def dm3():
    # 1. 准备数据
    np.random.seed(115)
    x = np.random.uniform(-3, 3, 100)
    # 模拟一个二次方程关系：y = 0.5x^2 + x + 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

    # 2. 增加特征维度 (重要：LinearRegression 要求输入是二维矩阵)
    # 使用 np.column_stack 将 x 和 x^2 组合，形状变为 (100, 2)
    x_features = np.column_stack([x, x ** 2,x ** 3,x ** 4,x ** 5,x ** 6,x ** 7,x ** 8,x ** 9,x ** 10,x ** 11,x ** 12,x ** 13,x ** 14,x ** 15,x ** 16,x ** 17,x ** 18,x ** 19,x ** 20])

    # 3. 训练模型
    estimator = LinearRegression()
    estimator.fit(x_features, y)

    # 4. 预测
    y_predict = estimator.predict(x_features)

    # 5. 模型评估
    print(f'均方误差 (MSE): {mean_squared_error(y, y_predict):.4f}')

    # 6. 绘制结果
    plt.scatter(x, y, alpha=0.6, label='Original Data')

    # 注意：为了让绘出的回归线平滑，需要根据 x 的大小对预测值进行排序
    sort_idx = np.argsort(x)
    plt.plot(x[sort_idx], y_predict[sort_idx], color='r', label='Polynomial Fit')

    plt.title("Polynomial Regression (Degree 2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
```

![image-20260426190552227](https://cdn.jsdelivr.net/gh/TokeyTuT/my-image-storage@main/img/image-20260426190552227.png)

由于我们给的特征值太多了，这使得模型太过复杂，这就是过拟合

## 逻辑回归
对于线性回归来说，我们解决的是标签值连续的问题，它并不能解决**分类问题**，用回归的思想解决分类问题呢？我们引出了**逻辑回归**。 —— 这里我们先只考虑二分类问题（比如某股市明天是否涨、明天是否下雨这种只有 0 和 1 两个答案的问题）




逻辑回归同样要用到线性回归，但是线性回归问题得到的值值域是属于 $(-\infty,+\infty)$ 的，我们想要尽可能把这个值映射到 $[0,1]$ 中 。

我们考虑这样一个函数 $y = \frac{1}{1+e^{-x}}$，它的图像应该是：

![image-20260511151319988](https://github.com/TokeyTuT/my-image-storage/blob/main/img/image-20260511151319988.png?raw=true)

这个函数有一个很好的性质，他可以把所有 $(-\infty,+\infty)$ 的数都映射到 $[0,1]$ 上，这个函数叫做 *Sigmoid* 激活函数，记作 $\sigma(z)=\frac 1{1+e^{-z}}$

并且，观察这个函数，它的一阶导函数为: $\sigma(z)'=\sigma(z)(1-\sigma(z))$



于是，我们可以得到逻辑回归的原理：通过线性回归处理特征值之后的值，再根据*Sigmoid* 激活函数映射到 $[0,1]$ 上，我们把这个映射的值成为***概率*** ，再结合阈值，划分正负样本

关于概率与阈值，这里做几个描述：

>   1.   如果我们把二分类两个类记为 A、B， 分别代表 1 、 0 
>   2.   如果$\sigma(z) ≥ \mu$，认为预测结果是 A，防止认为预测结果为 B
>   3.   这里计算出的概率指的是**发生 A 的概率**，也就是 $P(y_i = 1|\mathbf x_i)$




那么我们使用什么作为这个回归的损失函数呢？

如果我们依旧考虑 MSE，也就是均方误差，我们可以尝试一下：

设线性回归方程为$\hat z_i=w^\top \mathbf x_i$，那么预测值概率为： $\sigma(\hat z_i) = \frac 1{1+e^{-(w^\top \mathbf x_i)}}$

那么我们可以得到 MSE:
$$
J_{MSE}(w) = \frac1{m}\sum_{i=1}^m(\hat y_i-y_i)^2 \\
J_{MSE}(w) = \frac1{m}\sum_{i=1}^m\left(\sigma(z)-y_i\right)^2
$$
对 $w$ 求偏导
$$
\frac{\partial J_{MSE}(w)}{\partial w} = \frac 1{m}\sum_{i=1}^m(\sigma(z)-y_i)\sigma(z)'\mathbf x_i
$$
由 *Sigmoid*激活函数的导数性质:
$$
\frac{\partial J_{MSE}(w)}{\partial w} = \frac 1{m}\sum_{i=1}^m(\sigma(z)-y_i)\sigma(z)(1-\sigma(z))\mathbf x_i
$$
观察上面到导函数，标签值$y_i$ 只有两种情况 0 或 1，但是$\sigma(z)$分布在 $(0,1)$ 之间，接着，我们考虑下面这种极端情况：

>   *   标签值为 1 时，预测值出现极大偏差，预测到接近为 0，也就是 $\sigma(z) \to 0$，此时$\frac{\partial J_{MSE}(w)}{\partial w} \to 0$
>   *   标签值为 0 时，预测值出现极大偏差，预测到接近为 1，也就是 $\sigma(z) \to 1$，此时$\frac{\partial J_{MSE}(w)}{\partial w} \to 0$

可以发现，如果预测值预测到了一个完全相反的数，它的导函数还是等于 0，但我进行下降的时候，导函数等于 0 的时候就是停止迭代，那么梯度下降很有可能会陷入完全相反的预测值中，导致损失函数出现了**严重的梯度消失**。
同时，由于这个导函数基本上都集中在 0上，损失函数的原函数图像会在一个区间内非常平缓，以至于我们不能分辨哪个模型参数合适。也就是，由于损失函数变为**非凸函数**。

综上所述，由于**损失函数变成非凸（Non-convex）**，以及**严重的梯度消失（Vanishing Gradient）**。所以使用 MSE 套用在逻辑回归中的损失函数并不适合，那我们就得考虑一种新的损失函数估计方法。



对于一个样本 $(\mathbf x,y)$，其中 $y \in \{0,1\}$，假设样本属于某个类别的概率服从**伯努利分布**：

*   当 $y=1$ 时，概率为 $P(y=1|x) = \hat{y}$

*   当 $y=0$ 时，概率为 $P(y=0|x) = 1 - \hat{y}$

仔细观察，我们其实可以把这两个函数合并成一个函数，方便计算：
$$
P(y|x) = \hat{y}^y \cdot (1 - \hat{y})^{(1-y)}
$$
这个函数很优美的描述了上面这个伯努利分布

如果我们想要模型想要预测的最准确，也就是所有样本 $P(y_i|x)$ 同时发生的概率最大，这是一个条件概率，由于每一个事件都是独立的，所以它们同时预测正确的概率为：
$$
H(w) = \prod_{i=1}^{m} P(y^{(i)} | x^{(i)}) = \prod_{i=1}^{m} (\hat{y}^{(i)})^{y^{(i)}} (1 - \hat{y}^{(i)})^{(1-y^{(i)})}
$$
为了方便计算，我们对等式两边同时取对数：
$$
log(H(w)) =\sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]
$$
根据**最大似然估计**，我们的目标是找到一组参数 $(w)$，使得观测到这组数据的**可能性最大**。

在优化问题中，我们通常习惯求**极小值**而不是极大值。因此，我们在对数似然函数前加一个**负号**，并取平均值（为了让损失不受样本数量影响），这就得到了我们熟悉的**交叉熵损失函数**：
$$
J(w) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]
$$

## 

**交叉熵的梯度**实际上适合 MSE 一样的

下面给推导过程:
根据上面的公式，可以整理以下方程：
- 线性组合：$z = w x + b$
- Sigmoid 映射：$a = \sigma(z) = \frac{1}{1 + e^{-z}}$ （$a$ 即预测值 $\hat{y}$）
- 交叉熵损失：$L = - [y \ln a + (1 - y) \ln(1 - a)]

由链式法则：
$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

分别计算每部分导数:
$$
\begin{align}
\frac{\partial L}{\partial a} = \frac{a - y}{a(1 - a)}
\\
\frac{\partial a}{\partial z} = a(1-a)
\\
\frac{\partial z}{\partial w} = x
\end{align}
$$
最终只剩下
$$
\frac{\partial L}{\partial w} = (a - y)x
$$
因为 $a$ 就是我们的预测值 $\hat{y}$，所以单个样本的梯度就是：
$$\frac{\partial L}{\partial w} = (\hat{y} - y)x$$

这其实就和 MSE 中对 $\frac {\partial J}{\partial w_i}$ 一样了，所以实际上，交叉熵的梯度下降的公式和 MSE 的梯度下降的公式是一样的，甚至实现都是一样的


## 正则化
线性回归中，正则化是应对模型**过拟合**的核心技术。当模型过于复杂时，正则化在损失函数中引入惩罚项，限制参数的(权重)大小

#### 核心思想：结构风险最小化

标准的线性回归目标是最小化经验风险：
$$
J_{regularized}(\theta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \cdot \Omega(w)
$$
其中：

*   $\lambda \to 0$：退化为普通的线性回归
*   $\lambda \to +\infty$：参数将被压制趋向于 0，模型变得简单。

### Ridge Regression (L2 正则化)

岭回归的惩罚项是参数向量的 **L2 范数**（参数平方和）。
$$
\Omega(w) = \|w\|_2^2 = \sum_{j=1}^{mn} w_j^2
$$

$$
L(\theta) = \sum_{i=1}^{m}(y - w^Tx_i) + \lambda\|w \|_2^2
$$



**原理**：它会将参数推向 0，但不会使其真正等于 0。它保留了所有特征，但缩小了每个特征的影响力。

**适用场景**：特征之间存在多重共线性，或者你认为大部分特征都对结果有贡献。



### Lasso Regression(L1 正则化)

L1正则化惩罚项是参数向量的 **L1 范数**（参数绝对值之和）。

$$
\Omega(w) = \|w\|_1 = \sum_{j=1}^{m} |w_j|
$$

**原理**：由于 L1 范数在零点处是不可导的“尖角”，它产生的最优解往往会让很多不重要的特征系数**直接变为 0**。
**适用场景**：**特征选择**。当你怀疑特征中存在大量冗余或无关变量时，Lasso 能帮你筛选出最重要的特征。

实际应用中，还是用 L2 正则化比较多



![Gemini_Generated_Image_hwnnf2hwnnf2hwnn](https://cdn.jsdelivr.net/gh/TokeyTuT/my-image-storage@main/img/Gemini_Generated_Image_hwnnf2hwnnf2hwnn.png)





## 决策树

### ID3 决策树

**熵 (*Entropy*)** : 信息论中代表随机变量不确定度的变量

在数学上，如果一个离散随机变量 $X$ 有多个可能的结果，每个结果发生的概率为 $p(x_i)$，那么这个系统的香农熵 $H(X)$ 的公式定义为：
$$
H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)
$$


其中 \[p(x_i)\] 代表数据在类别中出现的概率， 



**信息增益(*Information Gain*）**:  特征 a 对训练集数据 D 的信息增益\[g(D,a)\]，定义为集合 D 的熵 \[H(D)\] 与特征值 a 给定条件下 D 的熵 \[H(D|a)\] 之差
$$
g(D,A) = H(D) - H(D|A)
$$

*   **$H(D)$：** 没切分前，整个数据集的**初始信息熵**（混乱度）。

*   **$H(D \mid A)$：** 用特征 $A$ 把数据集切分成好几个子集后，所有子集混乱度的**加权平均值**（也叫条件熵）。



**条件熵(*Conditional Entropy*)** : 
$$
H(D \mid A) = \sum_{i=1}^{v} \frac{\vert{}D_i\vert{}}{\vert{}D\vert{}} H(D_i)
$$


*   $\vert{}D\vert{}$：总样本数。

* $\vert{}D_i\vert{}$：第 $i$ 个子集的样本数。

*   $\frac{\vert{}D_i\vert{}}{\vert{}D\vert{}}$：第 $i$ 个子集占总人数的**权重（概率）**。

*   $H(D_i)$：第 $i$ 个子集**自己内部的信息熵**。





ID3 树构建流程：

1.   计算每个特征的信息增益
2.   使用信息增益最大的特征作为数据集 拆分为子集
3.   使用信息增益最大的特征作为决策树的第一个节点
4.   递归上述步骤



 #### 缺点

**ID3 决策树偏向于选择种类多的特征作为分类依据**，很容易导致过拟合

为此，我们引入了 C4.5 树



### C4.5树

C4.5 树和 ID3 树类似，但是 CD4.5 树在 ID3树的信息增益上乘上了一个惩罚系数，他们的结果叫做**信息增益率**，将信息增益率大的作为根节点

**信息增益率**
$$
Grain\_Ratio(D,A) = \frac{Grain(D,A)}{IV(A)}
$$
其中

*   $Grain(D,A)$ ：信息增益
*   $IV(A) = -\sum_{v=1}^n\frac{D^v}{D}Ent(\frac{D^v}{D})$，IV 特征熵

实际上，信息增益率的本质是：

*   特征的信息增益  / 特征的内在信息
*   对信息增益进行修正，映入一个惩罚系数
*   特征取值个数较多时，惩罚系数较小；特征值取值个数较少是，惩罚系数较大
*   惩罚系数：$\frac{1}{IV(A)}$

$$
\text{IV}(D) = -\sum_{i=1}^{v} \frac{\vert{}D_i\vert{}}{\vert{}D\vert{}} \log_2 \frac{\vert{}D_i\vert{}}{\vert{}D\vert{}}
$$

举个例子🌰：

![image-20260714090444510](https://cdn.jsdelivr.net/gh/TokeyTuT/my-image-storage@main/img/image-20260714090444510.png)

### CART 决策树

CARTt 决策树 (*Classification and Reagression Tree*)

*   CART 模型是一种决策树模型，就可以用于分类，也可以用回归
*   CART 回归树使用**平方误差最小化**策略
*   CART 分类生成树使用**基尼指数最小化**策略

同时，在主流的算法实现中，CART 决策树已经完全被**二叉树**统治，也就是问题会被分为二分类。

#### CART 分类树

**基尼值*Gini(D)***：从数据集 D 中随机抽取两个样本，其类别标记不一致的概率。基尼值越小，代表数据集 D 的纯度越高
$$
Gini(D) = \sum_{k=1}^{|y|}\sum_{k'≠k}p_kp_{k'} = 1-\sum_{k=1}^np_k
$$
**基尼指数*Gini_index(D)***：
$$
Gini\_index(D) =\sum_{v=1}^n\frac{D^v}{D}Gini(D^v)
$$
由于主流算法的 CART 树都是二叉树，所以如果遇到形如：

>   假设有一个特征是“学历”，包含 `[专科, 本科, 硕士, 博士]` 四个离散值。
>
>   **C4.5 算法：** 直接分出 4 个树叉（多叉树）。
>
>   **CART 算法：** 强行排列组合，把它们变成“二选一”的组合题。比如它会遍历以下组合：
>
>   -   `{专科} vs {本科, 硕士, 博士}`
>   -   `{本科} vs {专科, 硕士, 博士}`
>   -   `{专科, 本科} vs {硕士, 博士}`
>   -   …… CART 会计算哪种**组合方式**下的基尼系数（Gini）最低，然后挑出最好的那一种，把数据集一分为二。

### CART 回归树

回归树的特征值是一段连续的数，所以我们不能采用基尼值去度量，而采用**平方误差最小化**作为划分的依据

建树过程如下:
1. 遍历所有特征及所有切分点对于当前节点的数据集，算法会扫描每一个特征 $j$。针对特征 $j$，扫描它所有可能的切分值 $s$。利用值 $s$，将当前节点的数据集切分成两部分（二叉树）：左子区域 $R_1(j, s) = \{x \vert{} x^{(j)} \le s\}$右子区域 $R_2(j, s) = \{x \vert{} x^{(j)} > s\}$
2. 计算当前切分下的最佳输出值计算这两个子区域如果固定下来，各自的平均响应值（预测值）：$c_1 = \text{mean}(y_i \mid x_i \in R_1)$$c_2 = \text{mean}(y_i \mid x_i \in R_2)$
3. 计算最优切分特征 $j$ 和切分点 $s$我们的目标是让切分后的两个子区域的残差平方和最小。因此需要求解以下优化问题：
   $\min_{j, s} \left[ \sum_{x_i \in R_1(j, s)} (y_i - c_1)^2 + \sum_{x_i \in R_2(j, s)} (y_i - c_2)^2 \right]$
   
   算法会比对所有的特征和所有可能的切分值，挑出使上面这个式子总和最小的那一组 $(j, s)$。
4. 执行节点分裂根据找到的最优 $(j, s)$，将数据集正式切分为左右两个子节点。