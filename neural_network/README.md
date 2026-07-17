# 08 Neural Network 实现指引

## 目标

从零实现一个最小神经网络，理解前向传播、损失函数、反向传播和梯度下降。

## 建议文件

- `layers.py`：Dense 层、激活层。
- `activations.py`：Sigmoid、ReLU、Softmax。
- `losses.py`：MSE 或交叉熵。
- `neural_network.py`：模型容器和训练循环。
- `demo.py`：拟合 XOR 或简单分类数据。
- `test_neural_network.py`：验证前向输出形状和梯度更新。

## 推荐先做的最小版本

先实现一个只支持如下结构的小网络：

```text
input -> Dense -> ReLU -> Dense -> Sigmoid
```

用于二分类。

## 核心概念

Dense 层前向：

```text
Z = XW + b
```

激活函数：

```text
ReLU(x) = max(0, x)
Sigmoid(x) = 1 / (1 + exp(-x))
```

反向传播要保存每一层前向过程中的输入，方便计算梯度。

## 实现步骤

1. 实现 Dense 层的 `forward`。
2. 实现 Dense 层的 `backward`，计算 `dW`、`db` 和传给前一层的梯度。
3. 实现 ReLU 和 Sigmoid 的前向、反向。
4. 实现损失函数。
5. 在训练循环中：
   - 前向传播得到预测。
   - 计算损失。
   - 从损失开始反向传播。
   - 根据梯度更新每层参数。
6. 用一个很小的数据集验证损失是否下降。

## 注意点

- 这是最容易写乱的部分，建议每次只实现一个组件。
- 先不要追求通用框架，先让一个小网络跑通。
- 检查每一层输入输出 shape。
- 随机初始化权重时不要全是 0。

## 最小验收

- 前向传播能返回正确 shape。
- 训练若干轮后 loss 下降。
- 能拟合 XOR 数据，或者在简单二分类数据上正确率较高。
