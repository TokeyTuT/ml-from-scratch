# ml-from-scratch

用 Python 从零实现经典机器学习算法的练习仓库。

本项目以学习和复现基础算法为主，优先保留清晰、可读、便于调试的实现。
项目中的模型和方法参考了经典的 ml 包 `scikit-learning` 的实现以及实现风格
本项目在复现经典算法的同时，同时实现了一些基础的工具方法：网格搜索、交叉验证、标准化工具、模型评估工具 …
详情请见 [`utils`](utils)

## 目录

```text
ml-from-scratch/
├── 01_knn/
├── 02_linear_regression/
├── 03_logistic_regression/
├── 04_decision_tree/
├── 05_naive_bayes/
├── 06_kmeans/
├── 07_pca/
├── 08_neural_network/
├── utils/
├── datasets/
└── README.md
```

## 使用方式

建议先创建并激活本地虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装本仓库的本地工具包，便于各算法目录直接导入 `utils`：

```bash
python -m pip install -e .
```

运行语法检查：

```bash
python -m compileall .
```

运行某个算法示例，例如：

```bash
python 01_knn/demo.py
```

## 学习顺序

1. 阅读对应算法目录下的 `README.md`。
2. 根据“建议接口”和“实现步骤”手写核心模型。
3. 用 `demo.py` 或小数据集验证结果。
4. 将可复用的数据处理、评估指标放在 `utils/` 中。

## 说明

- `NOTE.md` 用于记录通用概念和学习笔记。
- `datasets/` 只存放小型样例数据；大型或本地数据不要提交到仓库。
- 本仓库是练习项目，文档会刻意保留实现思路，方便回看算法细节。
