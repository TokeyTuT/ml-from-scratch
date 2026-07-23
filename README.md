# ml-from-scratch

一个用 Python 从零手写经典机器学习算法的练习仓库。

这个项目以理解算法原理和复现核心流程为主，不追求工程封装的复杂度。
代码会优先保持清晰、可读、便于调试，方便在练习过程中逐步补全、验证和回看。

部分接口设计会参考 `scikit-learn` 的使用习惯，同时在 `utils/` 中实现一些基础工具，
例如数据划分、交叉验证、网格搜索、标准化和模型评估等。更多说明见 [`utils/`](utils/)。

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
├── 08_adaboost/
├── 09_gbdt/
├── 10_xgboost/
├── utils/
└── README.md
```

各编号目录对应一个算法主题，目录内通常包含算法笔记、核心实现、示例脚本和测试文件。
共享的数据处理、评估指标或基类工具放在 `utils/`

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

运行快速语法检查：

```bash
python -m compileall .
```

运行测试：

```bash
python -m pytest
```

运行某个算法示例：

```bash
python 01_knn/demo.py
```

## 学习顺序

1. 阅读对应算法目录下的 `README.md`。
2. 根据“建议接口”和“实现步骤”手写核心模型。
3. 用 `demo.py`、小型样例数据或单元测试验证结果。
4. 将通用的数据处理、模型选择和评估逻辑沉淀到 `utils/` 中。
5. 在 `NOTE.md` 中记录容易混淆的概念、公式推导和调试结论。

## 说明

- `NOTE.md` 用于记录通用概念和学习笔记。
- 本仓库是学习练习项目，文档会刻意保留实现思路和推导线索，方便复盘算法细节。
- 除非明确需要完整实现，否则优先通过脚手架、测试、示例和代码 review 来推进练习。
