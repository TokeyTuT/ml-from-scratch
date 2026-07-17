"""使用手写 DecisionTreeClassifier 分析 Titanic 生存数据。"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_tree_classifier import DecisionTreeClassifier


TRAIN_PATH = SCRIPT_DIR / "titanic_train.csv"
TEST_PATH = SCRIPT_DIR / "titanic_test.csv"
SUBMISSION_PATH = SCRIPT_DIR / "titanic_submission.csv"
RANDOM_SEED = 42


def load_csv(path, required_columns):
    """读取 CSV，并检查分析所需的字段。"""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = set(required_columns) - fieldnames
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path.name} 缺少字段: {missing_text}")
        return list(reader)


def parse_float(value):
    """把 CSV 字符串转换为浮点数，空字符串返回 None。"""
    value = value.strip()
    return None if value == "" else float(value)


class TitanicPreprocessor:
    """把 Titanic 的混合字段转换为决策树需要的数值矩阵。"""

    def __init__(self):
        self.age_median = None
        self.feature_names = [
            "Pclass",
            "Sex=female",
            "Age",
            "AgeMissing",
            "SibSp",
            "Parch",
            "FamilySize",
            "IsAlone",
            "CabinKnown",
        ]

    def fit(self, rows):
        """仅使用训练数据计算缺失值填充统计量。"""
        ages = [
            age
            for row in rows
            if (age := parse_float(row["Age"])) is not None
        ]

        if not ages:
            raise ValueError("训练数据不足，无法计算 Age 的填充值。")

        self.age_median = float(np.median(ages))
        return self

    def transform(self, rows):
        """将原始乘客记录转换为纯数值特征矩阵。"""
        if self.age_median is None:
            raise ValueError("TitanicPreprocessor must be fitted before transform.")

        transformed_rows = [self._transform_row(row) for row in rows]
        return np.asarray(transformed_rows, dtype=float)

    def fit_transform(self, rows):
        """拟合预处理参数并转换训练数据。"""
        return self.fit(rows).transform(rows)

    def _transform_row(self, row):
        """转换一条乘客记录。"""
        age = parse_float(row["Age"])
        age_missing = float(age is None)
        age = self.age_median if age is None else age

        sex = row["Sex"].strip().lower()
        if sex not in {"female", "male"}:
            raise ValueError(f"未知的 Sex 类别: {row['Sex']!r}")

        sib_sp = int(row["SibSp"])
        parch = int(row["Parch"])
        family_size = sib_sp + parch + 1

        return [
            int(row["Pclass"]),
            float(sex == "female"),
            age,
            age_missing,
            sib_sp,
            parch,
            family_size,
            float(family_size == 1),
            float(bool(row["Cabin"].strip())),
        ]


def stratified_split(rows, labels, test_size=0.2, random_seed=42):
    """按标签比例划分训练集和验证集。"""
    labels = np.asarray(labels)
    rng = np.random.default_rng(random_seed)
    train_indices = []
    validation_indices = []

    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        label_indices = rng.permutation(label_indices)
        validation_size = max(1, int(round(label_indices.size * test_size)))
        validation_indices.extend(label_indices[:validation_size])
        train_indices.extend(label_indices[validation_size:])

    train_indices = rng.permutation(train_indices)
    validation_indices = rng.permutation(validation_indices)

    training_rows = [rows[index] for index in train_indices]
    validation_rows = [rows[index] for index in validation_indices]
    return (
        training_rows,
        validation_rows,
        labels[train_indices],
        labels[validation_indices],
    )


def print_group_survival(rows, group_name, group_func):
    """打印某个分组下的乘客数量和生存率。"""
    grouped_labels = defaultdict(list)
    for row in rows:
        grouped_labels[group_func(row)].append(int(row["Survived"]))

    print(f"\n按{group_name}统计生存率:")
    for group, labels in sorted(grouped_labels.items(), key=lambda item: str(item[0])):
        print(
            f"  {group!s:<10} "
            f"人数={len(labels):>3}, 生存率={np.mean(labels):.2%}"
        )


def age_group(row):
    """把年龄划分为适合描述性分析的区间。"""
    age = parse_float(row["Age"])
    if age is None:
        return "年龄缺失"
    if age < 12:
        return "儿童(<12)"
    if age < 18:
        return "青少年(12-17)"
    if age < 60:
        return "成人(18-59)"
    return "老年(>=60)"


def print_data_overview(rows):
    """输出数据规模、缺失情况和主要分组生存率。"""
    labels = np.array([int(row["Survived"]) for row in rows])
    print("=== Titanic 数据概览 ===")
    print(f"样本数: {len(rows)}")
    print(f"生存人数: {labels.sum()}")
    print(f"总体生存率: {labels.mean():.2%}")

    print("\n主要字段缺失数:")
    for column in ("Age", "Cabin"):
        missing_count = sum(not row[column].strip() for row in rows)
        print(f"  {column:<8} {missing_count:>3} ({missing_count / len(rows):.2%})")

    print_group_survival(rows, "性别", lambda row: row["Sex"])
    print_group_survival(rows, "舱位等级", lambda row: f"Pclass={row['Pclass']}")
    print_group_survival(rows, "年龄段", age_group)


def binary_metrics(y_true, y_pred):
    """计算二分类准确率、精确率、召回率和混淆矩阵。"""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))

    accuracy = float(np.mean(y_true == y_pred))
    precision = true_positive / (true_positive + false_positive or 1)
    recall = true_positive / (true_positive + false_negative or 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": np.array(
            [[true_negative, false_positive], [false_negative, true_positive]]
        ),
    }


def tree_summary(root, feature_names):
    """统计树深、节点数、叶子数和各特征的使用次数。"""
    split_counts = Counter()

    def visit(node, depth):
        if node.feature_index is None:
            return depth, 1, 1

        split_counts[feature_names[node.feature_index]] += 1
        left_depth, left_nodes, left_leaves = visit(node.left, depth + 1)
        right_depth, right_nodes, right_leaves = visit(node.right, depth + 1)
        return (
            max(left_depth, right_depth),
            1 + left_nodes + right_nodes,
            left_leaves + right_leaves,
        )

    max_depth, node_count, leaf_count = visit(root, 0)
    return max_depth, node_count, leaf_count, split_counts


def print_model_report(model, feature_names, y_train, train_pred, y_valid, valid_pred):
    """输出训练集、验证集指标及树结构摘要。"""
    training_metrics = binary_metrics(y_train, train_pred)
    validation_metrics = binary_metrics(y_valid, valid_pred)
    baseline_accuracy = max(np.mean(y_valid == 0), np.mean(y_valid == 1))
    depth, node_count, leaf_count, split_counts = tree_summary(
        model._root,
        feature_names,
    )

    print("\n=== 决策树模型结果 ===")
    print(f"训练集准确率: {training_metrics['accuracy']:.2%}")
    print(f"验证集准确率: {validation_metrics['accuracy']:.2%}")
    print(f"多数类基线准确率: {baseline_accuracy:.2%}")
    print(f"验证集精确率: {validation_metrics['precision']:.2%}")
    print(f"验证集召回率: {validation_metrics['recall']:.2%}")
    print("验证集混淆矩阵 [[TN, FP], [FN, TP]]:")
    print(validation_metrics["confusion_matrix"])
    print(f"树深度: {depth}, 节点数: {node_count}, 叶子数: {leaf_count}")

    print("\n树中使用次数最多的特征（不是严格的 feature importance）:")
    for feature_name, count in split_counts.most_common(8):
        print(f"  {feature_name:<18} {count}")


def save_submission(passenger_ids, predictions, path):
    """保存 Kaggle Titanic 要求的提交格式。"""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["PassengerId", "Survived"])
        writer.writerows(zip(passenger_ids, predictions))


def main():
    required_features = {
        "PassengerId",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Cabin",
    }
    training_rows = load_csv(TRAIN_PATH, required_features | {"Survived"})
    test_rows = load_csv(TEST_PATH, required_features)
    labels = np.array([int(row["Survived"]) for row in training_rows])

    print_data_overview(training_rows)

    development_rows, validation_rows, y_development, y_validation = (
        stratified_split(
            training_rows,
            labels,
            test_size=0.2,
            random_seed=RANDOM_SEED,
        )
    )

    preprocessor = TitanicPreprocessor().fit(development_rows)
    X_development = preprocessor.transform(development_rows)
    X_validation = preprocessor.transform(validation_rows)

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=10,
    )
    model.fit(X_development, y_development)
    development_predictions = model.predict(X_development)
    validation_predictions = model.predict(X_validation)

    print(f"\n开发训练集: {len(development_rows)}, 验证集: {len(validation_rows)}")
    print(f"训练部分填充值: Age={preprocessor.age_median:.2f}")
    print_model_report(
        model,
        preprocessor.feature_names,
        y_development,
        development_predictions,
        y_validation,
        validation_predictions,
    )

    # 最终预测时重新使用完整训练集拟合预处理器和模型。
    final_preprocessor = TitanicPreprocessor().fit(training_rows)
    X_full = final_preprocessor.transform(training_rows)
    X_test = final_preprocessor.transform(test_rows)
    final_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=10,
    )
    final_model.fit(X_full, labels)
    test_predictions = final_model.predict(X_test).astype(int)
    passenger_ids = [int(row["PassengerId"]) for row in test_rows]
    save_submission(passenger_ids, test_predictions, SUBMISSION_PATH)

    print("\n=== Kaggle 测试集预测 ===")
    print(f"预测人数: {len(test_predictions)}")
    print(f"预测生存人数: {test_predictions.sum()}")
    print(f"预测生存率: {test_predictions.mean():.2%}")
    print(f"提交文件: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
